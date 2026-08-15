"""Nested ablation for selecting the smallest useful FKStock generator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from ..data.dataset import ProfileObservations
from ..evaluation import PredictionRecord, records_as_json
from ..stock_transformer.conservation import (
    ConservationConfig,
    ConservationLaw,
    ConservationLawDiscoverer,
    DensityProjector,
)
from ..walk_forward import WalkForwardConfig, build_walk_forward_folds
from .generator import FEATURE_NAMES, FKFeatureRow, FKGenerator, FKGeneratorConfig
from .pipeline import _rows_between, _select, _test_context
from .solver import FeynmanKacConfig, FeynmanKacPotentialConfig, FeynmanKacSolver


FEATURE_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("intercept-only", ()),
    *((f"single-{name}", (name,)) for name in FEATURE_NAMES),
    ("return-memory", ("rolling_drift", "latest_matured_return")),
    (
        "return-dynamics",
        ("rolling_drift", "latest_matured_return", "rolling_diffusion"),
    ),
    (
        "density-geometry",
        (
            "density_center_change",
            "density_spread_change",
            "density_gradient_energy",
        ),
    ),
    (
        "return-plus-density",
        (
            "rolling_drift",
            "latest_matured_return",
            "density_center_change",
            "density_spread_change",
        ),
    ),
    (
        "conservation-flow",
        (
            "signed_conservation_flow",
            "current_coherence",
            "stochastic_current_fraction",
        ),
    ),
    (
        "return-plus-flow",
        (
            "rolling_drift",
            "latest_matured_return",
            "signed_conservation_flow",
            "current_coherence",
            "stochastic_current_fraction",
        ),
    ),
    ("all-v1", FEATURE_NAMES),
)


@dataclass(frozen=True, slots=True)
class FKAblationConfig:
    log_loss_tolerance: float = 1e-3
    brier_tolerance: float = 5e-4

    def __post_init__(self) -> None:
        if self.log_loss_tolerance < 0 or self.brier_tolerance < 0:
            raise ValueError("ablation tolerances cannot be negative")


@dataclass(frozen=True, slots=True)
class FKAblationResult:
    selected_features: tuple[str, ...]
    selected_potential: str
    tuning_feature_results: tuple[dict[str, object], ...]
    tuning_potential_results: tuple[dict[str, object], ...]
    outer_feature_results: tuple[dict[str, object], ...]
    outer_potential_results: tuple[dict[str, object], ...]
    selected_predictions: tuple[PredictionRecord, ...]
    fold: dict[str, object]


def run_fk_stock_ablation(
    observations: Sequence[ProfileObservations],
    conservation_config: ConservationConfig,
    generator_config: FKGeneratorConfig,
    solver_config: FeynmanKacConfig,
    potential_config: FeynmanKacPotentialConfig,
    walk_forward_config: WalkForwardConfig,
    *,
    output_dir: str | Path,
    dataset_id: str,
    ablation_config: FKAblationConfig | None = None,
) -> FKAblationResult:
    config = ablation_config or FKAblationConfig()
    folds = build_walk_forward_folds(observations, walk_forward_config)
    if len(folds) != 1:
        raise ValueError("nested FK ablation requires exactly one outer fold")
    fold = folds[0]
    context = generator_config.drift_lookback_sessions + 2
    row_builder = FKGenerator(replace(generator_config, active_features=FEATURE_NAMES))

    fit_observations = _select(observations, target_before=fold.tuning_start)
    tuning_law = _fit_law(fit_observations, conservation_config)
    fit_rows = row_builder.build_rows(
        fit_observations, tuning_law, conservation_config
    )
    tune_observations = _select(
        observations,
        signal_at_or_after=fold.tuning_start,
        signal_before=fold.test_start,
        target_before=fold.test_start,
        context=context,
    )
    tune_rows = _rows_between(
        row_builder.build_rows(tune_observations, tuning_law, conservation_config),
        fold.tuning_start,
        fold.test_start,
    )
    tuning_feature_results = _evaluate_feature_candidates(
        fit_rows, tune_rows, generator_config, solver_config
    )
    selected_features = _select_features(tuning_feature_results, config)
    potential_candidates = _potential_candidates(potential_config)
    tuning_potential_results = _evaluate_potential_candidates(
        fit_rows,
        tune_rows,
        generator_config,
        solver_config,
        selected_features,
        potential_candidates,
    )
    selected_potential = _select_potential(tuning_potential_results, config)

    pretest_observations = _select(observations, target_before=fold.test_start)
    final_law = _fit_law(pretest_observations, conservation_config)
    pretest_rows = row_builder.build_rows(
        pretest_observations, final_law, conservation_config
    )
    test_observations = _test_context(observations, fold, context)
    test_rows = _rows_between(
        row_builder.build_rows(test_observations, final_law, conservation_config),
        fold.test_start,
        fold.test_end,
    )
    outer_feature_results = _evaluate_feature_candidates(
        pretest_rows, test_rows, generator_config, solver_config
    )
    outer_potential_results = _evaluate_potential_candidates(
        pretest_rows,
        test_rows,
        generator_config,
        solver_config,
        selected_features,
        potential_candidates,
    )
    selected_potential_config = dict(potential_candidates)[selected_potential]
    selected_generator = FKGenerator(
        replace(generator_config, active_features=selected_features)
    )
    selected_posterior = selected_generator.fit(pretest_rows)
    selected_predictions = tuple(
        FeynmanKacSolver(solver_config, selected_potential_config).predict(
            row, selected_posterior
        )[0]
        for row in test_rows
    )
    fold_payload: dict[str, object] = {
        "dataset_id": dataset_id,
        "tuning_start": str(fold.tuning_start),
        "test_start": str(fold.test_start),
        "test_end": str(fold.test_end),
        "fit_rows": len(fit_rows),
        "tuning_rows": len(tune_rows),
        "pretest_rows": len(pretest_rows),
        "outer_rows": len(test_rows),
        "selection_rule": {
            "log_loss_tolerance": config.log_loss_tolerance,
            "brier_tolerance": config.brier_tolerance,
            "tie_break": "fewest active features; no potential unless material gain",
        },
    }
    result = FKAblationResult(
        selected_features=selected_features,
        selected_potential=selected_potential,
        tuning_feature_results=tuple(tuning_feature_results),
        tuning_potential_results=tuple(tuning_potential_results),
        outer_feature_results=tuple(outer_feature_results),
        outer_potential_results=tuple(outer_potential_results),
        selected_predictions=selected_predictions,
        fold=fold_payload,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_features": list(result.selected_features),
        "selected_potential": result.selected_potential,
        "tuning_feature_results": list(result.tuning_feature_results),
        "tuning_potential_results": list(result.tuning_potential_results),
        "outer_feature_results": list(result.outer_feature_results),
        "outer_potential_results": list(result.outer_potential_results),
        "fold": result.fold,
        "conservation_config": asdict(conservation_config),
        "generator_config": asdict(generator_config),
        "solver_config": asdict(solver_config),
        "potential_config": asdict(potential_config),
    }
    (destination / "ablation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "predictions.json").write_text(
        json.dumps(records_as_json(selected_predictions), ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _fit_law(
    observations: Sequence[ProfileObservations], config: ConservationConfig
) -> ConservationLaw:
    projector = DensityProjector(config)
    series = [
        [
            projector.project(
                item.symbol,
                item.signal_times[index],
                item.profiles[index][:, 3],
                item.profiles[index][:, 1],
            )
            for index in range(len(item))
        ]
        for item in observations
    ]
    return ConservationLawDiscoverer(config).fit(series)


def _evaluate_feature_candidates(
    fit_rows: Sequence[FKFeatureRow],
    evaluation_rows: Sequence[FKFeatureRow],
    generator_config: FKGeneratorConfig,
    solver_config: FeynmanKacConfig,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for name, features in FEATURE_CANDIDATES:
        results.append(
            _evaluate_candidate(
                name,
                features,
                fit_rows,
                evaluation_rows,
                generator_config,
                solver_config,
            )
        )
    return results


def _evaluate_potential_candidates(
    fit_rows: Sequence[FKFeatureRow],
    evaluation_rows: Sequence[FKFeatureRow],
    generator_config: FKGeneratorConfig,
    solver_config: FeynmanKacConfig,
    features: tuple[str, ...],
    candidates: tuple[tuple[str, FeynmanKacPotentialConfig | None], ...],
) -> list[dict[str, object]]:
    return [
        _evaluate_candidate(
            name,
            features,
            fit_rows,
            evaluation_rows,
            generator_config,
            solver_config,
            potential_config=potential,
        )
        for name, potential in candidates
    ]


def _evaluate_candidate(
    name: str,
    features: tuple[str, ...],
    fit_rows: Sequence[FKFeatureRow],
    evaluation_rows: Sequence[FKFeatureRow],
    generator_config: FKGeneratorConfig,
    solver_config: FeynmanKacConfig,
    *,
    potential_config: FeynmanKacPotentialConfig | None = None,
) -> dict[str, object]:
    generator = FKGenerator(replace(generator_config, active_features=features))
    posterior = generator.fit(fit_rows)
    solver = FeynmanKacSolver(solver_config, potential_config)
    predictions = [solver.predict(row, posterior)[0] for row in evaluation_rows]
    return {
        "name": name,
        "features": list(features),
        "feature_count": len(features),
        **_metrics(predictions),
    }


def _metrics(predictions: Sequence[PredictionRecord]) -> dict[str, float | int]:
    probability = np.asarray([item.probability_mean for item in predictions])
    targets = np.asarray([item.target for item in predictions], dtype=np.int64)
    returns = np.asarray([item.forward_return for item in predictions])
    expected = np.asarray([item.expected_return for item in predictions])
    predicted = probability.argmax(axis=1)
    support = np.bincount(targets, minlength=4)
    recall = np.asarray(
        [
            (predicted[targets == state] == state).mean() if support[state] else 0.0
            for state in range(4)
        ]
    )
    return {
        "observations": len(predictions),
        "accuracy": float((predicted == targets).mean()),
        "balanced_accuracy": float(recall.mean()),
        "log_loss": float(
            -np.log(np.clip(probability[np.arange(len(targets)), targets], 1e-12, 1.0)).mean()
        ),
        "brier_score": float(
            np.square(np.eye(4)[targets] - probability).sum(axis=1).mean()
        ),
        "return_rmse": float(np.sqrt(np.square(returns - expected).mean())),
        "return_correlation": float(
            np.corrcoef(returns, expected)[0, 1]
            if returns.std() > 0 and expected.std() > 0
            else 0.0
        ),
    }


def _select_features(
    results: Sequence[dict[str, object]], config: FKAblationConfig
) -> tuple[str, ...]:
    best_log_loss = min(float(item["log_loss"]) for item in results)
    best_brier = min(float(item["brier_score"]) for item in results)
    eligible = [
        item
        for item in results
        if float(item["log_loss"]) <= best_log_loss + config.log_loss_tolerance
        and float(item["brier_score"]) <= best_brier + config.brier_tolerance
    ]
    selected = min(
        eligible,
        key=lambda item: (
            int(item["feature_count"]),
            float(item["log_loss"]),
            float(item["brier_score"]),
        ),
    )
    return tuple(str(name) for name in selected["features"])


def _select_potential(
    results: Sequence[dict[str, object]], config: FKAblationConfig
) -> str:
    baseline = next(item for item in results if item["name"] == "no-potential")
    alternatives = [item for item in results if item["name"] != "no-potential"]
    best = min(alternatives, key=lambda item: float(item["log_loss"]))
    material = (
        float(best["log_loss"])
        < float(baseline["log_loss"]) - config.log_loss_tolerance
        and float(best["brier_score"])
        < float(baseline["brier_score"]) - config.brier_tolerance
    )
    return str(best["name"] if material else "no-potential")


def _potential_candidates(
    config: FeynmanKacPotentialConfig,
) -> tuple[tuple[str, FeynmanKacPotentialConfig | None], ...]:
    return (
        ("no-potential", None),
        (
            "epistemic-only",
            replace(
                config,
                potential_floor=0.0,
                noise_potential_weight=0.0,
                uncertainty_potential_weight=1.0,
                incoherence_potential_weight=0.0,
            ),
        ),
        (
            "flow-only",
            replace(
                config,
                potential_floor=0.0,
                uncertainty_potential_weight=0.0,
            ),
        ),
        ("full-v1", config),
    )
