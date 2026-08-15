"""Strict walk-forward orchestration for conservation-conditioned FKStock."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..data.dataset import ProfileObservations
from ..evaluation import PredictionRecord, estimate_regime_return_moments, records_as_json
from ..stock_transformer.conservation import (
    ConservationConfig,
    ConservationLaw,
    ConservationLawDiscoverer,
    DensityProjector,
)
from ..walk_forward import WalkForwardConfig, WalkForwardFold, build_walk_forward_folds
from .generator import BayesianGeneratorPosterior, FKFeatureRow, FKGenerator, FKGeneratorConfig
from .solver import FeynmanKacConfig, FeynmanKacSolver


@dataclass(frozen=True, slots=True)
class FKStockPipelineConfig:
    training_scope: str = "sector-pooled"

    def __post_init__(self) -> None:
        if self.training_scope != "sector-pooled":
            raise ValueError("FKStock currently requires sector-pooled training")


@dataclass(frozen=True, slots=True)
class FKWalkForwardResult:
    folds: tuple[dict[str, object], ...]
    predictions: tuple[PredictionRecord, ...]
    diagnostics: tuple[dict[str, object], ...]
    regime_return_means: dict[str, tuple[float, ...]]
    regime_return_variances: dict[str, tuple[float, ...]]


def run_fk_stock_walk_forward(
    observations: Sequence[ProfileObservations],
    conservation_config: ConservationConfig,
    generator_config: FKGeneratorConfig,
    solver_config: FeynmanKacConfig,
    pipeline_config: FKStockPipelineConfig,
    walk_forward_config: WalkForwardConfig,
    *,
    output_dir: str | Path,
    dataset_id: str | None = None,
) -> FKWalkForwardResult:
    if not observations:
        raise ValueError("FKStock requires profile observations")
    if pipeline_config.training_scope == "sector-pooled" and not dataset_id:
        raise ValueError("FKStock sector training requires a dataset id")
    folds = build_walk_forward_folds(observations, walk_forward_config)
    if not folds:
        raise ValueError("dataset does not contain enough common sessions")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    generator = FKGenerator(generator_config)
    solver = FeynmanKacSolver(solver_config)
    selected: dict[tuple[str, int], PredictionRecord] = {}
    selected_diagnostics: dict[tuple[str, int], dict[str, object]] = {}
    summaries: list[dict[str, object]] = []
    context = generator_config.drift_lookback_sessions + 2

    for fold in folds:
        fit_observations = _select(observations, target_before=fold.tuning_start)
        tuning_law, tuning_posterior = _fit_generator(
            fit_observations, conservation_config, generator
        )
        tune_observations = _select(
            observations,
            signal_at_or_after=fold.tuning_start,
            signal_before=fold.test_start,
            target_before=fold.test_start,
            context=context,
        )
        tune_rows = _rows_between(
            generator.build_rows(tune_observations, tuning_law, conservation_config),
            fold.tuning_start,
            fold.test_start,
        )
        if not tune_rows:
            raise ValueError(f"FKStock fold {fold.index} has empty tuning data")
        tuning_predictions, tuning_diagnostics = _predict(
            tune_rows, tuning_posterior, solver
        )

        pretest_observations = _select(observations, target_before=fold.test_start)
        final_law, final_posterior = _fit_generator(
            pretest_observations, conservation_config, generator
        )
        test_observations = _test_context(observations, fold, context)
        test_rows = _rows_between(
            generator.build_rows(test_observations, final_law, conservation_config),
            fold.test_start,
            fold.test_end,
        )
        if not test_rows:
            raise ValueError(f"FKStock fold {fold.index} has empty outer data")
        predictions, diagnostics = _predict(test_rows, final_posterior, solver)
        for prediction, diagnostic in zip(predictions, diagnostics, strict=True):
            key = (prediction.symbol, int(prediction.signal_time.astype(np.int64)))
            selected.setdefault(key, prediction)
            selected_diagnostics.setdefault(key, diagnostic)

        fold_dir = destination / f"fold-{fold.index:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model_payload = {
            "format_version": 1,
            "strategy": "fk-stock",
            "dataset_id": dataset_id,
            "training_cutoff": str(fold.test_start),
            "conservation_config": asdict(conservation_config),
            "conservation_law": final_law.as_json(),
            "generator_config": asdict(generator_config),
            "generator_posterior": final_posterior.as_json(),
            "solver_config": asdict(solver_config),
        }
        (fold_dir / "model.json").write_text(
            json.dumps(model_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary: dict[str, object] = {
            "strategy": "fk-stock",
            "symbol": f"sector:{dataset_id}",
            "dataset_id": dataset_id,
            "fold": fold.index,
            "tuning_start": str(fold.tuning_start),
            "test_start": str(fold.test_start),
            "test_end": str(fold.test_end),
            "training_cutoff": str(fold.test_start),
            "generator_fit_rows": final_posterior.fit_rows,
            "tuning_predictions": len(tuning_predictions),
            "outer_test_predictions": len(predictions),
            "tuning_metrics": _prediction_metrics(tuning_predictions),
            "tuning_field_diagnostics": _diagnostic_summary(tuning_diagnostics),
            "outer_field_diagnostics": _diagnostic_summary(diagnostics),
            "conservation_law": final_law.as_json(),
        }
        (fold_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summaries.append(summary)

    ordered_keys = sorted(selected, key=lambda key: (selected[key].signal_time, key[0]))
    predictions = tuple(selected[key] for key in ordered_keys)
    diagnostics = tuple(selected_diagnostics[key] for key in ordered_keys)
    (destination / "predictions.json").write_text(
        json.dumps(records_as_json(predictions), ensure_ascii=False), encoding="utf-8"
    )
    (destination / "fk-diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    means, variances = _fixed_training_moments(observations, folds[0].test_start)
    return FKWalkForwardResult(
        folds=tuple(summaries),
        predictions=predictions,
        diagnostics=diagnostics,
        regime_return_means=means,
        regime_return_variances=variances,
    )


def _fit_generator(
    observations: Sequence[ProfileObservations],
    conservation_config: ConservationConfig,
    generator: FKGenerator,
) -> tuple[ConservationLaw, BayesianGeneratorPosterior]:
    projector = DensityProjector(conservation_config)
    series = []
    for item in observations:
        series.append(
            [
                projector.project(
                    item.symbol,
                    item.signal_times[index],
                    item.profiles[index][:, 3],
                    item.profiles[index][:, 1],
                )
                for index in range(len(item))
            ]
        )
    law = ConservationLawDiscoverer(conservation_config).fit(series)
    rows = generator.build_rows(observations, law, conservation_config)
    return law, generator.fit(rows)


def _predict(
    rows: Sequence[FKFeatureRow],
    posterior: BayesianGeneratorPosterior,
    solver: FeynmanKacSolver,
) -> tuple[list[PredictionRecord], list[dict[str, object]]]:
    records: list[PredictionRecord] = []
    diagnostics: list[dict[str, object]] = []
    for row in rows:
        record, fields = solver.predict(row, posterior)
        records.append(record)
        diagnostics.append(
            {"symbol": row.symbol, "signal_time": str(row.signal_time), **fields}
        )
    return records, diagnostics


def _rows_between(
    rows: Sequence[FKFeatureRow], start: np.datetime64, end: np.datetime64
) -> list[FKFeatureRow]:
    return [row for row in rows if start <= row.signal_time < end]


def _prediction_metrics(predictions: Sequence[PredictionRecord]) -> dict[str, float]:
    probabilities = np.asarray([item.probability_mean for item in predictions])
    targets = np.asarray([item.target for item in predictions], dtype=np.int64)
    return {
        "accuracy": float((probabilities.argmax(axis=1) == targets).mean()),
        "log_loss": float(
            -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)).mean()
        ),
        "brier_score": float(
            np.square(np.eye(4, dtype=np.float64)[targets] - probabilities)
            .sum(axis=1)
            .mean()
        ),
    }


def _diagnostic_summary(rows: Sequence[dict[str, object]]) -> dict[str, float]:
    names = (
        "raw_drift",
        "effective_drift",
        "diffusion_variance",
        "epistemic_variance",
        "effective_potential",
        "survival_factor",
        "stochastic_current_fraction",
        "current_coherence",
        "continuity_error",
    )
    result: dict[str, float] = {}
    for name in names:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        result[f"{name}_mean"] = float(values.mean())
        result[f"{name}_std"] = float(values.std())
    return result


def _fixed_training_moments(
    observations: Sequence[ProfileObservations], cutoff: np.datetime64
) -> tuple[dict[str, tuple[float, ...]], dict[str, tuple[float, ...]]]:
    selected = _select(observations, target_before=cutoff)
    pooled_mean, pooled_variance = estimate_regime_return_moments(selected)
    means: dict[str, tuple[float, ...]] = {}
    variances: dict[str, tuple[float, ...]] = {}
    for item in selected:
        mean, variance = estimate_regime_return_moments((item,))
        means[item.symbol] = tuple(float(value) for value in mean)
        variances[item.symbol] = tuple(float(value) for value in variance)
    for item in observations:
        means.setdefault(item.symbol, tuple(float(value) for value in pooled_mean))
        variances.setdefault(
            item.symbol, tuple(float(value) for value in pooled_variance)
        )
    return means, variances


def _select(
    observations: Sequence[ProfileObservations],
    *,
    signal_at_or_after: np.datetime64 | None = None,
    signal_before: np.datetime64 | None = None,
    target_before: np.datetime64 | None = None,
    context: int = 0,
) -> list[ProfileObservations]:
    selected: list[ProfileObservations] = []
    for item in observations:
        mask = np.ones(len(item), dtype=bool)
        if signal_at_or_after is not None:
            mask &= item.signal_times >= signal_at_or_after
        if signal_before is not None:
            mask &= item.signal_times < signal_before
        if target_before is not None:
            mask &= item.target_times < target_before
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        if context:
            start = max(0, int(indices[0]) - context)
            indices = np.arange(start, int(indices[-1]) + 1)
        selected.append(item.take(indices))
    return selected


def _test_context(
    observations: Sequence[ProfileObservations], fold: WalkForwardFold, context: int
) -> list[ProfileObservations]:
    result: list[ProfileObservations] = []
    for item in observations:
        indices = np.flatnonzero(
            (item.signal_times >= fold.test_start) & (item.signal_times < fold.test_end)
        )
        if not len(indices):
            continue
        start = max(0, int(indices[0]) - context)
        result.append(item.take(np.arange(start, int(indices[-1]) + 1)))
    return result
