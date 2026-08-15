"""Leakage-safe nested ablation for the sector-pooled BNHMM."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from .config import REGIME_COUNT, SequenceConfig
from .data.dataset import ProfileObservations, ProfileSequence, SequenceWindowBuilder
from .evaluation import PredictionRecord, infer_sequences, records_as_json
from .training import BNHMMTrainer, ModelConfig, TrainerConfig, save_checkpoint
from .walk_forward import (
    WalkForwardConfig,
    _select,
    _test_context,
    _windows,
    build_walk_forward_folds,
)


ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class BNHMMAblationConfig:
    """Selection tolerances and independent seeds used by the experiment."""

    screening_seed: int = 7
    confirmation_seeds: tuple[int, ...] = (7, 17, 29)
    log_loss_tolerance: float = 1e-3
    brier_tolerance: float = 5e-4
    minimum_seed_pass_ratio: float = 2.0 / 3.0

    def __post_init__(self) -> None:
        if not self.confirmation_seeds:
            raise ValueError("at least one confirmation seed is required")
        if self.screening_seed not in self.confirmation_seeds:
            raise ValueError("screening seed must be one of the confirmation seeds")
        if len(set(self.confirmation_seeds)) != len(self.confirmation_seeds):
            raise ValueError("confirmation seeds must be unique")
        if self.log_loss_tolerance < 0 or self.brier_tolerance < 0:
            raise ValueError("ablation tolerances cannot be negative")
        if not 0.0 < self.minimum_seed_pass_ratio <= 1.0:
            raise ValueError("minimum seed pass ratio must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class BNHMMCandidate:
    name: str
    stage: str
    model: ModelConfig
    trainer: TrainerConfig
    active_objectives: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BNHMMAblationResult:
    selected_candidate: BNHMMCandidate
    selected_predictions: tuple[PredictionRecord, ...]
    screening: tuple[dict[str, object], ...]
    confirmation: tuple[dict[str, object], ...]
    outer_metrics: dict[str, object]
    fold: dict[str, object]


def run_bnhmm_ablation(
    observations: Sequence[ProfileObservations],
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
    sequence_config: SequenceConfig,
    walk_forward_config: WalkForwardConfig,
    *,
    device: str | torch.device,
    output_dir: str | Path,
    dataset_id: str,
    mc_samples: int = 8,
    ablation_config: BNHMMAblationConfig | None = None,
    progress: ProgressCallback | None = None,
) -> BNHMMAblationResult:
    """Select on the inner tuning period and evaluate one final outer model."""
    if not observations:
        raise ValueError("observations cannot be empty")
    config = ablation_config or BNHMMAblationConfig()
    folds = build_walk_forward_folds(observations, walk_forward_config)
    if len(folds) != 1:
        raise ValueError("nested BNHMM ablation requires exactly one outer fold")
    fold = folds[0]
    builder = SequenceWindowBuilder(sequence_config)
    fit_sequences = _windows(
        _select(observations, target_before=fold.tuning_start), builder
    )
    tune_sequences = _windows(
        _select(
            observations,
            signal_at_or_after=fold.tuning_start,
            signal_before=fold.test_start,
            target_before=fold.test_start,
        ),
        builder,
    )
    pretest_sequences = _windows(
        _select(observations, target_before=fold.test_start), builder
    )
    test_sequences = _windows(
        _test_context(observations, fold, sequence_config.length), builder
    )
    if not all((fit_sequences, tune_sequences, pretest_sequences, test_sequences)):
        raise ValueError("ablation fold contains an empty sequence partition")

    baseline = _candidate("full-current", "baseline", model_config, trainer_config)
    screening: list[dict[str, object]] = []
    selection_seeds = config.confirmation_seeds

    input_results: list[tuple[BNHMMCandidate, dict[str, object]]] = []
    for mode in ("price-only", "price-volume", "price-velocity", "gated-full"):
        candidate = replace(
            baseline,
            name=f"input-{mode}",
            stage="input",
            model=replace(model_config, feature_mode=mode),
        )
        result = _evaluate_candidate(
            candidate,
            selection_seeds,
            fit_sequences,
            tune_sequences,
            device=device,
            mc_samples=mc_samples,
            progress=progress,
        )
        screening.append(result)
        input_results.append((candidate, result))
    current = _select_smallest_near_best(input_results, config)
    current_result = next(
        result for candidate, result in input_results if candidate == current
    )
    baseline_result = next(
        result
        for candidate, result in input_results
        if candidate.model == baseline.model and candidate.trainer == baseline.trainer
    )
    _emit(progress, "stage-selected", stage="input", candidate=current.name)

    architecture_steps: tuple[tuple[str, Callable[[ModelConfig], ModelConfig]], ...] = (
        ("one-transformer-layer", lambda value: replace(value, layers=min(value.layers, 1))),
        ("zero-transformer-layers", lambda value: replace(value, layers=0)),
        ("thirteen-latent-states", lambda value: replace(value, latent_states=13)),
        ("static-transition", lambda value: replace(value, conditional_transition=False)),
        ("no-direct-autoregression", lambda value: replace(value, direct_autoregression=False)),
        ("no-state-metadata", lambda value: replace(value, include_metadata=False)),
        (
            "independent-latent-states",
            lambda value: replace(
                value, use_latent_filter=False, conditional_transition=False
            ),
        ),
    )
    for name, transform in architecture_steps:
        reduced_model = transform(current.model)
        if reduced_model == current.model:
            continue
        reduced = replace(current, name=name, stage="architecture", model=reduced_model)
        reduced_result = _evaluate_candidate(
            reduced,
            selection_seeds,
            fit_sequences,
            tune_sequences,
            device=device,
            mc_samples=mc_samples,
            progress=progress,
        )
        accepted = _noninferior(reduced_result, current_result, config)
        reduced_result["accepted"] = accepted
        reduced_result["compared_with"] = current.name
        screening.append(reduced_result)
        if accepted:
            current, current_result = reduced, reduced_result
        _emit(
            progress,
            "ablation-decision",
            stage="architecture",
            candidate=name,
            accepted=accepted,
        )

    loss_steps: tuple[tuple[str, Callable[[TrainerConfig], TrainerConfig]], ...] = (
        ("no-pretraining", lambda value: replace(value, pretrain_epochs=0)),
        ("no-z2-symmetry", lambda value: replace(value, symmetry_weight=0.0)),
        (
            "no-latent-information",
            lambda value: replace(value, latent_information_weight=0.0),
        ),
        ("no-head-consistency", lambda value: replace(value, consistency_weight=0.0)),
        ("no-ordinal-loss", lambda value: replace(value, ordinal_weight=0.0)),
        ("no-brier-loss", lambda value: replace(value, brier_weight=0.0)),
        (
            "deterministic-training",
            lambda value: replace(value, kl_weight=0.0, sample_during_training=False),
        ),
    )
    for name, transform in loss_steps:
        reduced_trainer = transform(current.trainer)
        if reduced_trainer == current.trainer:
            continue
        reduced = _candidate(name, "loss", current.model, reduced_trainer)
        reduced_result = _evaluate_candidate(
            reduced,
            selection_seeds,
            fit_sequences,
            tune_sequences,
            device=device,
            mc_samples=mc_samples,
            progress=progress,
        )
        accepted = _noninferior(reduced_result, current_result, config)
        reduced_result["accepted"] = accepted
        reduced_result["compared_with"] = current.name
        screening.append(reduced_result)
        if accepted:
            current, current_result = reduced, reduced_result
        _emit(
            progress,
            "ablation-decision",
            stage="loss",
            candidate=name,
            accepted=accepted,
        )

    confirmation = [baseline_result, current_result]
    selected = current
    selected_confirmation = current_result
    _emit(
        progress,
        "confirmation-complete",
        candidate=selected.name,
        reduced_model_confirmed=True,
        rule="every accepted deletion passed all configured selection seeds",
    )

    selected_epoch = _seed_epoch(selected_confirmation, config.screening_seed)
    predictions, outer_metrics = _evaluate_outer(
        selected,
        selected_epoch,
        config.screening_seed,
        pretest_sequences,
        test_sequences,
        fold.test_start,
        fold.test_end,
        device=device,
        mc_samples=mc_samples,
        output_dir=output_dir,
        dataset_id=dataset_id,
    )
    fold_payload: dict[str, object] = {
        "dataset_id": dataset_id,
        "tuning_start": str(fold.tuning_start),
        "test_start": str(fold.test_start),
        "test_end": str(fold.test_end),
        "fit_sequences": len(fit_sequences),
        "tuning_sequences": len(tune_sequences),
        "pretest_sequences": len(pretest_sequences),
        "outer_sequences": len(test_sequences),
        "selection_used_outer_test": False,
    }
    result = BNHMMAblationResult(
        selected_candidate=selected,
        selected_predictions=tuple(predictions),
        screening=tuple(screening),
        confirmation=tuple(confirmation),
        outer_metrics=outer_metrics,
        fold=fold_payload,
    )
    _save_result(result, config, output_dir)
    _emit(progress, "ablation-complete", candidate=selected.name, **outer_metrics)
    return result


def _candidate(
    name: str, stage: str, model: ModelConfig, trainer: TrainerConfig
) -> BNHMMCandidate:
    objectives = ["return_nll", "classification"]
    weighted = (
        ("brier", trainer.brier_weight),
        ("ordinal", trainer.ordinal_weight),
        ("consistency", trainer.consistency_weight),
        ("latent_information", trainer.latent_information_weight),
        ("z2_symmetry", trainer.symmetry_weight),
        ("bayesian_kl", trainer.kl_weight),
    )
    objectives.extend(name for name, weight in weighted if weight > 0)
    if trainer.pretrain_epochs:
        objectives.append("self_supervised_pretraining")
    return BNHMMCandidate(name, stage, model, trainer, tuple(objectives))


def _evaluate_candidate(
    candidate: BNHMMCandidate,
    seeds: Sequence[int],
    fit_sequences: Sequence[ProfileSequence],
    tune_sequences: Sequence[ProfileSequence],
    *,
    device: str | torch.device,
    mc_samples: int,
    progress: ProgressCallback | None,
) -> dict[str, object]:
    seed_results: list[dict[str, object]] = []
    parameter_count = 0
    for seed in seeds:
        _emit(
            progress,
            "candidate-start",
            stage=candidate.stage,
            candidate=candidate.name,
            seed=seed,
        )
        started = time.perf_counter()
        seeded_trainer = replace(candidate.trainer, seed=seed)
        model = candidate.model.build(seed=seed)
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        trainer = BNHMMTrainer(model, seeded_trainer, device=device)
        training = trainer.fit(fit_sequences, tune_sequences)
        torch.manual_seed(seed + 100_000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 100_000)
        predictions = infer_sequences(
            model,
            tune_sequences,
            device=device,
            mc_samples=mc_samples,
            batch_size=max(1, min(seeded_trainer.batch_size, 64)),
        )
        metrics = _metrics(predictions)
        row: dict[str, object] = {
            "seed": seed,
            "best_epoch": training.best_epoch,
            "duration_seconds": time.perf_counter() - started,
            **metrics,
        }
        seed_results.append(row)
        _emit(
            progress,
            "candidate-complete",
            stage=candidate.stage,
            candidate=candidate.name,
            **row,
        )
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "name": candidate.name,
        "stage": candidate.stage,
        "model_config": asdict(candidate.model),
        "trainer_config": asdict(candidate.trainer),
        "active_objectives": list(candidate.active_objectives),
        "active_parameter_count": parameter_count,
        "seed_results": seed_results,
        **_mean_metrics(seed_results),
    }


def _evaluate_outer(
    candidate: BNHMMCandidate,
    epochs: int,
    seed: int,
    pretest_sequences: Sequence[ProfileSequence],
    test_sequences: Sequence[ProfileSequence],
    test_start: np.datetime64,
    test_end: np.datetime64,
    *,
    device: str | torch.device,
    mc_samples: int,
    output_dir: str | Path,
    dataset_id: str,
) -> tuple[list[PredictionRecord], dict[str, object]]:
    fixed_trainer = replace(candidate.trainer, seed=seed, epochs=max(epochs, 1))
    model = candidate.model.build(seed=seed + 1)
    trainer = BNHMMTrainer(model, fixed_trainer, device=device)
    training = trainer.fit_fixed(pretest_sequences)
    torch.manual_seed(seed + 200_000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 200_000)
    predictions = infer_sequences(
        model,
        test_sequences,
        device=device,
        mc_samples=mc_samples,
        batch_size=max(1, min(fixed_trainer.batch_size, 64)),
    )
    predictions = [
        item
        for item in predictions
        if item.signal_time >= test_start and item.signal_time < test_end
    ]
    if not predictions:
        raise ValueError("selected BNHMM produced no outer-test predictions")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        destination / "selected-model.pt",
        model,
        candidate.model,
        fixed_trainer,
        training,
        {
            "dataset_id": dataset_id,
            "experiment": "bnhmm-ablation",
            "candidate": candidate.name,
            "training_cutoff": str(test_start),
        },
    )
    return predictions, _metrics(predictions)


def _metrics(predictions: Sequence[PredictionRecord]) -> dict[str, object]:
    if not predictions:
        raise ValueError("candidate produced no predictions")
    probability = np.asarray(
        [item.probability_mean for item in predictions], dtype=np.float64
    )
    targets = np.asarray([item.target for item in predictions], dtype=np.int64)
    predicted = probability.argmax(axis=1)
    support = np.bincount(targets, minlength=REGIME_COUNT)
    recall = np.asarray(
        [
            float((predicted[targets == state] == state).mean())
            if support[state]
            else 0.0
            for state in range(REGIME_COUNT)
        ]
    )
    one_hot = np.eye(REGIME_COUNT)[targets]
    prior = np.asarray(
        [item.prior_probability for item in predictions], dtype=np.float64
    )
    return {
        "observations": len(predictions),
        "accuracy": float((predicted == targets).mean()),
        "balanced_accuracy": float(recall.mean()),
        "log_loss": float(
            -np.log(
                np.clip(probability[np.arange(len(targets)), targets], 1e-12, 1.0)
            ).mean()
        ),
        "brier_score": float(np.square(one_hot - probability).sum(axis=1).mean()),
        "prior_log_loss": float(
            -np.log(np.clip(prior[np.arange(len(targets)), targets], 1e-12, 1.0)).mean()
        ),
        "prior_brier_score": float(np.square(one_hot - prior).sum(axis=1).mean()),
        "class_support": support.tolist(),
        "predicted_class_counts": np.bincount(
            predicted, minlength=REGIME_COUNT
        ).tolist(),
    }


def _mean_metrics(seed_results: Sequence[dict[str, object]]) -> dict[str, float]:
    names = ("accuracy", "balanced_accuracy", "log_loss", "brier_score")
    return {
        f"mean_{name}": float(np.mean([float(row[name]) for row in seed_results]))
        for name in names
    }


def _select_smallest_near_best(
    results: Sequence[tuple[BNHMMCandidate, dict[str, object]]],
    config: BNHMMAblationConfig,
) -> BNHMMCandidate:
    best_log = min(float(result["mean_log_loss"]) for _, result in results)
    best_brier = min(float(result["mean_brier_score"]) for _, result in results)
    eligible = [
        (candidate, result)
        for candidate, result in results
        if float(result["mean_log_loss"]) <= best_log + config.log_loss_tolerance
        and float(result["mean_brier_score"]) <= best_brier + config.brier_tolerance
    ]
    if not eligible:
        eligible = sorted(
            results,
            key=lambda item: (
                float(item[1]["mean_log_loss"]),
                float(item[1]["mean_brier_score"]),
            ),
        )[:1]
    return min(
        eligible,
        key=lambda item: (
            int(item[1]["active_parameter_count"]),
            float(item[1]["mean_log_loss"]),
        ),
    )[0]


def _noninferior(
    candidate: dict[str, object],
    reference: dict[str, object],
    config: BNHMMAblationConfig,
) -> bool:
    mean_pass = (
        float(candidate["mean_log_loss"])
        <= float(reference["mean_log_loss"]) + config.log_loss_tolerance
        and float(candidate["mean_brier_score"])
        <= float(reference["mean_brier_score"]) + config.brier_tolerance
    )
    reference_by_seed = {
        int(row["seed"]): row for row in reference["seed_results"]  # type: ignore[index]
    }
    compared = 0
    passed = 0
    for row in candidate["seed_results"]:  # type: ignore[index]
        baseline = reference_by_seed.get(int(row["seed"]))
        if baseline is None:
            continue
        compared += 1
        passed += int(
            float(row["log_loss"])
            <= float(baseline["log_loss"]) + config.log_loss_tolerance
            and float(row["brier_score"])
            <= float(baseline["brier_score"]) + config.brier_tolerance
        )
    seed_pass = compared > 0 and passed / compared >= config.minimum_seed_pass_ratio
    return mean_pass and seed_pass


def _seed_epoch(result: dict[str, object], seed: int) -> int:
    for row in result["seed_results"]:  # type: ignore[index]
        if int(row["seed"]) == seed:
            return max(int(row["best_epoch"]), 1)
    raise ValueError(f"confirmation result does not contain seed {seed}")


def _save_result(
    result: BNHMMAblationResult,
    config: BNHMMAblationConfig,
    output_dir: str | Path,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_candidate": {
            "name": result.selected_candidate.name,
            "stage": result.selected_candidate.stage,
            "model_config": asdict(result.selected_candidate.model),
            "trainer_config": asdict(result.selected_candidate.trainer),
            "active_objectives": list(result.selected_candidate.active_objectives),
        },
        "selection_config": asdict(config),
        "screening": list(result.screening),
        "confirmation": list(result.confirmation),
        "outer_metrics": result.outer_metrics,
        "fold": result.fold,
    }
    (destination / "ablation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "predictions.json").write_text(
        json.dumps(records_as_json(result.selected_predictions), ensure_ascii=False),
        encoding="utf-8",
    )
    (destination / "report.md").write_text(
        _markdown_report(payload), encoding="utf-8"
    )


def _markdown_report(payload: dict[str, object]) -> str:
    selected = payload["selected_candidate"]
    outer = payload["outer_metrics"]
    fold = payload["fold"]
    assert isinstance(selected, dict) and isinstance(outer, dict) and isinstance(fold, dict)
    lines = [
        "# BNHMM Nested Ablation Report",
        "",
        "## Experiment discipline",
        "",
        "- Candidate selection used only the inner tuning period.",
        "- The outer period was evaluated once with the confirmed final configuration.",
        "- Every input and deletion candidate was evaluated on all configured seeds.",
        "",
        "## Selected configuration",
        "",
        f"- Candidate: `{selected['name']}`",
        f"- Active objectives: {', '.join(selected['active_objectives'])}",
        f"- Tuning start: {fold['tuning_start']}",
        f"- Outer test: {fold['test_start']} to {fold['test_end']}",
        "",
        "## Untouched outer test",
        "",
        f"- Observations: {outer['observations']}",
        f"- Accuracy: {float(outer['accuracy']):.4f}",
        f"- Balanced accuracy: {float(outer['balanced_accuracy']):.4f}",
        f"- Log loss: {float(outer['log_loss']):.6f}",
        f"- Training-prior log loss: {float(outer['prior_log_loss']):.6f}",
        f"- Brier score: {float(outer['brier_score']):.6f}",
        f"- Training-prior Brier: {float(outer['prior_brier_score']):.6f}",
        "",
        "Full candidate and per-seed results are stored in `ablation.json`.",
    ]
    return "\n".join(lines) + "\n"


def _emit(progress: ProgressCallback | None, event: str, **values: object) -> None:
    if progress is not None:
        progress({"event": event, **values})
