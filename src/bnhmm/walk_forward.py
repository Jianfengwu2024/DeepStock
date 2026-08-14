"""Expanding-window training with inner tuning and untouched outer tests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .config import SequenceConfig
from .data.dataset import ProfileObservations, ProfileSequence, SequenceWindowBuilder
from .evaluation import (
    PredictionRecord,
    estimate_regime_return_moments,
    infer_sequences,
    records_as_json,
)
from .model import BNHMMNetwork
from .training import BNHMMTrainer, ModelConfig, TrainerConfig, save_checkpoint


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    minimum_train_sessions: int = 252
    tuning_sessions: int = 60
    test_sessions: int = 60
    step_sessions: int = 60
    max_folds: int | None = None

    def __post_init__(self) -> None:
        if self.minimum_train_sessions <= self.tuning_sessions:
            raise ValueError("minimum_train_sessions must exceed tuning_sessions")
        if min(self.tuning_sessions, self.test_sessions, self.step_sessions) < 1:
            raise ValueError("walk-forward session counts must be positive")
        if self.max_folds is not None and self.max_folds < 1:
            raise ValueError("max_folds must be positive")


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    index: int
    tuning_start: np.datetime64
    test_start: np.datetime64
    test_end: np.datetime64


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    folds: tuple[dict[str, object], ...]
    predictions: tuple[PredictionRecord, ...]
    regime_return_means: dict[str, tuple[float, ...]]
    regime_return_variances: dict[str, tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class _SymbolWalkForwardResult:
    folds: tuple[dict[str, object], ...]
    predictions: tuple[PredictionRecord, ...]
    regime_return_means: tuple[float, ...]
    regime_return_variances: tuple[float, ...]


def build_walk_forward_folds(
    observations: Sequence[ProfileObservations], config: WalkForwardConfig
) -> list[WalkForwardFold]:
    sessions = np.unique(
        np.concatenate(
            [item.signal_times.astype("datetime64[D]") for item in observations]
        )
    )
    folds: list[WalkForwardFold] = []
    test_index = config.minimum_train_sessions
    while test_index < len(sessions):
        end_index = min(test_index + config.test_sessions, len(sessions))
        test_end = (
            sessions[end_index]
            if end_index < len(sessions)
            else sessions[-1] + np.timedelta64(1, "D")
        )
        folds.append(
            WalkForwardFold(
                index=len(folds),
                tuning_start=sessions[test_index - config.tuning_sessions],
                test_start=sessions[test_index],
                test_end=test_end,
            )
        )
        if end_index == len(sessions):
            break
        test_index += config.step_sessions
    if config.max_folds is not None:
        folds = folds[: config.max_folds]
    return folds


def run_walk_forward(
    observations: Sequence[ProfileObservations],
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
    sequence_config: SequenceConfig,
    walk_forward_config: WalkForwardConfig,
    *,
    device: str | torch.device,
    output_dir: str | Path,
    mc_samples: int = 16,
) -> WalkForwardResult:
    if not observations:
        raise ValueError("observations cannot be empty")
    symbols = [item.symbol for item in observations]
    if len(symbols) != len(set(symbols)):
        raise ValueError("each symbol must have exactly one observation set")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    all_predictions: list[PredictionRecord] = []
    fold_summaries: list[dict[str, object]] = []
    return_means: dict[str, tuple[float, ...]] = {}
    return_variances: dict[str, tuple[float, ...]] = {}

    for observation in observations:
        symbol_destination = destination / "symbols" / observation.symbol
        result = _run_symbol_walk_forward(
            observation,
            model_config,
            trainer_config,
            sequence_config,
            walk_forward_config,
            device=device,
            output_dir=symbol_destination,
            mc_samples=mc_samples,
        )
        all_predictions.extend(result.predictions)
        fold_summaries.extend(result.folds)
        return_means[observation.symbol] = result.regime_return_means
        return_variances[observation.symbol] = result.regime_return_variances

    ordered_predictions = tuple(
        sorted(
            all_predictions,
            key=lambda item: (item.signal_time, item.symbol, item.timeframe_id),
        )
    )
    (destination / "predictions.json").write_text(
        json.dumps(records_as_json(ordered_predictions), ensure_ascii=False),
        encoding="utf-8",
    )
    return WalkForwardResult(
        folds=tuple(fold_summaries),
        predictions=ordered_predictions,
        regime_return_means=return_means,
        regime_return_variances=return_variances,
    )


def _run_symbol_walk_forward(
    observation: ProfileObservations,
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
    sequence_config: SequenceConfig,
    walk_forward_config: WalkForwardConfig,
    *,
    device: str | torch.device,
    output_dir: str | Path,
    mc_samples: int,
) -> _SymbolWalkForwardResult:
    observations = [observation]
    folds = build_walk_forward_folds(observations, walk_forward_config)
    if not folds:
        raise ValueError(f"{observation.symbol} does not have enough sessions")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    window_builder = SequenceWindowBuilder(sequence_config)
    all_predictions: dict[tuple[str, int, int], PredictionRecord] = {}
    fold_summaries: list[dict[str, object]] = []
    fixed_moments: tuple[np.ndarray, np.ndarray] | None = None

    for fold in folds:
        fit_observations = _select(
            observations, target_before=fold.tuning_start
        )
        tune_observations = _select(
            observations,
            signal_at_or_after=fold.tuning_start,
            signal_before=fold.test_start,
            target_before=fold.test_start,
        )
        pretest_observations = _select(
            observations, target_before=fold.test_start
        )
        test_context = _test_context(
            observations, fold, sequence_config.length
        )
        fit_sequences = _windows(fit_observations, window_builder)
        tune_sequences = _windows(tune_observations, window_builder)
        pretest_sequences = _windows(pretest_observations, window_builder)
        test_sequences = _windows(test_context, window_builder)
        if not all((fit_sequences, tune_sequences, pretest_sequences, test_sequences)):
            raise ValueError(
                f"{observation.symbol} fold {fold.index} contains an empty sequence partition"
            )

        tuning_model = model_config.build(seed=trainer_config.seed + fold.index * 2)
        tuning_trainer = BNHMMTrainer(
            tuning_model, trainer_config, device=device
        )
        tuning_result = tuning_trainer.fit(fit_sequences, tune_sequences)
        selected_epochs = max(tuning_result.best_epoch, 1)
        del tuning_trainer, tuning_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        final_config = replace(trainer_config, epochs=selected_epochs)
        final_model = model_config.build(seed=trainer_config.seed + fold.index * 2 + 1)
        final_trainer = BNHMMTrainer(final_model, final_config, device=device)
        final_result = final_trainer.fit_fixed(pretest_sequences)
        inference_seed = trainer_config.seed + 100_000 + fold.index
        torch.manual_seed(inference_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(inference_seed)
        predictions = infer_sequences(
            final_model,
            test_sequences,
            device=device,
            mc_samples=mc_samples,
        )
        predictions = [
            item
            for item in predictions
            if item.signal_time >= fold.test_start
            and item.signal_time < fold.test_end
        ]
        if not predictions:
            raise ValueError(
                f"{observation.symbol} fold {fold.index} produced no outer-test predictions"
            )
        for prediction in predictions:
            key = (
                prediction.symbol,
                int(prediction.signal_time.astype(np.int64)),
                prediction.timeframe_id,
            )
            all_predictions.setdefault(key, prediction)

        if fixed_moments is None:
            fixed_moments = estimate_regime_return_moments(pretest_observations)
        fold_dir = destination / f"fold-{fold.index:02d}"
        metadata = {
            "symbol": observation.symbol,
            "model_scope": "single-symbol",
            "fold": fold.index,
            "tuning_start": str(fold.tuning_start),
            "test_start": str(fold.test_start),
            "test_end": str(fold.test_end),
            "selected_epochs": selected_epochs,
            "pretrain_epochs": trainer_config.pretrain_epochs,
            "symbols": [observation.symbol],
            "training_cutoff": str(fold.test_start),
        }
        save_checkpoint(
            fold_dir / "model.pt",
            final_model,
            model_config,
            final_config,
            final_result,
            metadata,
        )
        summary = {
            **metadata,
            "fit_sequences": len(fit_sequences),
            "tune_sequences": len(tune_sequences),
            "pretest_sequences": len(pretest_sequences),
            "outer_test_predictions": len(predictions),
            "tuning_best_validation_loss": tuning_result.best_validation_loss,
            "training_class_prior": final_model.class_prior.detach().cpu().tolist(),
            "training_class_weights": final_model.class_weights.detach().cpu().tolist(),
            "tuning_history": list(tuning_result.history),
            "final_training_history": list(final_result.history),
            "transition_diagnostics": _transition_diagnostics(final_model),
            "auxiliary_gates": dict(
                zip(
                    ("relative_volume", "joint_price_volume_velocity"),
                    final_model.tokenizer.effective_auxiliary_gates()
                    .detach()
                    .cpu()
                    .tolist(),
                    strict=True,
                )
            ),
        }
        (fold_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fold_summaries.append(summary)
        del final_trainer, final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ordered_predictions = tuple(
        sorted(
            all_predictions.values(),
            key=lambda item: (item.signal_time, item.symbol, item.timeframe_id),
        )
    )
    (destination / "predictions.json").write_text(
        json.dumps(records_as_json(ordered_predictions), ensure_ascii=False),
        encoding="utf-8",
    )
    assert fixed_moments is not None
    return _SymbolWalkForwardResult(
        folds=tuple(fold_summaries),
        predictions=ordered_predictions,
        regime_return_means=tuple(float(value) for value in fixed_moments[0]),
        regime_return_variances=tuple(float(value) for value in fixed_moments[1]),
    )


def _transition_diagnostics(model: BNHMMNetwork) -> dict[str, float]:
    transition_mu = model.transition_mu.detach()
    probability = torch.softmax(transition_mu, dim=-1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
    initial = torch.eye(
        model.latent_states, device=transition_mu.device, dtype=transition_mu.dtype
    ) * 2.0
    return {
        "base_self_transition_probability": float(
            probability.diagonal().mean().cpu()
        ),
        "base_effective_destinations": float(entropy.exp().mean().cpu()),
        "base_parameter_delta_norm": float((transition_mu - initial).norm().cpu()),
        "latent_emission_weight_norm": float(
            model.latent_emission.weight_mu.detach().norm().cpu()
        ),
        "regime_evidence_weight_norm": float(
            model.regime_evidence.weight_mu.detach().norm().cpu()
        ),
    }


def _select(
    observations: Sequence[ProfileObservations],
    *,
    signal_at_or_after: np.datetime64 | None = None,
    signal_before: np.datetime64 | None = None,
    target_before: np.datetime64 | None = None,
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
        if len(indices):
            selected.append(item.take(indices))
    return selected


def _test_context(
    observations: Sequence[ProfileObservations],
    fold: WalkForwardFold,
    sequence_length: int,
) -> list[ProfileObservations]:
    result: list[ProfileObservations] = []
    for item in observations:
        test_indices = np.flatnonzero(
            (item.signal_times >= fold.test_start)
            & (item.signal_times < fold.test_end)
        )
        if not len(test_indices):
            continue
        start = max(0, int(test_indices[0]) - sequence_length + 1)
        end = int(test_indices[-1]) + 1
        result.append(item.take(np.arange(start, end)))
    return result


def _windows(
    observations: Sequence[ProfileObservations], builder: SequenceWindowBuilder
) -> list[ProfileSequence]:
    return [sequence for item in observations for sequence in builder.build(item)]
