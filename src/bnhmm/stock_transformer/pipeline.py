"""Walk-forward orchestration across isolated conservation, token, and model layers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..data.dataset import ProfileObservations
from ..evaluation import (
    PredictionRecord,
    estimate_regime_return_moments,
    records_as_json,
)
from ..walk_forward import WalkForwardConfig, WalkForwardFold, build_walk_forward_folds
from .conservation import (
    ConservationConfig,
    ConservationLaw,
    ConservationLawDiscoverer,
    DensityFrame,
    DensityProjector,
)
from .model import StockTransformer, StockTransformerConfig
from .tokenization import ConservationTokenizer, TokenizationConfig, TokenizedState
from .training import (
    StockTrainerConfig,
    StockTransformerTrainer,
    StockWindow,
    collate_stock_windows,
)


@dataclass(frozen=True, slots=True)
class StockPipelineConfig:
    sequence_length: int = 5
    mc_samples: int = 16

    def __post_init__(self) -> None:
        if not 2 <= self.sequence_length <= 20:
            raise ValueError("StockTransformer context must contain 2-20 days")
        if self.mc_samples < 2:
            raise ValueError("posterior sampling requires at least two passes")


@dataclass(frozen=True, slots=True)
class StockWalkForwardResult:
    folds: tuple[dict[str, object], ...]
    predictions: tuple[PredictionRecord, ...]
    regime_return_means: dict[str, tuple[float, ...]]
    regime_return_variances: dict[str, tuple[float, ...]]


def run_stock_transformer_walk_forward(
    observations: Sequence[ProfileObservations],
    conservation_config: ConservationConfig,
    tokenization_config: TokenizationConfig,
    model_config: StockTransformerConfig,
    trainer_config: StockTrainerConfig,
    pipeline_config: StockPipelineConfig,
    walk_forward_config: WalkForwardConfig,
    *,
    device: str | torch.device,
    output_dir: str | Path,
    dataset_id: str | None = None,
) -> StockWalkForwardResult:
    if not observations:
        raise ValueError("StockTransformer requires profile observations")
    folds = build_walk_forward_folds(observations, walk_forward_config)
    if not folds:
        raise ValueError("dataset does not contain enough common sessions")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    all_predictions: dict[tuple[str, int], PredictionRecord] = {}
    fold_summaries: list[dict[str, object]] = []
    return_means: dict[str, tuple[float, ...]] = {}
    return_variances: dict[str, tuple[float, ...]] = {}

    for fold in folds:
        fit_observations = _select(observations, target_before=fold.tuning_start)
        tune_observations = _select(
            observations,
            signal_at_or_after=fold.tuning_start,
            signal_before=fold.test_start,
            target_before=fold.test_start,
            context=pipeline_config.sequence_length + 1,
        )
        pretest_observations = _select(observations, target_before=fold.test_start)
        test_observations = _test_context(
            observations, fold, pipeline_config.sequence_length + 1
        )
        tuning_law, fit_windows = _fit_law_and_windows(
            fit_observations,
            conservation_config,
            tokenization_config,
            pipeline_config.sequence_length,
        )
        tune_windows = _windows_with_law(
            tune_observations,
            tuning_law,
            conservation_config,
            tokenization_config,
            pipeline_config.sequence_length,
        )
        if not fit_windows or not tune_windows:
            raise ValueError(f"StockTransformer fold {fold.index} has empty tuning data")
        tuning_model = model_config.build(seed=trainer_config.seed + fold.index * 2)
        tuning_trainer = StockTransformerTrainer(
            tuning_model, trainer_config, device=device
        )
        tuning_result = tuning_trainer.fit(fit_windows, tune_windows)
        selected_epochs = max(tuning_result.best_epoch, 1)
        del tuning_trainer, tuning_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        final_law, pretest_windows = _fit_law_and_windows(
            pretest_observations,
            conservation_config,
            tokenization_config,
            pipeline_config.sequence_length,
        )
        test_windows = _windows_with_law(
            test_observations,
            final_law,
            conservation_config,
            tokenization_config,
            pipeline_config.sequence_length,
        )
        test_windows = [
            item
            for item in test_windows
            if fold.test_start <= item.signal_time < fold.test_end
        ]
        if not pretest_windows or not test_windows:
            raise ValueError(f"StockTransformer fold {fold.index} has empty outer data")
        final_trainer_config = replace(trainer_config, epochs=selected_epochs)
        final_model = model_config.build(seed=trainer_config.seed + fold.index * 2 + 1)
        final_trainer = StockTransformerTrainer(
            final_model, final_trainer_config, device=device
        )
        final_result = final_trainer.fit_fixed(pretest_windows)
        predictions = _infer(
            final_model,
            test_windows,
            device=device,
            mc_samples=pipeline_config.mc_samples,
            batch_size=trainer_config.batch_size,
        )
        for prediction in predictions:
            all_predictions.setdefault(
                (prediction.symbol, int(prediction.signal_time.astype(np.int64))),
                prediction,
            )
        token_diagnostics = _token_diagnostics(test_windows)
        fold_dir = destination / f"fold-{fold.index:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "strategy": "stock-transformer",
                "model_state": final_model.state_dict(),
                "model_config": asdict(model_config),
                "trainer_config": asdict(final_trainer_config),
                "pipeline_config": asdict(pipeline_config),
                "tokenization_config": asdict(tokenization_config),
                "conservation_config": asdict(conservation_config),
                "conservation_law": final_law.as_json(),
                "training_cutoff": str(fold.test_start),
            },
            fold_dir / "model.pt",
        )
        summary: dict[str, object] = {
            "strategy": "stock-transformer",
            "symbol": f"sector:{dataset_id or 'pooled'}",
            "dataset_id": dataset_id,
            "fold": fold.index,
            "tuning_start": str(fold.tuning_start),
            "test_start": str(fold.test_start),
            "test_end": str(fold.test_end),
            "training_cutoff": str(fold.test_start),
            "selected_epochs": selected_epochs,
            "pretest_sequences": len(pretest_windows),
            "outer_test_predictions": len(predictions),
            "tuning_best_validation_loss": tuning_result.best_validation_loss,
            "tuning_history": list(tuning_result.history),
            "final_training_history": list(final_result.history),
            "conservation_law": final_law.as_json(),
            "token_diagnostics": token_diagnostics,
        }
        (fold_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fold_summaries.append(summary)
        del final_trainer, final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for item in observations:
        means, variances = estimate_regime_return_moments([item])
        return_means[item.symbol] = tuple(float(value) for value in means)
        return_variances[item.symbol] = tuple(float(value) for value in variances)
    ordered = tuple(
        sorted(
            all_predictions.values(),
            key=lambda item: (item.signal_time, item.symbol),
        )
    )
    (destination / "predictions.json").write_text(
        json.dumps(records_as_json(ordered), ensure_ascii=False), encoding="utf-8"
    )
    return StockWalkForwardResult(
        folds=tuple(fold_summaries),
        predictions=ordered,
        regime_return_means=return_means,
        regime_return_variances=return_variances,
    )


def _fit_law_and_windows(
    observations: Sequence[ProfileObservations],
    conservation_config: ConservationConfig,
    tokenization_config: TokenizationConfig,
    sequence_length: int,
) -> tuple[ConservationLaw, list[StockWindow]]:
    projector = DensityProjector(conservation_config)
    series = [_frames(item, projector) for item in observations]
    law = ConservationLawDiscoverer(conservation_config).fit(series)
    return law, _build_windows(
        observations,
        series,
        law,
        conservation_config,
        tokenization_config,
        sequence_length,
    )


def _windows_with_law(
    observations: Sequence[ProfileObservations],
    law: ConservationLaw,
    conservation_config: ConservationConfig,
    tokenization_config: TokenizationConfig,
    sequence_length: int,
) -> list[StockWindow]:
    projector = DensityProjector(conservation_config)
    series = [_frames(item, projector) for item in observations]
    return _build_windows(
        observations,
        series,
        law,
        conservation_config,
        tokenization_config,
        sequence_length,
    )


def _frames(
    observations: ProfileObservations, projector: DensityProjector
) -> list[DensityFrame]:
    return [
        projector.project(
            observations.symbol,
            observations.signal_times[index],
            observations.profiles[index][:, 3],
            observations.profiles[index][:, 1],
        )
        for index in range(len(observations))
    ]


def _build_windows(
    observations: Sequence[ProfileObservations],
    series: Sequence[Sequence[DensityFrame]],
    law: ConservationLaw,
    conservation_config: ConservationConfig,
    tokenization_config: TokenizationConfig,
    sequence_length: int,
) -> list[StockWindow]:
    discoverer = ConservationLawDiscoverer(conservation_config)
    tokenizer = ConservationTokenizer(tokenization_config)
    windows: list[StockWindow] = []
    for item, frames in zip(observations, series, strict=True):
        states: list[tuple[int, TokenizedState]] = []
        for index in range(1, len(frames)):
            transition = discoverer.transform(frames[index - 1], frames[index], law)
            states.append((index, tokenizer.tokenize(transition, law)))
        for end in range(sequence_length, len(states) + 1):
            selected = states[end - sequence_length : end]
            endpoint = selected[-1][0]
            windows.append(
                StockWindow(
                    symbol=item.symbol,
                    states=tuple(state for _, state in selected),
                    signal_time=item.signal_times[endpoint],
                    execution_time=item.execution_times[endpoint],
                    target_time=item.target_times[endpoint],
                    target=int(item.targets[endpoint]),
                    forward_return=float(item.forward_returns[endpoint]),
                    standardized_return=float(item.standardized_returns[endpoint]),
                    return_location=float(item.return_locations[endpoint]),
                    return_scale=float(item.return_scales[endpoint]),
                )
            )
    return windows


@torch.no_grad()
def _infer(
    model: StockTransformer,
    windows: Sequence[StockWindow],
    *,
    device: str | torch.device,
    mc_samples: int,
    batch_size: int,
) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    model.eval()
    prior = tuple(float(value) for value in model.class_prior.detach().cpu())
    for offset in range(0, len(windows), batch_size):
        items = windows[offset : offset + batch_size]
        batch = collate_stock_windows(items)
        posterior = model.predict_posterior(
            batch["token_features"].to(device),
            batch["token_mask"].to(device),
            mc_samples=mc_samples,
        )
        for index, item in enumerate(items):
            scale = item.return_scale
            records.append(
                PredictionRecord(
                    symbol=item.symbol,
                    timeframe_id=1,
                    signal_time=item.signal_time,
                    execution_time=item.execution_time,
                    target_time=item.target_time,
                    target=item.target,
                    forward_return=item.forward_return,
                    standardized_return=item.standardized_return,
                    return_location=item.return_location,
                    return_scale=scale,
                    prior_probability=prior,
                    probability_mean=tuple(
                        float(value) for value in posterior.probability_mean[index].cpu()
                    ),
                    aleatoric_variance=tuple(
                        float(value) for value in posterior.aleatoric_variance[index].cpu()
                    ),
                    epistemic_variance=tuple(
                        float(value) for value in posterior.epistemic_variance[index].cpu()
                    ),
                    expected_score=float(posterior.expected_score[index].cpu()),
                    expected_return=float(posterior.expected_return[index].cpu()) * scale,
                    expected_standardized_return=float(
                        posterior.expected_return[index].cpu()
                    ),
                    return_variance=float(posterior.return_variance[index].cpu())
                    * scale**2,
                    aleatoric_return_variance=float(
                        posterior.aleatoric_return_variance[index].cpu()
                    )
                    * scale**2,
                    epistemic_return_variance=float(
                        posterior.epistemic_return_variance[index].cpu()
                    )
                    * scale**2,
                    context_length=len(item.states),
                )
            )
    return records


def _token_diagnostics(windows: Sequence[StockWindow]) -> dict[str, float | int]:
    states = [state for window in windows for state in window.states]
    token_counts = np.asarray([len(state.tokens) for state in states], dtype=np.float64)
    sentence_counts = np.asarray(
        [max(token.sentence_id for token in state.tokens) + 1 for state in states],
        dtype=np.float64,
    )
    return {
        "state_count": len(states),
        "mean_tokens_per_day": float(token_counts.mean()),
        "p90_tokens_per_day": float(np.quantile(token_counts, 0.9)),
        "mean_sentences_per_day": float(sentence_counts.mean()),
    }


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
    observations: Sequence[ProfileObservations],
    fold: WalkForwardFold,
    context: int,
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
