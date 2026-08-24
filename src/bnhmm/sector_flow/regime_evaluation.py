"""Leakage-safe metrics for three-state sector forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .regime_data import SectorRegimeState


MODEL_VERSION = "sector-regime-bnhmm-v1"


@dataclass(frozen=True, slots=True)
class SectorRegimePredictionRecord:
    signal_time: np.datetime64
    available_at: np.datetime64
    target_time: np.datetime64
    training_cutoff: np.datetime64
    model_version: str
    probability_mean: NDArray[np.float64]
    probability_variance: NDArray[np.float64]
    previous_state: SectorRegimeState
    realized_state: SectorRegimeState
    realized_return: float
    return_scale: float
    next_image_mean: NDArray[np.float64]
    next_image_variance: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.probability_mean.shape != (3,) or self.probability_variance.shape != (3,):
            raise ValueError("sector regime prediction must contain three probabilities")
        if not np.isclose(float(self.probability_mean.sum()), 1.0, atol=1e-5):
            raise ValueError("sector regime probabilities must sum to one")
        if np.any(self.probability_mean < 0.0) or np.any(self.probability_variance < 0.0):
            raise ValueError("sector regime posterior values must be nonnegative")
        if self.next_image_mean.shape != self.next_image_variance.shape:
            raise ValueError("sector state forecast mean and variance must align")
        if not self.training_cutoff < self.signal_time < self.available_at <= self.target_time:
            raise ValueError("sector regime prediction timing must be strictly causal")
        if not self.model_version or self.return_scale <= 0.0:
            raise ValueError("sector regime prediction metadata is invalid")

    def as_json(self) -> dict[str, object]:
        return {
            "signal_time": str(self.signal_time),
            "available_at": str(self.available_at),
            "target_time": str(self.target_time),
            "training_cutoff": str(self.training_cutoff),
            "model_version": self.model_version,
            "probability_mean": self.probability_mean.tolist(),
            "probability_variance": self.probability_variance.tolist(),
            "predicted_state": int(np.argmax(self.probability_mean)),
            "previous_state": int(self.previous_state),
            "realized_state": int(self.realized_state),
            "realized_return": self.realized_return,
            "return_scale": self.return_scale,
        }


def evaluate_sector_regimes(
    records: Sequence[SectorRegimePredictionRecord],
    *,
    training_class_prior: NDArray[np.float64],
) -> dict[str, object]:
    if not records:
        raise ValueError("cannot evaluate empty sector regime predictions")
    prior = np.asarray(training_class_prior, dtype=np.float64)
    if prior.shape != (3,) or np.any(prior < 0.0) or not np.isclose(prior.sum(), 1.0):
        raise ValueError("training class prior must contain three probabilities")
    probability = np.stack([item.probability_mean for item in records])
    realized = np.asarray([int(item.realized_state) for item in records], dtype=np.int64)
    predicted = probability.argmax(axis=1)
    previous = np.asarray([int(item.previous_state) for item in records], dtype=np.int64)
    confusion = np.zeros((3, 3), dtype=np.int64)
    for target, estimate in zip(realized, predicted, strict=True):
        confusion[target, estimate] += 1
    support = confusion.sum(axis=1)
    recall = np.divide(
        np.diag(confusion), support, out=np.zeros(3, dtype=np.float64), where=support > 0
    )
    predicted_count = confusion.sum(axis=0)
    precision = np.divide(
        np.diag(confusion),
        predicted_count,
        out=np.zeros(3, dtype=np.float64),
        where=predicted_count > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(3, dtype=np.float64),
        where=(precision + recall) > 0.0,
    )
    present = support > 0
    one_hot = np.eye(3, dtype=np.float64)[realized]
    clipped = np.clip(probability, 1e-9, 1.0)
    majority = int(np.argmax(prior))
    return {
        "prediction_count": len(records),
        "accuracy": float(np.mean(predicted == realized)),
        "balanced_accuracy": float(np.mean(recall[present])),
        "macro_f1": float(np.mean(f1[present])),
        "brier_score": float(np.mean(np.sum(np.square(probability - one_hot), axis=1))),
        "log_loss": float(-np.mean(np.log(clipped[np.arange(len(realized)), realized]))),
        "confusion_matrix": confusion.tolist(),
        "support": support.tolist(),
        "majority_state": majority,
        "majority_baseline_accuracy": float(np.mean(realized == majority)),
        "persistence_baseline_accuracy": float(np.mean(realized == previous)),
    }

