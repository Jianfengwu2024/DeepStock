"""Causal probability and trading evaluation for BoundaryFlowFK."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, pi, sqrt

import numpy as np
from scipy.special import ndtr

from .feynman_kac import BoundaryFeynmanKacForecast
from .policy import BoundaryPolicyDecision


@dataclass(frozen=True, slots=True)
class BoundaryEvaluationConfig:
    transaction_cost_bps: float = 10.0
    calibration_bins: int = 10
    coverage_probability: float = 0.90

    def __post_init__(self) -> None:
        if not np.isfinite(self.transaction_cost_bps) or self.transaction_cost_bps < 0.0:
            raise ValueError("transaction cost must be finite and non-negative")
        if self.calibration_bins < 2:
            raise ValueError("calibration bins must be at least two")
        if not 0.5 < self.coverage_probability < 1.0:
            raise ValueError("coverage probability must lie in (0.5, 1)")


@dataclass(frozen=True, slots=True)
class BoundaryForecastObservation:
    signal_time: np.datetime64
    available_at: np.datetime64
    target_time: np.datetime64
    training_cutoff: np.datetime64
    model_version: str
    forecast: BoundaryFeynmanKacForecast
    actual_return: float
    neutral_threshold: float
    decision: BoundaryPolicyDecision
    gross_strategy_return: float


@dataclass(frozen=True, slots=True)
class BoundaryEvaluationMetrics:
    observations: int
    log_loss: float
    brier_score: float
    ece: float
    crps: float
    interval_coverage: float
    balanced_accuracy: float
    macro_f1: float
    probability_residual_autocorrelation: float
    gross_total_return: float
    net_total_return: float
    maximum_drawdown: float
    average_turnover: float
    open_rate: float
    profit_concentration: float


def evaluate_boundary_forecasts(
    observations: tuple[BoundaryForecastObservation, ...],
    config: BoundaryEvaluationConfig | None = None,
) -> BoundaryEvaluationMetrics:
    settings = config or BoundaryEvaluationConfig()
    if not observations:
        raise ValueError("boundary evaluation requires observations")
    for item in observations:
        _validate_observation(item)

    probabilities = np.asarray(
        [item.forecast.raw_state_probabilities for item in observations],
        dtype=np.float64,
    )
    actual_returns = np.asarray([item.actual_return for item in observations])
    actual_classes = np.asarray(
        [_return_class(item.actual_return, item.neutral_threshold) for item in observations]
    )
    predicted_classes = np.argmax(probabilities, axis=1)
    one_hot = np.eye(3, dtype=np.float64)[actual_classes]
    selected_probability = probabilities[np.arange(len(observations)), actual_classes]
    log_loss = -float(np.mean(np.log(np.clip(selected_probability, 1e-12, 1.0))))
    brier = float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1)))

    confidence = probabilities.max(axis=1)
    correct = predicted_classes == actual_classes
    ece = _expected_calibration_error(confidence, correct, settings.calibration_bins)
    crps = float(
        np.mean(
            [
                _normal_crps(
                    item.forecast.expected_return,
                    sqrt(max(item.forecast.total_return_variance, 1e-16)),
                    item.actual_return,
                )
                for item in observations
            ]
        )
    )
    interval_z = _normal_quantile((1.0 + settings.coverage_probability) * 0.5)
    covered = [
        abs(item.actual_return - item.forecast.expected_return)
        <= interval_z * sqrt(max(item.forecast.total_return_variance, 0.0))
        for item in observations
    ]

    opened = np.asarray([item.decision.open_position for item in observations])
    gross_returns = np.asarray(
        [item.gross_strategy_return if is_open else 0.0 for item, is_open in zip(observations, opened)]
    )
    turnover = opened.astype(np.float64) * 2.0
    net_returns = gross_returns - turnover * settings.transaction_cost_bps / 10_000.0
    gross_equity = np.cumprod(1.0 + gross_returns)
    net_equity = np.cumprod(1.0 + net_returns)
    probability_residual = one_hot[:, 2] - probabilities[:, 2]

    return BoundaryEvaluationMetrics(
        observations=len(observations),
        log_loss=log_loss,
        brier_score=brier,
        ece=ece,
        crps=crps,
        interval_coverage=float(np.mean(covered)),
        balanced_accuracy=_balanced_accuracy(actual_classes, predicted_classes),
        macro_f1=_macro_f1(actual_classes, predicted_classes),
        probability_residual_autocorrelation=_lag_one_autocorrelation(probability_residual),
        gross_total_return=float(gross_equity[-1] - 1.0),
        net_total_return=float(net_equity[-1] - 1.0),
        maximum_drawdown=_maximum_drawdown(net_equity),
        average_turnover=float(np.mean(turnover)),
        open_rate=float(np.mean(opened)),
        profit_concentration=_profit_concentration(net_returns[opened]),
    )


def _validate_observation(item: BoundaryForecastObservation) -> None:
    signal = np.datetime64(item.signal_time, "ns")
    available = np.datetime64(item.available_at, "ns")
    target = np.datetime64(item.target_time, "ns")
    cutoff = np.datetime64(item.training_cutoff, "ns")
    if cutoff > signal:
        raise ValueError("training_cutoff cannot exceed signal_time")
    if available < signal or available >= target:
        raise ValueError("available_at must lie between signal_time and target_time")
    if target <= signal:
        raise ValueError("target_time must follow signal_time")
    if not item.model_version:
        raise ValueError("model_version is required")
    values = (item.actual_return, item.neutral_threshold, item.gross_strategy_return)
    if any(not np.isfinite(value) for value in values) or item.neutral_threshold < 0.0:
        raise ValueError("boundary evaluation returns and threshold must be finite")
    probabilities = np.asarray(item.forecast.raw_state_probabilities)
    if np.any(probabilities < 0.0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-8):
        raise ValueError("raw state probabilities must form a distribution")


def _return_class(value: float, threshold: float) -> int:
    if value < -threshold:
        return 0
    if value > threshold:
        return 2
    return 1


def _expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, bins: int
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidence)
    result = 0.0
    for index in range(bins):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        if np.any(selected):
            result += float(np.sum(selected)) / total * abs(
                float(np.mean(correct[selected])) - float(np.mean(confidence[selected]))
            )
    return result


def _normal_crps(mean: float, std: float, observation: float) -> float:
    z = (observation - mean) / std
    density = exp(-0.5 * z * z) / sqrt(2.0 * pi)
    return std * (z * (2.0 * float(ndtr(z)) - 1.0) + 2.0 * density - 1.0 / sqrt(pi))


def _normal_quantile(probability: float) -> float:
    low, high = -8.0, 8.0
    for _ in range(80):
        middle = (low + high) * 0.5
        if float(ndtr(middle)) < probability:
            low = middle
        else:
            high = middle
    return (low + high) * 0.5


def _balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    recalls = [float(np.mean(predicted[actual == label] == label)) for label in np.unique(actual)]
    return float(np.mean(recalls))


def _macro_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    scores: list[float] = []
    for label in np.unique(actual):
        true_positive = int(np.sum((actual == label) & (predicted == label)))
        false_positive = int(np.sum((actual != label) & (predicted == label)))
        false_negative = int(np.sum((actual == label) & (predicted != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2.0 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def _lag_one_autocorrelation(values: np.ndarray) -> float:
    if len(values) < 2 or np.std(values[:-1]) <= 1e-15 or np.std(values[1:]) <= 1e-15:
        return 0.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _maximum_drawdown(equity: np.ndarray) -> float:
    with_initial = np.concatenate((np.ones(1), equity))
    peaks = np.maximum.accumulate(with_initial)
    return float(np.max(1.0 - with_initial / peaks))


def _profit_concentration(trade_returns: np.ndarray) -> float:
    positive = np.sort(trade_returns[trade_returns > 0.0])[::-1]
    if not len(positive):
        return 0.0
    top_count = max(1, ceil(len(positive) * 0.1))
    return float(np.sum(positive[:top_count]) / np.sum(positive))
