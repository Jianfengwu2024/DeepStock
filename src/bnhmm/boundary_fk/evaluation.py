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
    transaction_cost_bps: float = 5.0
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
class TwoBranchForecastObservation:
    signal_time: np.datetime64
    available_at: np.datetime64
    target_time: np.datetime64
    training_cutoff: np.datetime64
    model_version: str
    forecasts: tuple[BoundaryFeynmanKacForecast, ...]
    actual_returns: tuple[float, ...]
    neutral_thresholds: tuple[float, ...]
    decision: BoundaryPolicyDecision
    gross_strategy_return: float
    turnover: float
    branch_ids: tuple[int, ...] = ()
    actual_symbols: tuple[str, ...] = ()
    actual_weights: tuple[float, ...] = ()
    failed_buy_orders: int = 0
    failed_sell_orders: int = 0
    branch_equities: tuple[float, ...] = ()
    branch_peak_equities: tuple[float, ...] = ()
    branch_trailing_drawdowns: tuple[float, ...] = ()
    trailing_stop_pending: bool = False
    trailing_stop_triggered_branches: tuple[bool, ...] = ()
    trailing_stop_forced_exit: bool = False


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
    average_exposure: float
    open_rate: float
    profit_concentration: float
    direction_counts: tuple[int, int, int]
    drift_rank_ic: float
    drift_direction_accuracy: float
    direction_signal_coverage: float = 0.0
    direction_signal_accuracy: float = 0.0
    failed_buy_orders: int = 0
    failed_sell_orders: int = 0


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

    exposure = np.asarray(
        [sum(item.decision.target_weights) for item in observations], dtype=np.float64
    )
    opened = exposure > 0.0
    gross_returns = np.asarray(
        [item.gross_strategy_return if is_open else 0.0 for item, is_open in zip(observations, opened)]
    )
    turnover = opened.astype(np.float64) * 2.0
    net_returns = gross_returns - turnover * settings.transaction_cost_bps / 10_000.0
    gross_equity = np.cumprod(1.0 + gross_returns)
    net_equity = np.cumprod(1.0 + net_returns)
    probability_residual = one_hot[:, 2] - probabilities[:, 2]
    expected_drift = np.asarray(
        [item.forecast.expected_drift for item in observations], dtype=np.float64
    )
    direction_signal = np.abs(expected_drift) > 1e-12

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
        average_exposure=float(np.mean(exposure)),
        open_rate=float(np.mean(opened)),
        profit_concentration=_profit_concentration(net_returns[opened]),
        direction_counts=tuple(
            int(np.sum(predicted_classes == label)) for label in range(3)
        ),
        drift_rank_ic=_rank_correlation(expected_drift, actual_returns),
        drift_direction_accuracy=float(
            np.mean(np.sign(expected_drift) == np.sign(actual_returns))
        ),
        direction_signal_coverage=float(np.mean(direction_signal)),
        direction_signal_accuracy=(
            float(
                np.mean(
                    np.sign(expected_drift[direction_signal])
                    == np.sign(actual_returns[direction_signal])
                )
            )
            if np.any(direction_signal)
            else 0.0
        ),
    )


def evaluate_two_branch_forecasts(
    observations: tuple[TwoBranchForecastObservation, ...],
    config: BoundaryEvaluationConfig | None = None,
) -> BoundaryEvaluationMetrics:
    settings = config or BoundaryEvaluationConfig()
    if not observations:
        raise ValueError("two-branch evaluation requires observations")
    for item in observations:
        _validate_two_branch_observation(item)

    probabilities = np.asarray(
        [forecast.raw_state_probabilities for item in observations for forecast in item.forecasts],
        dtype=np.float64,
    )
    actual_returns = np.asarray(
        [value for item in observations for value in item.actual_returns],
        dtype=np.float64,
    )
    thresholds = np.asarray(
        [value for item in observations for value in item.neutral_thresholds],
        dtype=np.float64,
    )
    actual_classes = np.asarray(
        [_return_class(value, threshold) for value, threshold in zip(actual_returns, thresholds, strict=True)]
    )
    predicted_classes = np.asarray(
        [
            _direction_class(direction)
            for item in observations
            for direction in item.decision.branch_directions
        ],
        dtype=np.int64,
    )
    one_hot = np.eye(3, dtype=np.float64)[actual_classes]
    selected_probability = probabilities[np.arange(len(probabilities)), actual_classes]
    log_loss = -float(np.mean(np.log(np.clip(selected_probability, 1e-12, 1.0))))
    brier = float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1)))
    confidence = probabilities.max(axis=1)
    correct = predicted_classes == actual_classes
    ece = _expected_calibration_error(confidence, correct, settings.calibration_bins)

    flattened = [
        (forecast, actual, threshold)
        for item in observations
        for forecast, actual, threshold in zip(
            item.forecasts, item.actual_returns, item.neutral_thresholds, strict=True
        )
    ]
    crps = float(
        np.mean(
            [
                _normal_crps(
                    forecast.expected_return,
                    sqrt(max(forecast.total_return_variance, 1e-16)),
                    actual,
                )
                for forecast, actual, _ in flattened
            ]
        )
    )
    interval_z = _normal_quantile((1.0 + settings.coverage_probability) * 0.5)
    covered = [
        abs(actual - forecast.expected_return)
        <= interval_z * sqrt(max(forecast.total_return_variance, 0.0))
        for forecast, actual, _ in flattened
    ]
    gross_returns = np.asarray([item.gross_strategy_return for item in observations])
    turnover = np.asarray([item.turnover for item in observations])
    net_returns = gross_returns - turnover * settings.transaction_cost_bps / 10_000.0
    gross_equity = np.cumprod(1.0 + gross_returns)
    net_equity = np.cumprod(1.0 + net_returns)
    exposure = np.asarray(
        [sum(item.actual_weights) for item in observations], dtype=np.float64
    )
    opened = exposure > 0.0
    probability_residual = one_hot[:, 2] - probabilities[:, 2]
    expected_drift = np.asarray(
        [forecast.expected_drift for item in observations for forecast in item.forecasts],
        dtype=np.float64,
    )
    direction_signal = predicted_classes != 1

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
        average_exposure=float(np.mean(exposure)),
        open_rate=float(np.mean(opened)),
        profit_concentration=_profit_concentration(net_returns[opened]),
        direction_counts=tuple(
            int(np.sum(predicted_classes == label)) for label in range(3)
        ),
        drift_rank_ic=_rank_correlation(expected_drift, actual_returns),
        drift_direction_accuracy=float(
            np.mean(np.sign(expected_drift) == np.sign(actual_returns))
        ),
        direction_signal_coverage=float(np.mean(direction_signal)),
        direction_signal_accuracy=(
            float(np.mean(predicted_classes[direction_signal] == actual_classes[direction_signal]))
            if np.any(direction_signal)
            else 0.0
        ),
        failed_buy_orders=int(sum(item.failed_buy_orders for item in observations)),
        failed_sell_orders=int(sum(item.failed_sell_orders for item in observations)),
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


def _validate_two_branch_observation(item: TwoBranchForecastObservation) -> None:
    signal = np.datetime64(item.signal_time, "ns")
    available = np.datetime64(item.available_at, "ns")
    target = np.datetime64(item.target_time, "ns")
    cutoff = np.datetime64(item.training_cutoff, "ns")
    if cutoff > signal or available < signal or available >= target or target <= signal:
        raise ValueError("two-branch forecast timing is not causal")
    branch_count = len(item.forecasts)
    if not 1 <= branch_count <= 8:
        raise ValueError("flow observation requires one to eight forecasts")
    if (
        len(item.actual_returns) != branch_count
        or len(item.neutral_thresholds) != branch_count
        or len(item.decision.branch_directions) != branch_count
    ):
        raise ValueError("branch forecasts, returns, thresholds, and decisions must align")
    if item.branch_ids and (
        len(item.branch_ids) != branch_count
        or len(set(item.branch_ids)) != branch_count
        or any(branch_id < 0 or branch_id >= 8 for branch_id in item.branch_ids)
    ):
        raise ValueError("stable branch ids must be unique values in [0, 7]")
    values = (*item.actual_returns, *item.neutral_thresholds, item.gross_strategy_return, item.turnover)
    if any(not np.isfinite(value) for value in values) or item.turnover < 0.0:
        raise ValueError("two-branch evaluation values must be finite")
    for forecast in item.forecasts:
        probabilities = np.asarray(forecast.raw_state_probabilities)
        if np.any(probabilities < 0.0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-8):
            raise ValueError("branch state probabilities must form a distribution")


def _return_class(value: float, threshold: float) -> int:
    if value < -threshold:
        return 0
    if value > threshold:
        return 2
    return 1


def _direction_class(direction: str) -> int:
    try:
        return {"bear": 0, "flat": 1, "bull": 2}[direction]
    except KeyError as error:
        raise ValueError(f"unknown branch direction: {direction}") from error


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


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if np.std(left_rank) <= 1e-15 or np.std(right_rank) <= 1e-15:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) * 0.5
        start = end
    return ranks


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
