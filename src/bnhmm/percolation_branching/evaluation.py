"""Four-phase prediction metrics, baselines, and portfolio diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .types import (
    PHASE_NAMES,
    PercolationEvaluationObservation,
    PercolationMetrics,
    PhaseBaselines,
    PhaseName,
)


def evaluate_percolation(
    observations: tuple[PercolationEvaluationObservation, ...],
    transaction_cost_bps: float | None = None,
    calibration_bins: int = 10,
) -> PercolationMetrics:
    if not observations:
        raise ValueError("percolation evaluation requires observations")
    if calibration_bins < 2:
        raise ValueError("calibration bins must be at least two")
    actual = np.asarray([PHASE_NAMES.index(item.actual_phase) for item in observations])
    probabilities = np.asarray(
        [item.prediction.phase_probabilities for item in observations], dtype=np.float64
    )
    predicted = np.argmax(probabilities, axis=1)
    confusion = np.zeros((4, 4), dtype=np.int64)
    for target, forecast in zip(actual, predicted, strict=True):
        confusion[target, forecast] += 1
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for label in range(4):
        true_positive = int(confusion[label, label])
        predicted_count = int(confusion[:, label].sum())
        actual_count = int(confusion[label].sum())
        label_precision = true_positive / predicted_count if predicted_count else 0.0
        label_recall = true_positive / actual_count if actual_count else 0.0
        precision.append(label_precision)
        recall.append(label_recall)
        f1.append(
            2 * label_precision * label_recall / (label_precision + label_recall)
            if label_precision + label_recall
            else 0.0
        )
    present = confusion.sum(axis=1) > 0
    balanced_accuracy = float(np.mean(np.asarray(recall)[present]))
    macro_f1 = float(np.mean(np.asarray(f1)[present]))
    one_hot = np.eye(4)[actual]
    brier = float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1)))
    log_loss = -float(
        np.mean(np.log(np.clip(probabilities[np.arange(len(actual)), actual], 1e-12, 1.0)))
    )
    confidence = probabilities.max(axis=1)
    correct = predicted == actual
    ece = _expected_calibration_error(confidence, correct, calibration_bins)
    gross_returns = np.asarray([item.execution.gross_return for item in observations])
    if transaction_cost_bps is None:
        net_returns = np.asarray([item.execution.net_return for item in observations])
    else:
        if transaction_cost_bps < 0.0:
            raise ValueError("transaction cost cannot be negative")
        turnover = np.asarray([item.execution.turnover for item in observations])
        net_returns = gross_returns - turnover * transaction_cost_bps / 10_000.0
    gross_equity = np.cumprod(1.0 + gross_returns)
    net_equity = np.cumprod(1.0 + net_returns)
    exposure = np.asarray([item.execution.actual_weights.sum() for item in observations])
    turnover = np.asarray([item.execution.turnover for item in observations])
    return PercolationMetrics(
        observations=len(observations),
        confusion_matrix=confusion,
        balanced_accuracy=balanced_accuracy,
        macro_f1=macro_f1,
        precision=tuple(float(value) for value in precision),
        recall=tuple(float(value) for value in recall),
        brier_score=brier,
        log_loss=log_loss,
        ece=ece,
        gross_total_return=float(gross_equity[-1] - 1.0),
        net_total_return=float(net_equity[-1] - 1.0),
        maximum_drawdown=_maximum_drawdown(net_equity),
        average_exposure=float(exposure.mean()),
        average_turnover=float(turnover.mean()),
        open_rate=float(np.mean(exposure > 0.0)),
        failed_buy_orders=sum(item.execution.failed_buy_orders for item in observations),
        failed_sell_orders=sum(item.execution.failed_sell_orders for item in observations),
    )


def fit_phase_baselines(
    mature_labels: tuple[PhaseName, ...],
    *,
    current_phase: PhaseName,
    bull_reproduction: float | None = None,
    bear_reproduction: float | None = None,
) -> PhaseBaselines:
    if not mature_labels:
        raise ValueError("phase baselines require mature labels")
    if current_phase not in PHASE_NAMES or any(label not in PHASE_NAMES for label in mature_labels):
        raise ValueError("unknown phase label")
    majority = _mode(mature_labels)
    transitions: dict[PhaseName, list[PhaseName]] = defaultdict(list)
    for left, right in zip(mature_labels[:-1], mature_labels[1:], strict=True):
        transitions[left].append(right)
    transition = _mode(tuple(transitions[current_phase])) if transitions[current_phase] else majority
    if bull_reproduction is None or bear_reproduction is None:
        simple = current_phase
    else:
        bull = bull_reproduction > 1.0
        bear = bear_reproduction > 1.0
        simple = "conflict" if bull and bear else "bull" if bull else "bear" if bear else "none"
    return PhaseBaselines(
        majority=majority,
        persistence=current_phase,
        transition=transition,
        simple_reproduction=simple,
    )


def _mode(labels: tuple[PhaseName, ...]) -> PhaseName:
    counts = Counter(labels)
    return max(PHASE_NAMES, key=lambda label: (counts[label], -PHASE_NAMES.index(label)))


def _expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, bins: int
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == bins - 1
            else confidence < edges[index + 1]
        )
        if np.any(selected):
            result += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return result


def _maximum_drawdown(equity: np.ndarray) -> float:
    values = np.concatenate((np.ones(1), equity))
    peaks = np.maximum.accumulate(values)
    return float(np.max(1.0 - values / peaks))
