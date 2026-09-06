from __future__ import annotations

import numpy as np
import pytest

from bnhmm.percolation_branching.evaluation import (
    evaluate_percolation,
    fit_phase_baselines,
)
from bnhmm.percolation_branching.policy import PercolationPolicy
from bnhmm.percolation_branching.types import (
    PercolationEvaluationObservation,
    PercolationExecution,
    PercolationPrediction,
)


def _prediction(label: str, day: int) -> PercolationPrediction:
    labels = ("bear", "none", "conflict", "bull")
    probabilities = np.full(4, 0.02)
    probabilities[labels.index(label)] = 0.94
    signal_day = np.datetime64("2025-01-01") + np.timedelta64(day, "D")
    target = signal_day + np.timedelta64(1, "D")
    return PercolationPrediction(
        phase_probabilities=tuple(probabilities.tolist()),
        bull_reproduction_mean=1.1,
        bear_reproduction_mean=0.9,
        bull_supercritical_probability=0.6,
        bear_supercritical_probability=0.4,
        bull_giant_fraction_mean=0.5,
        bear_giant_fraction_mean=0.4,
        bull_membership_probability=np.asarray([0.8, 0.5, 0.2]),
        bear_membership_probability=np.asarray([0.1, 0.2, 0.4]),
        signal_time=signal_day + np.timedelta64(15 * 60 + 1, "m"),
        available_at=target + np.timedelta64(9 * 60 + 25, "m"),
        target_session=target,
        training_cutoff=signal_day,
    )


def test_evaluation_reports_prediction_and_economic_metrics() -> None:
    labels = ("bull", "none", "bear", "conflict")
    observations = []
    for day, label in enumerate(labels):
        prediction = _prediction(label, day)
        decision = PercolationPolicy().decide(prediction, ("A", "B", "C"))
        execution = PercolationExecution(
            actual_weights=np.asarray(decision.target_weights),
            cash_weight=1.0 - sum(decision.target_weights),
            gross_return=0.01 if decision.open_position else 0.0,
            net_return=0.009 if decision.open_position else 0.0,
            turnover=1.0 if decision.open_position else 0.0,
            failed_buy_orders=0,
            failed_sell_orders=0,
        )
        observations.append(
            PercolationEvaluationObservation(prediction, label, decision, execution)
        )

    metrics = evaluate_percolation(tuple(observations))

    assert metrics.balanced_accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.confusion_matrix.shape == (4, 4)
    assert metrics.net_total_return > 0.0
    assert metrics.average_exposure == pytest.approx(0.25)


def test_baselines_are_fitted_from_prior_labels_only() -> None:
    labels = ("none", "none", "bull", "bear", "bull")
    baselines = fit_phase_baselines(labels[:-1], current_phase=labels[-2])

    assert baselines.majority == "none"
    assert baselines.persistence == "bear"
    assert baselines.transition in ("bear", "bull", "none", "conflict")
    assert baselines.simple_reproduction in ("bear", "bull", "none", "conflict")

