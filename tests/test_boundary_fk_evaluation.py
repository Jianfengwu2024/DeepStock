from __future__ import annotations

import numpy as np
import pytest

from bnhmm.boundary_fk.evaluation import (
    BoundaryEvaluationConfig,
    BoundaryForecastObservation,
    evaluate_boundary_forecasts,
)
from bnhmm.boundary_fk.feynman_kac import BoundaryFeynmanKacForecast
from bnhmm.boundary_fk.policy import BoundaryPolicyDecision


def _forecast(probabilities, mean: float) -> BoundaryFeynmanKacForecast:
    return BoundaryFeynmanKacForecast(
        raw_state_probabilities=probabilities,
        survival_joint_probabilities=tuple(value * 0.7 for value in probabilities),
        trend_survival_probability=0.7,
        expected_return=mean,
        epistemic_return_variance=0.0001,
        aleatoric_return_variance=0.0003,
        total_return_variance=0.0004,
        expected_drift=mean,
        expected_diffusion_variance=0.0003,
        expected_effective_potential=0.35,
        bullish_probability_std=0.05,
    )


def _observation(day: int, probabilities, actual: float, opened: bool, gross: float):
    signal = np.datetime64(f"2024-04-{day:02d}T15:00")
    target = np.datetime64(f"2024-04-{day + 1:02d}T15:00")
    return BoundaryForecastObservation(
        signal_time=signal,
        available_at=signal,
        target_time=target,
        training_cutoff=signal,
        model_version="BoundaryFlowFK-test",
        forecast=_forecast(probabilities, actual * 0.8),
        actual_return=actual,
        neutral_threshold=0.002,
        decision=BoundaryPolicyDecision(
            opened,
            "all_gates_passed" if opened else "raw_bullish_probability",
            0.7,
            ("000001",) if opened else (),
            (1.0,) if opened else (),
        ),
        gross_strategy_return=gross,
    )


def test_evaluation_reports_probability_and_strategy_metrics() -> None:
    observations = (
        _observation(1, (0.05, 0.10, 0.85), 0.02, True, 0.02),
        _observation(2, (0.80, 0.15, 0.05), -0.01, True, -0.01),
        _observation(3, (0.10, 0.80, 0.10), 0.0, False, 0.0),
    )

    metrics = evaluate_boundary_forecasts(
        observations, BoundaryEvaluationConfig(transaction_cost_bps=10.0)
    )

    assert metrics.observations == 3
    assert metrics.balanced_accuracy == pytest.approx(1.0)
    assert metrics.macro_f1 == pytest.approx(1.0)
    assert metrics.open_rate == pytest.approx(2 / 3)
    assert metrics.average_turnover == pytest.approx(4 / 3)
    assert metrics.net_total_return == pytest.approx((1.018 * 0.988) - 1.0)
    assert metrics.maximum_drawdown == pytest.approx(0.012)
    assert 0.0 <= metrics.ece <= 1.0
    assert metrics.crps >= 0.0


def test_evaluation_rejects_noncausal_prediction_trace() -> None:
    observation = _observation(1, (0.1, 0.2, 0.7), 0.01, False, 0.0)
    invalid = BoundaryForecastObservation(
        signal_time=observation.signal_time,
        available_at=observation.available_at,
        target_time=observation.target_time,
        training_cutoff=observation.target_time,
        model_version=observation.model_version,
        forecast=observation.forecast,
        actual_return=observation.actual_return,
        neutral_threshold=observation.neutral_threshold,
        decision=observation.decision,
        gross_strategy_return=observation.gross_strategy_return,
    )

    with pytest.raises(ValueError, match="training_cutoff"):
        evaluate_boundary_forecasts((invalid,))
