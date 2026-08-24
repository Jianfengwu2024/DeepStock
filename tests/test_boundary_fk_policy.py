from __future__ import annotations

from dataclasses import replace

import pytest

from bnhmm.boundary_fk.feynman_kac import BoundaryFeynmanKacForecast
from bnhmm.boundary_fk.policy import BoundaryPolicy, BoundaryPolicyConfig


class _Ranker:
    def __init__(self) -> None:
        self.calls = 0

    def rank(self, paths, attributions):
        self.calls += 1
        return (
            _Ranked("000003", 0.9),
            _Ranked("000001", 0.8),
            _Ranked("000002", 0.7),
        )


class _Ranked:
    def __init__(self, symbol: str, score: float) -> None:
        self.symbol = symbol
        self.source_proximity_probability = score


def _forecast() -> BoundaryFeynmanKacForecast:
    return BoundaryFeynmanKacForecast(
        raw_state_probabilities=(0.15, 0.20, 0.65),
        survival_joint_probabilities=(0.05, 0.10, 0.55),
        trend_survival_probability=0.70,
        expected_return=0.01,
        epistemic_return_variance=1e-5,
        aleatoric_return_variance=2e-4,
        total_return_variance=2.1e-4,
        expected_drift=0.01,
        expected_diffusion_variance=2e-4,
        expected_effective_potential=0.35,
        bullish_probability_std=0.08,
    )


def test_policy_ranks_only_after_all_five_gates_pass() -> None:
    ranker = _Ranker()
    policy = BoundaryPolicy(BoundaryPolicyConfig(top_k=2), ranker=ranker)

    decision = policy.decide(
        _forecast(), paths=("paths",), attributions=("attrs",), thickness_risk_quantile=0.4
    )

    assert decision.open_position
    assert decision.selected_symbols == ("000003", "000001")
    assert decision.target_weights == pytest.approx((0.5, 0.5))
    assert ranker.calls == 1


def test_failed_gate_stays_flat_without_calling_ranker() -> None:
    ranker = _Ranker()
    policy = BoundaryPolicy(ranker=ranker)
    weak = replace(_forecast(), raw_state_probabilities=(0.25, 0.25, 0.50))

    decision = policy.decide(
        weak, paths=("paths",), attributions=("attrs",), thickness_risk_quantile=0.4
    )

    assert not decision.open_position
    assert decision.gate_reason == "raw_bullish_probability"
    assert ranker.calls == 0


def test_survival_bullish_gate_uses_conditional_probability() -> None:
    ranker = _Ranker()
    policy = BoundaryPolicy(ranker=ranker)
    forecast = replace(
        _forecast(),
        survival_joint_probabilities=(0.04, 0.10, 0.41),
        trend_survival_probability=0.55,
    )

    decision = policy.decide(
        forecast, paths=(), attributions=(), thickness_risk_quantile=0.2
    )

    assert not decision.open_position
    assert decision.gate_reason == "trend_survival_probability"
    assert decision.survival_conditional_bullish_probability == pytest.approx(0.41 / 0.55)
