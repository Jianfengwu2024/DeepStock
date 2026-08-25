from __future__ import annotations

from dataclasses import replace

import pytest

from bnhmm.boundary_fk.feynman_kac import BoundaryFeynmanKacForecast
from bnhmm.boundary_fk.policy import BoundaryPolicy


def _forecast(direction: str) -> BoundaryFeynmanKacForecast:
    probabilities = {
        "bear": (0.75, 0.15, 0.10),
        "flat": (0.15, 0.70, 0.15),
        "bull": (0.10, 0.15, 0.75),
    }[direction]
    drift = {"bear": -0.01, "flat": 0.0, "bull": 0.01}[direction]
    return BoundaryFeynmanKacForecast(
        raw_state_probabilities=probabilities,
        survival_joint_probabilities=tuple(value * 0.8 for value in probabilities),
        trend_survival_probability=0.8,
        expected_return=drift * 0.8,
        epistemic_return_variance=1e-5,
        aleatoric_return_variance=2e-4,
        total_return_variance=2.1e-4,
        expected_drift=drift,
        expected_diffusion_variance=2e-4,
        expected_effective_potential=0.2,
        bullish_probability_std=0.05,
    )


def test_bull_opens_and_bear_closes_each_branch_independently() -> None:
    policy = BoundaryPolicy()
    opened = policy.decide(
        (_forecast("bull"), _forecast("flat")),
        branches=(("A", "B"), ("C", "D")),
        held_branches=(False, False),
        neutral_thresholds=(0.002, 0.002),
    )
    closed = policy.decide(
        (_forecast("bear"), _forecast("bull")),
        branches=(("A", "B"), ("C", "D")),
        held_branches=opened.held_branches,
        neutral_thresholds=(0.002, 0.002),
    )

    assert opened.held_branches == (True, False)
    assert opened.selected_symbols == ("A", "B")
    assert closed.held_branches == (False, True)
    assert closed.selected_symbols == ("C", "D")


def test_flat_keeps_existing_position_but_does_not_open_a_new_one() -> None:
    decision = BoundaryPolicy().decide(
        (_forecast("flat"), _forecast("flat")),
        branches=(("A", "B"), ("C", "D")),
        held_branches=(True, False),
        neutral_thresholds=(0.002, 0.002),
    )

    assert decision.branch_directions == ("flat", "flat")
    assert decision.held_branches == (True, False)
    assert decision.selected_symbols == ("A", "B")


def test_all_held_symbols_receive_one_equal_portfolio_weight() -> None:
    decision = BoundaryPolicy().decide(
        (_forecast("bull"), _forecast("bull")),
        branches=(("A", "B"), ("C", "D")),
        held_branches=(False, False),
        neutral_thresholds=(0.002, 0.002),
    )

    assert decision.selected_symbols == ("A", "B", "C", "D")
    assert decision.target_weights == pytest.approx((0.25, 0.25, 0.25, 0.25))


def test_diffusion_probability_argmax_cannot_force_a_weak_drift_direction() -> None:
    noisy = replace(
        _forecast("bull"),
        raw_state_probabilities=(0.34, 0.26, 0.40),
        expected_return=0.0005,
    )

    decision = BoundaryPolicy().decide(
        (noisy, noisy),
        branches=(("A", "B"), ("C", "D")),
        held_branches=(False, False),
        neutral_thresholds=(0.002, 0.002),
    )

    assert decision.branch_directions == ("flat", "flat")
    assert not decision.open_position
