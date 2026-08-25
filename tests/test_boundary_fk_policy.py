from __future__ import annotations

from dataclasses import replace

import pytest

from bnhmm.boundary_fk.feynman_kac import BoundaryFeynmanKacForecast
from bnhmm.boundary_fk.policy import (
    BoundaryPolicy,
    BoundaryPolicyConfig,
    BranchPositionState,
)


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
        expected_return=drift,
        epistemic_return_variance=1e-5,
        aleatoric_return_variance=2e-4,
        total_return_variance=2.1e-4,
        expected_drift=drift,
        expected_diffusion_variance=2e-4,
        expected_effective_potential=0.2,
        bullish_probability_std=0.05,
        expected_attenuated_drift=drift * 0.8,
        bullish_drift_probability={"bear": 0.1, "flat": 0.5, "bull": 0.9}[
            direction
        ],
        drift_signal_to_noise=0.7 if direction != "flat" else 0.0,
    )


@pytest.mark.parametrize(
    ("directions", "branch_exposure", "total_exposure"),
    [
        (("bull", "bull"), (0.5, 0.5), 1.0),
        (("bull", "flat"), (0.5, 0.375), 0.875),
        (("bull", "bear"), (0.0, 0.0), 0.0),
        (("flat", "flat"), (0.375, 0.375), 0.75),
        (("flat", "bear"), (0.0, 0.0), 0.0),
        (("bear", "bear"), (0.0, 0.0), 0.0),
    ],
)
def test_branch_directions_map_to_absolute_capital_exposure(
    directions: tuple[str, str],
    branch_exposure: tuple[float, float],
    total_exposure: float,
) -> None:
    decision = BoundaryPolicy().decide(
        (_forecast(directions[0]), _forecast(directions[1])),
        branches=(("A", "B"), ("C", "D")),
        neutral_thresholds=(0.002, 0.002),
        previous_branch_states=(
            BranchPositionState(exposure=0.5),
            BranchPositionState(exposure=0.5),
        ),
    )

    assert decision.branch_target_exposures == pytest.approx(branch_exposure)
    assert sum(decision.target_weights) == pytest.approx(total_exposure)


def test_bull_and_flat_allocate_each_branch_budget_across_its_members() -> None:
    decision = BoundaryPolicy().decide(
        (_forecast("bull"), _forecast("flat")),
        branches=(("A", "B"), ("C", "D")),
        neutral_thresholds=(0.002, 0.002),
        previous_branch_states=(
            BranchPositionState(exposure=0.5),
            BranchPositionState(exposure=0.5),
        ),
    )

    assert decision.selected_symbols == ("A", "B", "C", "D")
    assert decision.target_weights == pytest.approx((0.25, 0.25, 0.1875, 0.1875))


def test_flat_signal_only_reduces_an_existing_long_position() -> None:
    decision = BoundaryPolicy().decide(
        (_forecast("flat"), _forecast("flat")),
        branches=(("A", "B"), ("C", "D")),
        neutral_thresholds=(0.002, 0.002),
    )

    assert decision.branch_target_exposures == (0.0, 0.0)
    assert decision.branch_position_states == (
        BranchPositionState(),
        BranchPositionState(),
    )


def test_four_consecutive_flat_signals_reduce_full_position_by_quarters() -> None:
    policy = BoundaryPolicy()
    bullish = policy.decide(
        (_forecast("bull"), _forecast("bull")),
        branches=(("A", "B"), ("C", "D")),
        neutral_thresholds=(0.002, 0.002),
    )
    decisions = []
    previous = bullish.branch_position_states
    for _ in range(4):
        decision = policy.decide(
            (_forecast("flat"), _forecast("flat")),
            branches=(("A", "B"), ("C", "D")),
            neutral_thresholds=(0.002, 0.002),
            previous_branch_states=previous,
        )
        decisions.append(decision)
        previous = decision.branch_position_states

    assert [item.branch_target_exposures for item in decisions] == [
        (0.375, 0.375),
        (0.25, 0.25),
        (0.125, 0.125),
        (0.0, 0.0),
    ]
    assert [sum(item.target_weights) for item in decisions] == pytest.approx(
        [0.75, 0.5, 0.25, 0.0]
    )
    assert not decisions[-1].open_position


def test_any_bear_signal_exits_both_branches_immediately() -> None:
    policy = BoundaryPolicy()
    prior = (
        BranchPositionState(exposure=0.125, consecutive_flat_signals=3),
        BranchPositionState(exposure=0.125, consecutive_flat_signals=3),
    )
    decision = policy.decide(
        (_forecast("bull"), _forecast("bear")),
        branches=(("A", "B"), ("C", "D")),
        neutral_thresholds=(0.002, 0.002),
        previous_branch_states=prior,
    )

    assert decision.branch_target_exposures == (0.0, 0.0)
    assert decision.branch_position_states == (
        BranchPositionState(),
        BranchPositionState(),
    )


def test_two_branch_exit_memory_is_independent() -> None:
    decision = BoundaryPolicy().decide(
        (_forecast("flat"), _forecast("flat")),
        branches=(("A", "B"), ("C", "D")),
        neutral_thresholds=(0.002, 0.002),
        previous_branch_states=(
            BranchPositionState(exposure=0.5),
            BranchPositionState(exposure=0.25, consecutive_flat_signals=2),
        ),
    )

    assert decision.branch_target_exposures == (0.375, 0.125)


def test_diffusion_probability_argmax_cannot_force_a_weak_drift_direction() -> None:
    noisy = replace(
        _forecast("bull"),
        raw_state_probabilities=(0.34, 0.26, 0.40),
        bullish_drift_probability=0.55,
        drift_signal_to_noise=0.02,
    )

    decision = BoundaryPolicy().decide(
        (noisy, noisy),
        branches=(("A", "B"), ("C", "D")),
        neutral_thresholds=(0.002, 0.002),
        previous_branch_states=(
            BranchPositionState(exposure=0.5),
            BranchPositionState(exposure=0.5),
        ),
    )

    assert decision.branch_directions == ("flat", "flat")
    assert decision.open_position
    assert sum(decision.target_weights) == pytest.approx(0.75)


def test_label_threshold_does_not_suppress_a_confident_posterior_direction() -> None:
    decision = BoundaryPolicy().decide(
        (_forecast("bull"), _forecast("bear")),
        branches=(("A", "B"), ("C", "D")),
        neutral_thresholds=(1.0, 1.0),
    )

    assert decision.branch_directions == ("bull", "bear")
    assert decision.held_branches == (False, False)


def test_weak_survival_confidence_abstains_even_with_positive_drift() -> None:
    weak_survival = replace(
        _forecast("bull"),
        trend_survival_probability=0.2,
    )

    decision = BoundaryPolicy(
        BoundaryPolicyConfig(minimum_survival_probability=0.35)
    ).decide(
        (weak_survival, weak_survival),
        branches=(("A", "B"), ("C", "D")),
        neutral_thresholds=(0.002, 0.002),
    )

    assert decision.branch_directions == ("flat", "flat")


def test_convection_caps_can_only_reduce_each_branch_exposure() -> None:
    decision = BoundaryPolicy().decide(
        (_forecast("bull"), _forecast("bull")),
        branches=(("A", "B"), ("C", "D")),
        neutral_thresholds=(0.002, 0.002),
        branch_exposure_caps=(0.375, 0.25),
    )

    assert decision.branch_target_exposures == (0.375, 0.25)
    assert sum(decision.target_weights) == pytest.approx(0.625)


def test_convection_cannot_reopen_an_empty_branch() -> None:
    decision = BoundaryPolicy().decide(
        (_forecast("flat"), _forecast("flat")),
        branches=(("A", "B"), ("C", "D")),
        neutral_thresholds=(0.002, 0.002),
        branch_exposure_caps=(0.5, 0.5),
    )

    assert decision.branch_target_exposures == (0.0, 0.0)


def test_extreme_convection_clears_only_its_branch() -> None:
    decision = BoundaryPolicy().decide(
        (_forecast("bull"), _forecast("bull")),
        branches=(("A", "B"), ("C", "D")),
        neutral_thresholds=(0.002, 0.002),
        branch_exposure_caps=(0.0, 0.5),
    )

    assert decision.branch_target_exposures == (0.0, 0.5)
    assert decision.selected_symbols == ("C", "D")


def test_bear_global_exit_overrides_convection_caps() -> None:
    decision = BoundaryPolicy().decide(
        (_forecast("bull"), _forecast("bear")),
        branches=(("A", "B"), ("C", "D")),
        neutral_thresholds=(0.002, 0.002),
        branch_exposure_caps=(0.5, 0.5),
        previous_branch_states=(
            BranchPositionState(exposure=0.5),
            BranchPositionState(exposure=0.5),
        ),
    )

    assert decision.branch_target_exposures == (0.0, 0.0)
