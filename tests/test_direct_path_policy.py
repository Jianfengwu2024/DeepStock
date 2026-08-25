from __future__ import annotations

import pytest

from bnhmm.boundary_fk.direct_policy import DirectPathPolicy


def test_positive_buys_and_negative_sells_each_branch() -> None:
    decision = DirectPathPolicy().decide(
        (0.7, -0.2),
        branches=(("A", "B"), ("C", "D")),
        previous_exposures=(0.0, 0.5),
    )

    assert decision.branch_directions == ("bull", "bear")
    assert decision.branch_target_exposures == (0.5, 0.0)
    assert decision.selected_symbols == ("A", "B")
    assert decision.target_weights == pytest.approx((0.25, 0.25))


@pytest.mark.parametrize("previous", (0.0, 0.5))
def test_zero_keeps_previous_branch_exposure(previous: float) -> None:
    decision = DirectPathPolicy().decide(
        (0.0, 0.0),
        branches=(("A", "B"), ("C", "D")),
        previous_exposures=(previous, 0.0),
    )

    assert decision.branch_directions == ("hold", "hold")
    assert decision.branch_target_exposures == (previous, 0.0)


def test_two_branches_change_independently() -> None:
    decision = DirectPathPolicy().decide(
        (-0.1, 0.4),
        branches=(("A", "B"), ("C", "D", "E", "F")),
        previous_exposures=(0.5, 0.0),
    )

    assert decision.branch_target_exposures == (0.0, 0.5)
    assert decision.selected_symbols == ("C", "D", "E", "F")
    assert decision.target_weights == pytest.approx((0.125,) * 4)


def test_non_finite_path_efficiency_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        DirectPathPolicy().decide(
            (float("nan"), 0.1),
            branches=(("A",), ("B",)),
            previous_exposures=(0.0, 0.0),
        )
