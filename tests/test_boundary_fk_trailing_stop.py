from __future__ import annotations

import pytest

from bnhmm.boundary_fk.trailing_stop import TwoBranchTrailingStop


def test_branch_drawdown_is_measured_from_highest_equity_since_entry() -> None:
    stop = TwoBranchTrailingStop(maximum_drawdown=0.05)

    first = stop.update((0.10, 0.00), (True, False))
    second = stop.update((-0.04, 0.00), (True, False))
    third = stop.update((-0.02, 0.00), (True, False))

    assert first.branch_equities[0] == pytest.approx(1.10)
    assert first.branch_peaks[0] == pytest.approx(1.10)
    assert not any(first.newly_triggered)
    assert second.branch_drawdowns[0] == pytest.approx(0.04)
    assert not any(second.newly_triggered)
    assert third.branch_drawdowns[0] > 0.05
    assert third.newly_triggered == (True, False)
    assert third.liquidation_pending


def test_either_branch_drawdown_locks_global_liquidation_until_flat() -> None:
    stop = TwoBranchTrailingStop(maximum_drawdown=0.05)

    stop.update((0.02, 0.01), (True, True))
    triggered = stop.update((0.00, -0.06), (True, True))
    still_held = stop.update((0.00, -0.01), (False, True))
    released = stop.update((0.00, 0.00), (False, False))

    assert triggered.newly_triggered == (False, True)
    assert triggered.liquidation_pending
    assert still_held.liquidation_pending
    assert not released.liquidation_pending
    assert released.branch_equities == (1.0, 1.0)
    assert released.branch_peaks == (1.0, 1.0)


def test_normal_signal_exit_resets_only_the_empty_branch_episode() -> None:
    stop = TwoBranchTrailingStop(maximum_drawdown=0.05)

    stop.update((0.03, 0.04), (True, True))
    snapshot = stop.update((0.00, 0.01), (False, True))

    assert snapshot.branch_equities[0] == 1.0
    assert snapshot.branch_peaks[0] == 1.0
    assert snapshot.branch_equities[1] == pytest.approx(1.0504)
    assert snapshot.branch_peaks[1] == pytest.approx(1.0504)


def test_exactly_five_percent_drawdown_does_not_trigger() -> None:
    stop = TwoBranchTrailingStop(maximum_drawdown=0.05)

    stop.update((0.10, 0.00), (True, False))
    snapshot = stop.update((-0.05, 0.00), (True, False))

    assert snapshot.branch_drawdowns[0] == pytest.approx(0.05)
    assert not any(snapshot.newly_triggered)
    assert not snapshot.liquidation_pending
