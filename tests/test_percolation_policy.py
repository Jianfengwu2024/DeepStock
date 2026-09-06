from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from bnhmm.percolation_branching.policy import (
    PercolationPolicy,
    execute_percolation_portfolio,
)
from bnhmm.percolation_branching.types import PercolationPrediction


def _prediction(probabilities: tuple[float, float, float, float]) -> PercolationPrediction:
    return PercolationPrediction(
        phase_probabilities=probabilities,
        bull_reproduction_mean=1.2,
        bear_reproduction_mean=0.7,
        bull_supercritical_probability=0.8,
        bear_supercritical_probability=0.2,
        bull_giant_fraction_mean=0.7,
        bear_giant_fraction_mean=0.2,
        bull_membership_probability=np.asarray([0.9, 0.7, 0.4, 0.8, 0.2]),
        bear_membership_probability=np.asarray([0.1, 0.2, 0.3, 0.2, 0.1]),
        signal_time=np.datetime64("2025-01-30T15:01"),
        available_at=np.datetime64("2025-01-31T09:25"),
        target_session=np.datetime64("2025-01-31"),
        training_cutoff=np.datetime64("2025-01-30"),
    )


def test_only_confident_bull_phase_opens_position() -> None:
    symbols = tuple(f"S{i}" for i in range(5))
    decision = PercolationPolicy().decide(
        _prediction((0.08, 0.20, 0.11, 0.61)), symbols
    )

    assert decision.open_position
    assert len(decision.selected_symbols) == math.ceil(len(symbols) / 3)
    assert sum(decision.target_weights) == pytest.approx(1.0)
    assert decision.selected_symbols == ("S0", "S3")


@pytest.mark.parametrize("p_bull", [0.0, 0.40, 0.5999])
def test_every_non_confident_phase_is_cash(p_bull: float) -> None:
    remainder = 1.0 - p_bull
    probabilities = (remainder * 0.4, remainder * 0.4, remainder * 0.2, p_bull)
    decision = PercolationPolicy().decide(
        _prediction(probabilities), tuple(f"S{i}" for i in range(5))
    )
    assert not decision.open_position
    assert sum(decision.target_weights) == 0.0


def test_execution_keeps_failed_sell_and_leaves_failed_buy_as_cash() -> None:
    symbols = ("A", "B", "C")
    decision = PercolationPolicy(selection_fraction=1 / 3).decide(
        replace(
            _prediction((0.05, 0.10, 0.10, 0.75)),
            bull_membership_probability=np.asarray([0.1, 0.9, 0.2]),
            bear_membership_probability=np.asarray([0.0, 0.0, 0.0]),
        ),
        symbols,
    )
    execution = execute_percolation_portfolio(
        symbols=symbols,
        previous_weights=np.asarray([0.5, 0.0, 0.0]),
        decision=decision,
        hold_returns=np.asarray([-0.10, 0.02, 0.0]),
        buy_returns=np.asarray([0.0, 0.02, 0.0]),
        buy_executable=np.asarray([True, False, True]),
        sell_returns=np.asarray([-0.10, 0.0, 0.0]),
        sell_executable=np.asarray([False, True, True]),
        transaction_cost_bps=5.0,
    )

    assert execution.actual_weights[0] > 0.0
    assert execution.actual_weights[1] == 0.0
    assert execution.failed_sell_orders == 1
    assert execution.failed_buy_orders == 1
    assert execution.cash_weight > 0.0
