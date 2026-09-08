from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bnhmm.boundary_fk.execution import build_member_execution_constraints
from bnhmm.boundary_fk.pipeline import BoundaryFlowPipeline, BoundaryFlowPipelineConfig
from bnhmm.sector_flow.types import SectorMinutePanel
from test_boundary_fk_pipeline import _walk_forward_data


def _panel(next_prices: tuple[tuple[float, float], ...]) -> SectorMinutePanel:
    timestamps = np.asarray(
        [
            "2026-01-05T14:59",
            "2026-01-05T15:00",
            "2026-01-06T09:31",
            "2026-01-06T09:32",
            "2026-01-06T09:33",
        ],
        dtype="datetime64[m]",
    )
    prices = ((10.0, 10.0), (10.0, 10.0), *next_prices)
    open_prices = np.asarray([[item[0]] for item in prices], dtype=np.float64)
    close_prices = np.asarray([[item[1]] for item in prices], dtype=np.float64)
    return SectorMinutePanel(
        symbols=("600000",),
        timestamps=timestamps,
        open=open_prices,
        close=close_prices,
        volume=np.ones_like(open_prices),
        valid=np.ones_like(open_prices, dtype=np.bool_),
    )


def test_boundary_fk_defaults_to_five_basis_points() -> None:
    assert BoundaryFlowPipelineConfig().transaction_cost_bps == 5.0
    assert BoundaryFlowPipelineConfig().maximum_drawdown == 0.05
    assert BoundaryFlowPipelineConfig().minimum_direction_probability == 0.51


def test_all_day_limit_up_rejects_new_position() -> None:
    constraints = build_member_execution_constraints(
        _panel(((11.0, 11.0), (11.0, 11.0), (11.0, 11.0)))
    )

    assert not constraints.buy_executable[0, 0]
    assert constraints.sell_executable[0, 0]


def test_limit_up_that_opens_intraday_can_be_bought_at_first_open() -> None:
    constraints = build_member_execution_constraints(
        _panel(((11.0, 11.0), (11.0, 10.8), (10.8, 10.9)))
    )

    assert constraints.buy_executable[0, 0]
    np.testing.assert_allclose(constraints.buy_returns[0, 0], 10.9 / 10.8 - 1.0)
    np.testing.assert_allclose(constraints.hold_returns[0, 0], 10.9 / 10.0 - 1.0)


def test_all_day_limit_down_rejects_sale_and_keeps_position() -> None:
    data = _walk_forward_data(180)
    row = 120
    sellable = data.member_next_valid.copy()
    sellable[row, 0] = False
    data = replace(
        data,
        member_next_buy_returns=data.member_next_returns,
        member_next_buy_executable=data.member_next_valid,
        member_next_sell_returns=np.zeros_like(data.member_next_returns),
        member_next_sell_executable=sellable,
    )

    execution = BoundaryFlowPipeline._execute_portfolio(
        data,
        row,
        previous={"A": 0.5},
        target={},
    )

    assert execution.weights == {"A": 0.5}
    assert execution.turnover == 0.0
    assert execution.failed_sell_orders == 1
    np.testing.assert_allclose(
        execution.gross_return, 0.5 * data.member_next_returns[row, 0]
    )


def test_limit_down_that_opens_intraday_can_be_sold_at_first_open() -> None:
    constraints = build_member_execution_constraints(
        _panel(((9.0, 9.0), (9.0, 9.2), (9.2, 9.1)))
    )

    assert constraints.sell_executable[0, 0]
    np.testing.assert_allclose(constraints.sell_returns[0, 0], 9.2 / 10.0 - 1.0)


def test_turnover_and_cost_use_only_executed_orders() -> None:
    data = _walk_forward_data(180)
    row = 120
    buyable = data.member_next_valid.copy()
    buyable[row, 0] = False
    data = replace(
        data,
        member_next_buy_returns=data.member_next_returns,
        member_next_buy_executable=buyable,
        member_next_sell_returns=np.zeros_like(data.member_next_returns),
        member_next_sell_executable=data.member_next_valid,
    )

    execution = BoundaryFlowPipeline._execute_portfolio(
        data,
        row,
        previous={},
        target={"A": 0.5, "B": 0.5},
    )

    assert execution.weights == {"B": 0.5}
    assert execution.turnover == 0.5
    assert execution.failed_buy_orders == 1
    np.testing.assert_allclose(
        execution.gross_return, 0.5 * data.member_next_buy_returns[row, 1]
    )


def test_existing_position_uses_previous_close_to_current_close_return() -> None:
    data = _walk_forward_data(180)
    row = 120
    data = replace(
        data,
        member_next_hold_returns=np.full_like(data.member_next_returns, 0.04),
    )

    execution = BoundaryFlowPipeline._execute_portfolio(
        data,
        row,
        previous={"A": 0.5},
        target={"A": 0.5},
    )

    np.testing.assert_allclose(execution.gross_return, 0.02)


def test_execution_attributes_actual_returns_to_each_branch_sleeve() -> None:
    data = _walk_forward_data(180)
    row = 120
    hold_returns = np.zeros_like(data.member_next_returns)
    hold_returns[row, 0] = 0.04
    hold_returns[row, 2] = -0.06
    data = replace(data, member_next_hold_returns=hold_returns)

    execution = BoundaryFlowPipeline._execute_portfolio(
        data,
        row,
        previous={"A": 0.5, "C": 0.5},
        target={"A": 0.5, "C": 0.5},
        previous_symbol_branches={"A": 0, "C": 1},
        target_symbol_branches={"A": 0, "C": 1},
    )

    assert execution.branch_returns == pytest.approx((0.04, -0.06))
    assert execution.symbol_branches == {"A": 0, "C": 1}


def test_failed_limit_down_sale_keeps_branch_ownership_for_stop_lock() -> None:
    data = _walk_forward_data(180)
    row = 120
    sellable = data.member_next_valid.copy()
    sellable[row, 0] = False
    data = replace(
        data,
        member_next_hold_returns=np.full_like(data.member_next_returns, -0.06),
        member_next_sell_executable=sellable,
    )

    execution = BoundaryFlowPipeline._execute_portfolio(
        data,
        row,
        previous={"A": 0.5},
        target={},
        previous_symbol_branches={"A": 0},
        target_symbol_branches={},
    )

    assert execution.weights == {"A": 0.5}
    assert execution.symbol_branches == {"A": 0}
    assert execution.branch_returns == pytest.approx((-0.06, 0.0))
