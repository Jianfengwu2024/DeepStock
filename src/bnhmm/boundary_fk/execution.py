"""Causal A-share execution constraints derived from one-minute bars."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
from numpy.typing import NDArray

from ..sector_flow.types import SectorMinutePanel


@dataclass(frozen=True, slots=True)
class MemberExecutionConstraints:
    sessions: NDArray[np.datetime64]
    hold_returns: NDArray[np.float64]
    buy_returns: NDArray[np.float64]
    buy_executable: NDArray[np.bool_]
    sell_returns: NDArray[np.float64]
    sell_executable: NDArray[np.bool_]


def build_member_execution_constraints(
    panel: SectorMinutePanel,
) -> MemberExecutionConstraints:
    sessions = np.unique(panel.timestamps.astype("datetime64[D]"))
    shape = (len(sessions), len(panel.symbols))
    hold_returns = np.zeros(shape, dtype=np.float64)
    buy_returns = np.full(shape, np.nan, dtype=np.float64)
    sell_returns = np.full(shape, np.nan, dtype=np.float64)
    buy_executable = np.zeros(shape, dtype=np.bool_)
    sell_executable = np.zeros(shape, dtype=np.bool_)
    minute_sessions = panel.timestamps.astype("datetime64[D]")
    last_traded_close = np.full(len(panel.symbols), np.nan, dtype=np.float64)

    for row in range(len(sessions) - 1):
        previous_rows = np.flatnonzero(minute_sessions == sessions[row])
        target_rows = np.flatnonzero(minute_sessions == sessions[row + 1])
        for column, symbol in enumerate(panel.symbols):
            previous_valid = previous_rows[panel.valid[previous_rows, column]]
            if len(previous_valid):
                last_traded_close[column] = float(
                    panel.close[previous_valid[-1], column]
                )
            previous_close = float(last_traded_close[column])
            if not np.isfinite(previous_close):
                continue
            target_valid = target_rows[panel.valid[target_rows, column]]
            if not len(target_valid):
                continue
            last_close = float(panel.close[target_valid[-1], column])
            hold_returns[row, column] = last_close / previous_close - 1.0
            upper_limits, lower_limits = _limit_candidates(symbol, previous_close)

            buy_index = _first_executable_row(
                panel, target_valid, column, upper_limits, upper=True
            )
            if buy_index is not None:
                buy_executable[row, column] = True
                buy_price = _execution_price(
                    panel, buy_index, column, upper_limits, upper=True
                )
                buy_returns[row, column] = last_close / buy_price - 1.0

            sell_index = _first_executable_row(
                panel, target_valid, column, lower_limits, upper=False
            )
            if sell_index is not None:
                sell_executable[row, column] = True
                sell_price = _execution_price(
                    panel, sell_index, column, lower_limits, upper=False
                )
                sell_returns[row, column] = sell_price / previous_close - 1.0

    return MemberExecutionConstraints(
        sessions=sessions,
        hold_returns=hold_returns,
        buy_returns=buy_returns,
        buy_executable=buy_executable,
        sell_returns=sell_returns,
        sell_executable=sell_executable,
    )


def _limit_candidates(
    symbol: str, previous_close: float
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    normalized = symbol.split(".")[0]
    if normalized.startswith(("300", "301", "688", "689")):
        primary = 0.20
    elif normalized.startswith(("4", "8", "92")):
        primary = 0.30
    else:
        primary = 0.10
    rates = tuple(dict.fromkeys((primary, 0.05)))
    upper = tuple(_round_price(previous_close * (1.0 + rate)) for rate in rates)
    lower = tuple(_round_price(previous_close * (1.0 - rate)) for rate in rates)
    return upper, lower


def _round_price(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _first_executable_row(
    panel: SectorMinutePanel,
    rows: NDArray[np.int64],
    column: int,
    limits: tuple[float, ...],
    *,
    upper: bool,
) -> int | None:
    if not _bar_locked(panel, int(rows[0]), column, limits, upper=upper):
        return int(rows[0])
    for row in rows[1:]:
        if not _bar_locked(panel, int(row), column, limits, upper=upper):
            return int(row)
    return None


def _bar_locked(
    panel: SectorMinutePanel,
    row: int,
    column: int,
    limits: tuple[float, ...],
    *,
    upper: bool,
) -> bool:
    open_price = float(panel.open[row, column])
    close_price = float(panel.close[row, column])
    for limit in limits:
        if abs(open_price - limit) <= 0.005 and abs(close_price - limit) <= 0.005:
            return True
    return False


def _execution_price(
    panel: SectorMinutePanel,
    row: int,
    column: int,
    limits: tuple[float, ...],
    *,
    upper: bool,
) -> float:
    open_price = float(panel.open[row, column])
    if not any(abs(open_price - limit) <= 0.005 for limit in limits):
        return open_price
    return float(panel.close[row, column])
