"""Simple long-only policy and A-share constrained execution."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from .types import (
    PercolationDecision,
    PercolationExecution,
    PercolationPrediction,
)


@dataclass(frozen=True, slots=True)
class PercolationPolicy:
    bull_probability_threshold: float = 0.60
    selection_fraction: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        if not 0.5 <= self.bull_probability_threshold <= 1.0:
            raise ValueError("bull probability threshold must lie in [0.5, 1]")
        if not 0.0 < self.selection_fraction <= 1.0:
            raise ValueError("selection fraction must lie in (0, 1]")

    def decide(
        self, prediction: PercolationPrediction, symbols: tuple[str, ...]
    ) -> PercolationDecision:
        if len(symbols) != len(prediction.bull_membership_probability):
            raise ValueError("policy symbols must align with prediction membership")
        bull_probability = prediction.phase_probabilities[3]
        open_position = (
            bull_probability >= self.bull_probability_threshold
            and bull_probability >= max(prediction.phase_probabilities)
        )
        scores = (
            prediction.bull_membership_probability
            - prediction.bear_membership_probability
        )
        weights = np.zeros(len(symbols), dtype=np.float64)
        selected: tuple[str, ...] = ()
        if open_position:
            count = max(1, ceil(len(symbols) * self.selection_fraction))
            order = sorted(range(len(symbols)), key=lambda index: (-scores[index], symbols[index]))
            chosen = order[:count]
            weights[chosen] = 1.0 / count
            selected = tuple(symbols[index] for index in chosen)
        return PercolationDecision(
            open_position=open_position,
            symbols=symbols,
            selected_symbols=selected,
            target_weights=tuple(float(value) for value in weights),
            scores=tuple(float(value) for value in scores),
        )


def execute_percolation_portfolio(
    *,
    symbols: tuple[str, ...],
    previous_weights: np.ndarray,
    decision: PercolationDecision,
    hold_returns: np.ndarray,
    buy_returns: np.ndarray,
    buy_executable: np.ndarray,
    sell_returns: np.ndarray,
    sell_executable: np.ndarray,
    transaction_cost_bps: float = 5.0,
) -> PercolationExecution:
    count = len(symbols)
    arrays = (
        previous_weights,
        hold_returns,
        buy_returns,
        buy_executable,
        sell_returns,
        sell_executable,
    )
    if decision.symbols != symbols or any(np.asarray(item).shape != (count,) for item in arrays):
        raise ValueError("execution inputs must align with symbols")
    if transaction_cost_bps < 0.0 or not np.isfinite(transaction_cost_bps):
        raise ValueError("transaction cost must be finite and nonnegative")
    previous = np.asarray(previous_weights, dtype=np.float64)
    if np.any(previous < 0.0) or float(previous.sum()) > 1.0 + 1e-9:
        raise ValueError("previous weights are invalid")
    target = np.asarray(decision.target_weights, dtype=np.float64)
    actual = previous.copy()
    sold = np.zeros(count, dtype=np.float64)
    bought = np.zeros(count, dtype=np.float64)
    cash = max(0.0, 1.0 - float(previous.sum()))
    turnover = 0.0
    failed_sells = 0
    failed_buys = 0

    for index in range(count):
        requested = max(float(actual[index] - target[index]), 0.0)
        if requested <= 1e-12:
            continue
        if bool(sell_executable[index]) and np.isfinite(sell_returns[index]):
            actual[index] -= requested
            sold[index] += requested
            cash += requested
            turnover += requested
        else:
            failed_sells += 1

    for index in range(count):
        requested = max(float(target[index] - actual[index]), 0.0)
        if requested <= 1e-12:
            continue
        if not bool(buy_executable[index]) or not np.isfinite(buy_returns[index]):
            failed_buys += 1
            continue
        amount = min(requested, cash)
        if amount <= 1e-12:
            continue
        actual[index] += amount
        bought[index] += amount
        cash -= amount
        turnover += amount

    retained = np.maximum(previous - sold, 0.0)
    gross_return = float(
        np.sum(retained * np.nan_to_num(hold_returns))
        + np.sum(sold * np.nan_to_num(sell_returns))
        + np.sum(bought * np.nan_to_num(buy_returns))
    )
    net_return = gross_return - turnover * transaction_cost_bps / 10_000.0
    return PercolationExecution(
        actual_weights=actual,
        cash_weight=float(max(cash, 0.0)),
        gross_return=gross_return,
        net_return=float(net_return),
        turnover=float(turnover),
        failed_buy_orders=failed_buys,
        failed_sell_orders=failed_sells,
    )
