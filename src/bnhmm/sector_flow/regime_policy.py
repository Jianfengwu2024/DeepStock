"""Long-only gate joining sector timing with constituent selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .regime_data import SectorRegimeState
from .regime_evaluation import SectorRegimePredictionRecord


@dataclass(frozen=True, slots=True)
class SectorRegimePolicyConfig:
    top_k: int = 10
    bull_probability_threshold: float = 0.6
    maximum_bull_variance: float = 0.02
    transaction_cost_bps: float = 10.0

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("sector policy top_k must be positive")
        if not 1.0 / 3.0 < self.bull_probability_threshold <= 1.0:
            raise ValueError("sector policy bull threshold must exceed random probability")
        if self.maximum_bull_variance < 0.0 or self.transaction_cost_bps < 0.0:
            raise ValueError("sector policy uncertainty and cost cannot be negative")


@dataclass(frozen=True, slots=True)
class StockSelectionSnapshot:
    signal_time: np.datetime64
    symbols: tuple[str, ...]
    relative_scores: NDArray[np.float64]
    next_open_close_returns: NDArray[np.float64]

    def __post_init__(self) -> None:
        shape = (len(self.symbols),)
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("stock selection symbols must be unique and non-empty")
        if self.relative_scores.shape != shape or self.next_open_close_returns.shape != shape:
            raise ValueError("stock selection arrays must match symbols")
        if np.any(~np.isfinite(self.relative_scores)) or np.any(
            ~np.isfinite(self.next_open_close_returns)
        ):
            raise ValueError("stock selection values must be finite")


@dataclass(frozen=True, slots=True)
class SectorPolicyObservation:
    signal_time: np.datetime64
    selected_symbols: tuple[str, ...]
    gross_return: float
    transaction_cost: float
    net_return: float
    index_return: float
    bull_probability: float
    bull_variance: float
    relative_scores: NDArray[np.float64]
    stock_returns: NDArray[np.float64]

    def as_json(self) -> dict[str, object]:
        return {
            "signal_time": str(self.signal_time),
            "selected_symbols": list(self.selected_symbols),
            "gross_return": self.gross_return,
            "transaction_cost": self.transaction_cost,
            "net_return": self.net_return,
            "index_return": self.index_return,
            "bull_probability": self.bull_probability,
            "bull_variance": self.bull_variance,
        }


def apply_sector_regime_policy(
    regimes: Sequence[SectorRegimePredictionRecord],
    snapshots: Sequence[StockSelectionSnapshot],
    config: SectorRegimePolicyConfig | None = None,
) -> tuple[SectorPolicyObservation, ...]:
    policy = config or SectorRegimePolicyConfig()
    by_time = {int(item.signal_time.astype("datetime64[ns]").astype(np.int64)): item for item in snapshots}
    observations: list[SectorPolicyObservation] = []
    for regime in sorted(regimes, key=lambda item: item.signal_time):
        key = int(regime.signal_time.astype("datetime64[ns]").astype(np.int64))
        snapshot = by_time.get(key)
        probability = regime.probability_mean
        bull_probability = float(probability[int(SectorRegimeState.BULL)])
        bull_variance = float(
            regime.probability_variance[int(SectorRegimeState.BULL)]
        )
        confident_bull = (
            int(np.argmax(probability)) == int(SectorRegimeState.BULL)
            and bull_probability >= policy.bull_probability_threshold
            and bull_variance <= policy.maximum_bull_variance
            and snapshot is not None
        )
        selected_symbols: tuple[str, ...] = ()
        gross = 0.0
        cost = 0.0
        scores = np.asarray([], dtype=np.float64)
        returns = np.asarray([], dtype=np.float64)
        if confident_bull and snapshot is not None:
            count = min(policy.top_k, len(snapshot.symbols))
            selected = np.argsort(snapshot.relative_scores)[-count:][::-1]
            selected_symbols = tuple(snapshot.symbols[index] for index in selected)
            gross = float(np.mean(snapshot.next_open_close_returns[selected]))
            cost = 2.0 * policy.transaction_cost_bps / 10_000.0
            scores = snapshot.relative_scores.copy()
            returns = snapshot.next_open_close_returns.copy()
        observations.append(
            SectorPolicyObservation(
                signal_time=regime.signal_time,
                selected_symbols=selected_symbols,
                gross_return=gross,
                transaction_cost=cost,
                net_return=gross - cost,
                index_return=regime.realized_return,
                bull_probability=bull_probability,
                bull_variance=bull_variance,
                relative_scores=scores,
                stock_returns=returns,
            )
        )
    return tuple(observations)


def evaluate_sector_regime_policy(
    observations: Sequence[SectorPolicyObservation],
) -> dict[str, float | int]:
    if not observations:
        raise ValueError("cannot evaluate an empty sector regime policy")
    net = np.asarray([item.net_return for item in observations], dtype=np.float64)
    index = np.asarray([item.index_return for item in observations], dtype=np.float64)
    exposed = np.asarray([bool(item.selected_symbols) for item in observations])
    equity = np.cumprod(1.0 + net)
    running_peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))
    equity_with_start = np.concatenate(([1.0], equity))
    drawdown = equity_with_start / running_peak - 1.0
    rank_correlations: list[float] = []
    for item in observations:
        if len(item.relative_scores) < 2:
            continue
        score_rank = _rank(item.relative_scores)
        return_rank = _rank(item.stock_returns)
        if np.std(score_rank) > 0.0 and np.std(return_rank) > 0.0:
            rank_correlations.append(float(np.corrcoef(score_rank, return_rank)[0, 1]))
    exposed_returns = net[exposed]
    return {
        "sessions": len(observations),
        "exposed_sessions": int(exposed.sum()),
        "exposure_rate": float(np.mean(exposed)),
        "ending_equity": float(equity[-1]),
        "total_return": float(equity[-1] - 1.0),
        "average_session_return": float(np.mean(net)),
        "win_rate": float(np.mean(exposed_returns > 0.0)) if len(exposed_returns) else 0.0,
        "maximum_drawdown": float(np.min(drawdown)),
        "total_transaction_cost": float(
            np.sum([item.transaction_cost for item in observations])
        ),
        "conditional_rank_ic": float(np.mean(rank_correlations))
        if rank_correlations
        else 0.0,
        "always_hold_index_equity": float(np.prod(1.0 + index)),
        "always_hold_index_return": float(np.prod(1.0 + index) - 1.0),
    }


def _rank(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks

