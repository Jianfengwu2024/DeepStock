"""Monthly slow candidate graph for directional propagation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import ActivationPanel, MonthlyBackboneSupport


@dataclass(frozen=True, slots=True)
class SlowBackboneBuilder:
    lookback_sessions: int = 60
    candidate_neighbors: int = 10
    minimum_complete_fraction: float = 0.8
    correlation_shrinkage: float = 60.0

    def __post_init__(self) -> None:
        if self.lookback_sessions < 2:
            raise ValueError("backbone lookback must contain multiple sessions")
        if self.candidate_neighbors < 1:
            raise ValueError("candidate neighbor count must be positive")
        if not 0.0 < self.minimum_complete_fraction <= 1.0:
            raise ValueError("complete fraction must be in (0, 1]")

    def build(
        self, events: ActivationPanel, month: np.datetime64
    ) -> MonthlyBackboneSupport:
        month_value = np.datetime64(month, "M")
        month_start = month_value.astype("datetime64[D]")
        row_sessions = events.timestamps.astype("datetime64[D]")
        sessions = np.unique(row_sessions[row_sessions < month_start])
        if len(sessions) < self.lookback_sessions:
            raise ValueError("insufficient completed sessions for monthly backbone")
        sessions = sessions[-self.lookback_sessions :]
        rows = np.isin(row_sessions, sessions)
        velocity = events.standardized_velocity[rows].copy()
        valid = events.valid[rows] & np.isfinite(velocity)
        selected_sessions = row_sessions[rows]
        completeness = np.asarray(
            [
                np.mean(
                    [
                        bool(np.any(valid[selected_sessions == session, column]))
                        for session in sessions
                    ]
                )
                for column in range(len(events.symbols))
            ],
            dtype=np.float64,
        )
        eligible = completeness >= self.minimum_complete_fraction
        raw = _pairwise_correlation(velocity, valid, self.correlation_shrinkage)
        with np.errstate(invalid="ignore"):
            common = np.nanmean(np.where(valid, velocity, np.nan), axis=1)
        common = np.where(np.isfinite(common), common, 0.0)
        residual = velocity - common[:, None]
        residual_corr = _pairwise_correlation(
            residual, valid, self.correlation_shrinkage
        )
        similarity = 0.5 * raw + 0.5 * residual_corr
        similarity[~np.outer(eligible, eligible)] = 0.0
        np.fill_diagonal(similarity, 0.0)
        adjacency = np.zeros_like(similarity, dtype=np.bool_)
        for source in range(len(events.symbols)):
            positive = np.flatnonzero(similarity[source] > 0.0)
            if len(positive) == 0:
                continue
            order = positive[np.argsort(similarity[source, positive], kind="stable")]
            chosen = order[-min(self.candidate_neighbors, len(order)) :]
            adjacency[source, chosen] = True
        adjacency |= adjacency.T
        np.fill_diagonal(adjacency, False)
        return MonthlyBackboneSupport(
            month=month_value,
            training_cutoff=sessions[-1],
            symbols=events.symbols,
            adjacency=adjacency,
            similarity=similarity,
            session_count=len(sessions),
            completeness=completeness,
        )


def _pairwise_correlation(
    values: np.ndarray, valid: np.ndarray, shrinkage: float
) -> np.ndarray:
    symbols = values.shape[1]
    result = np.zeros((symbols, symbols), dtype=np.float64)
    for left in range(symbols):
        for right in range(left + 1, symbols):
            mask = valid[:, left] & valid[:, right]
            count = int(mask.sum())
            if count < 3:
                continue
            x = values[mask, left]
            y = values[mask, right]
            x = x - float(x.mean())
            y = y - float(y.mean())
            denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
            correlation = float(np.dot(x, y) / denominator) if denominator > 0.0 else 0.0
            correlation *= count / (count + shrinkage)
            result[left, right] = correlation
            result[right, left] = correlation
    return result
