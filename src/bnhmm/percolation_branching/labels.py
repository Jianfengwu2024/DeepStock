"""Temporal event reachability and causal percolation phase labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .types import Direction, NullThresholdTable, PhaseName, ReachabilityResult


def temporal_reachability(
    events: NDArray[np.bool_],
    adjacency: NDArray[np.bool_],
    *,
    maximum_lag: int = 3,
) -> ReachabilityResult:
    """Measure directed reachability using only strictly later events."""
    event_matrix = np.asarray(events, dtype=np.bool_)
    support = np.asarray(adjacency, dtype=np.bool_)
    if event_matrix.ndim != 2:
        raise ValueError("events must be a minute-by-symbol matrix")
    minutes, symbols = event_matrix.shape
    if symbols == 0 or support.shape != (symbols, symbols):
        raise ValueError("adjacency must align with event symbols")
    if maximum_lag < 1:
        raise ValueError("maximum lag must be positive")

    event_sources = np.zeros((minutes, symbols), dtype=object)
    source_membership = np.zeros((symbols, symbols), dtype=np.bool_)
    edge_count = 0
    for minute in range(minutes):
        for child in np.flatnonzero(event_matrix[minute]):
            sources = 1 << int(child)
            source_membership[child, child] = True
            for lag in range(1, min(maximum_lag, minute) + 1):
                parent_minute = minute - lag
                parents = np.flatnonzero(
                    event_matrix[parent_minute] & support[:, child]
                )
                for parent in parents:
                    if parent == child:
                        continue
                    parent_sources = int(event_sources[parent_minute, parent])
                    if parent_sources and parent_sources & ~sources:
                        sources |= parent_sources
                        edge_count += 1
            event_sources[minute, child] = sources
            for source in range(symbols):
                if sources & (1 << source):
                    source_membership[source, child] = True

    giant = (
        float(np.max(source_membership.sum(axis=1)) / symbols)
        if np.any(event_matrix)
        else 0.0
    )
    density = float(event_matrix.mean()) if event_matrix.size else 0.0
    return ReachabilityResult(
        giant_fraction=giant,
        source_membership=source_membership,
        event_density=density,
        edge_count=edge_count,
    )


def classify_phase(
    bull_fraction: float,
    bear_fraction: float,
    bull_threshold: float,
    bear_threshold: float,
) -> PhaseName:
    values = (bull_fraction, bear_fraction, bull_threshold, bear_threshold)
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("phase fractions and thresholds must be probabilities")
    bull = bull_fraction >= bull_threshold
    bear = bear_fraction >= bear_threshold
    if bull and bear:
        return "conflict"
    if bull:
        return "bull"
    if bear:
        return "bear"
    return "none"


@dataclass(frozen=True, slots=True)
class NullThresholdCalibrator:
    permutations: int = 500
    density_bins: int = 4
    quantile: float = 0.95
    maximum_lag: int = 3

    def __post_init__(self) -> None:
        if self.permutations < 1:
            raise ValueError("permutations must be positive")
        if self.density_bins < 1:
            raise ValueError("density bins must be positive")
        if not 0.5 <= self.quantile < 1.0:
            raise ValueError("null quantile must be in [0.5, 1)")
        if self.maximum_lag < 1:
            raise ValueError("maximum lag must be positive")

    def fit(
        self,
        events: NDArray[np.bool_],
        timestamps: NDArray[np.datetime64],
        support: NDArray[np.bool_],
        training_sessions: Iterable[np.datetime64],
        *,
        direction: Direction,
        month: np.datetime64,
        rng: np.random.Generator,
    ) -> NullThresholdTable:
        matrix = np.asarray(events, dtype=np.bool_)
        times = np.asarray(timestamps)
        if matrix.ndim != 2 or matrix.shape[0] != len(times):
            raise ValueError("events and timestamps must align")
        sessions = np.asarray(tuple(training_sessions), dtype="datetime64[D]")
        if len(sessions) == 0:
            raise ValueError("null calibration requires training sessions")
        sessions = np.unique(sessions)
        month_start = np.datetime64(month, "M").astype("datetime64[D]")
        if sessions[-1] >= month_start:
            raise ValueError("training sessions must precede the target month")
        row_sessions = times.astype("datetime64[D]")
        session_rows = [np.flatnonzero(row_sessions == session) for session in sessions]
        if any(len(rows) == 0 for rows in session_rows):
            raise ValueError("every training session must have event rows")

        densities = np.asarray([float(matrix[rows].mean()) for rows in session_rows])
        edges = np.quantile(
            densities, np.linspace(0.0, 1.0, self.density_bins + 1)
        ).astype(np.float64)
        edges[0] = 0.0
        edges[-1] = max(1.0, float(edges[-1]))
        edges = np.maximum.accumulate(edges)
        samples: list[list[float]] = [[] for _ in range(self.density_bins)]

        for _ in range(self.permutations):
            for density, rows in zip(densities, session_rows, strict=True):
                shuffled = self._shuffle_session(matrix[rows], rng)
                result = temporal_reachability(
                    shuffled, support, maximum_lag=self.maximum_lag
                )
                index = int(np.searchsorted(edges[1:-1], density, side="right"))
                samples[min(index, self.density_bins - 1)].append(
                    result.giant_fraction
                )

        pooled = [value for bucket in samples for value in bucket]
        fallback = float(np.quantile(pooled, self.quantile)) if pooled else 1.0
        thresholds = np.asarray(
            [
                float(np.quantile(bucket, self.quantile)) if bucket else fallback
                for bucket in samples
            ],
            dtype=np.float64,
        )
        return NullThresholdTable(
            month=np.datetime64(month, "M"),
            training_cutoff=sessions[-1],
            direction=direction,
            permutations=self.permutations,
            density_edges=edges,
            thresholds=np.clip(thresholds, 0.0, 1.0),
        )

    @staticmethod
    def _shuffle_session(
        events: NDArray[np.bool_], rng: np.random.Generator
    ) -> NDArray[np.bool_]:
        minutes, symbols = events.shape
        shuffled = np.zeros_like(events)
        bucket_ids = np.minimum(np.arange(minutes) * 4 // max(minutes, 1), 3)
        for symbol in range(symbols):
            event_rows = np.flatnonzero(events[:, symbol])
            for bucket in range(4):
                count = int(np.sum(bucket_ids[event_rows] == bucket))
                candidates = np.flatnonzero(bucket_ids == bucket)
                if count:
                    selected = rng.choice(candidates, size=count, replace=False)
                    shuffled[selected, symbol] = True
        return shuffled
