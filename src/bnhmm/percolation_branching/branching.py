"""Sparse Gamma-Poisson branching inference on a frozen monthly support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .types import ActivationPanel, BranchingPosterior, Direction, MonthlyBackboneSupport


def spectral_radius(matrix: np.ndarray) -> float:
    values = np.linalg.eigvals(np.asarray(matrix, dtype=np.float64))
    return float(np.max(np.abs(values))) if len(values) else 0.0


@dataclass(frozen=True, slots=True)
class BayesianBranchingModel:
    maximum_lag: int = 3
    iterations: int = 25
    edge_prior_shape: float = 0.2
    edge_prior_rate: float = 2.0
    rate_prior_shape: float = 0.5
    rate_prior_rate: float = 5.0

    def __post_init__(self) -> None:
        if self.maximum_lag < 1 or self.iterations < 1:
            raise ValueError("branching lag and iterations must be positive")

    def fit(
        self,
        events: ActivationPanel,
        support: MonthlyBackboneSupport,
        training_sessions: Iterable[np.datetime64],
        direction: Direction,
    ) -> BranchingPosterior:
        if events.symbols != support.symbols:
            raise ValueError("branching support symbols do not align with events")
        sessions = np.unique(
            np.asarray(tuple(training_sessions), dtype="datetime64[D]")
        )
        if len(sessions) == 0 or sessions[-1] > support.training_cutoff:
            raise ValueError("branching training sessions exceed support cutoff")
        selected = np.isin(events.timestamps.astype("datetime64[D]"), sessions)
        rows = np.flatnonzero(selected)
        matrix = events.direction(direction)
        count = len(events.symbols)
        lag_weights = np.full(self.maximum_lag, 1.0 / self.maximum_lag)
        edge_exposure = np.zeros((count, count), dtype=np.float64)
        self_exposure = np.zeros(count, dtype=np.float64)
        minute_ids = events.timestamps.astype("datetime64[m]").astype(np.int64)
        row_by_minute = {int(minute_ids[row]): int(row) for row in rows}

        for parent_row in rows:
            parent_session = events.timestamps[parent_row].astype("datetime64[D]")
            for lag, weight in enumerate(lag_weights, start=1):
                child_row = row_by_minute.get(int(minute_ids[parent_row] + lag))
                if child_row is None or events.timestamps[child_row].astype("datetime64[D]") != parent_session:
                    continue
                for parent in np.flatnonzero(matrix[parent_row]):
                    mark = float(events.marks[parent_row, parent])
                    edge_exposure[parent, support.adjacency[parent]] += mark * weight
                    self_exposure[parent] += mark * weight

        edge_shape = np.where(support.adjacency, self.edge_prior_shape, 0.0)
        edge_rate = np.where(
            support.adjacency, self.edge_prior_rate + edge_exposure, np.inf
        )
        external_shape = np.full(count, self.rate_prior_shape)
        external_rate_denominator = np.full(count, self.rate_prior_rate + len(rows))
        self_shape = np.full(count, self.rate_prior_shape)
        self_rate_denominator = self.rate_prior_rate + self_exposure
        mean = np.divide(edge_shape, edge_rate, out=np.zeros_like(edge_shape), where=np.isfinite(edge_rate))
        external = external_shape / external_rate_denominator
        own_rate = self_shape / self_rate_denominator

        for _ in range(self.iterations):
            edge_responsibility = np.zeros_like(mean)
            external_responsibility = np.zeros(count)
            self_responsibility = np.zeros(count)
            lag_responsibility = np.ones(self.maximum_lag, dtype=np.float64)
            for child_row in rows:
                active_children = np.flatnonzero(matrix[child_row])
                common = len(active_children) >= max(2, int(np.ceil(count * 0.3)))
                for child in active_children:
                    if common:
                        external_responsibility[child] += 1.0
                        continue
                    candidates: list[tuple[int, int, float]] = []
                    self_terms: list[tuple[int, float]] = []
                    child_session = events.timestamps[child_row].astype("datetime64[D]")
                    for lag, lag_weight in enumerate(lag_weights, start=1):
                        parent_row = row_by_minute.get(int(minute_ids[child_row] - lag))
                        if parent_row is None or events.timestamps[parent_row].astype("datetime64[D]") != child_session:
                            continue
                        for parent in np.flatnonzero(matrix[parent_row]):
                            mark = float(events.marks[parent_row, parent])
                            if parent == child:
                                self_terms.append((lag - 1, own_rate[child] * mark * lag_weight))
                            elif support.adjacency[parent, child]:
                                candidates.append((parent, lag - 1, mean[parent, child] * mark * lag_weight))
                    total = external[child]
                    total += sum(value for _, value in self_terms)
                    total += sum(value for _, _, value in candidates)
                    if total <= 0.0:
                        external_responsibility[child] += 1.0
                        continue
                    external_responsibility[child] += external[child] / total
                    for lag, value in self_terms:
                        responsibility = value / total
                        self_responsibility[child] += responsibility
                        lag_responsibility[lag] += responsibility
                    for parent, lag, value in candidates:
                        responsibility = value / total
                        edge_responsibility[parent, child] += responsibility
                        lag_responsibility[lag] += responsibility
            edge_shape = np.where(
                support.adjacency,
                self.edge_prior_shape + edge_responsibility,
                0.0,
            )
            mean = np.divide(
                edge_shape,
                edge_rate,
                out=np.zeros_like(edge_shape),
                where=np.isfinite(edge_rate),
            )
            external_shape = self.rate_prior_shape + external_responsibility
            self_shape = self.rate_prior_shape + self_responsibility
            external = external_shape / external_rate_denominator
            own_rate = self_shape / self_rate_denominator
            lag_weights = lag_responsibility / float(lag_responsibility.sum())

        variance = np.divide(
            edge_shape,
            np.square(edge_rate),
            out=np.zeros_like(edge_shape),
            where=np.isfinite(edge_rate),
        )
        radius = spectral_radius(mean)
        normalized = mean / radius if radius > 1e-12 else mean.copy()
        return BranchingPosterior(
            direction=direction,
            symbols=events.symbols,
            adjacency=support.adjacency.copy(),
            mean=mean,
            variance=variance,
            normalized_mean=normalized,
            lag_weights=lag_weights,
            external_rate=external,
            self_rate=own_rate,
            training_cutoff=sessions[-1],
        )
