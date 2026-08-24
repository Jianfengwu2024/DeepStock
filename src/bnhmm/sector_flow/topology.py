"""Causal monthly synchronization topology for sector river slices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import (
    fcluster,
    leaves_list,
    linkage,
    optimal_leaf_ordering,
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from .factor import BayesianSectorFactor
from .types import MonthlyRiverTopology, NormalizedSectorPanel


@dataclass(frozen=True, slots=True)
class TopologyConfig:
    lookback_sessions: int = 60
    minimum_sessions: int = 40
    minimum_pair_sessions: int = 30
    minimum_minutes_per_session: int = 30
    activity_cap: float = 3.0
    shrinkage_sessions: float = 20.0
    maximum_bands: int = 8
    minimum_band_size: int = 2

    def __post_init__(self) -> None:
        if self.lookback_sessions < 2:
            raise ValueError("topology lookback must contain at least two sessions")
        if not 2 <= self.minimum_sessions <= self.lookback_sessions:
            raise ValueError("minimum topology sessions must fit the lookback")
        if not 2 <= self.minimum_pair_sessions <= self.lookback_sessions:
            raise ValueError("minimum pair sessions must fit the lookback")
        if self.minimum_minutes_per_session < 2:
            raise ValueError("minimum pair minutes must exceed one")
        if self.activity_cap <= 1.0 or self.shrinkage_sessions <= 0.0:
            raise ValueError("topology reliability settings are invalid")
        if self.maximum_bands < 2 or self.minimum_band_size < 2:
            raise ValueError("topology band settings are invalid")


class RiverTopologyBuilder:
    def __init__(self, config: TopologyConfig | None = None) -> None:
        self.config = config or TopologyConfig()

    def build(
        self,
        panel: NormalizedSectorPanel,
        cutoff: np.datetime64,
        month: np.datetime64,
        previous: MonthlyRiverTopology | None = None,
    ) -> MonthlyRiverTopology:
        canonical = self._canonical_panel(panel)
        rows = self._window_rows(canonical.timestamps, cutoff)
        sessions = canonical.timestamps[rows].astype("datetime64[D]")
        assigned = self._assigned_symbols(canonical, rows, sessions)
        count = len(canonical.symbols)
        common = np.eye(count, dtype=np.float64)
        residual = np.eye(count, dtype=np.float64)

        columns = np.flatnonzero(assigned)
        if len(columns) >= 2:
            window = self._take(canonical, rows, columns)
            common_small, residual_small = self._similarities(window)
            common[np.ix_(columns, columns)] = common_small
            residual[np.ix_(columns, columns)] = residual_small

        fused = np.clip((common + residual) * 0.5, -1.0, 1.0)
        graph = np.maximum(fused, 0.0)
        graph[~assigned, :] = 0.0
        graph[:, ~assigned] = 0.0
        np.fill_diagonal(graph, 0.0)
        band_ids = self._bands(fused, assigned, canonical.symbols, previous)
        row_order = self._row_order(
            fused, band_ids, canonical.symbols, previous
        )
        cohesion = self._cohesion(fused, band_ids)
        denominator = max(int(assigned.sum()) - 1, 1)
        centrality = graph.sum(axis=1) / denominator

        return MonthlyRiverTopology(
            month=np.datetime64(month, "M"),
            training_cutoff=np.datetime64(cutoff, "ns"),
            symbols=canonical.symbols,
            assigned=assigned,
            band_ids=band_ids,
            row_order=row_order,
            common_similarity=common,
            residual_similarity=residual,
            fused_similarity=fused,
            graph=graph,
            cohesion=cohesion,
            centrality=centrality,
        )

    def _window_rows(
        self, timestamps: NDArray[np.datetime64], cutoff: np.datetime64
    ) -> NDArray[np.int64]:
        eligible = timestamps <= np.datetime64(cutoff, "ns")
        sessions = np.unique(timestamps[eligible].astype("datetime64[D]"))
        if len(sessions) < self.config.minimum_sessions:
            raise ValueError("insufficient completed sessions for river topology")
        selected_sessions = sessions[-self.config.lookback_sessions :]
        selected = eligible & np.isin(
            timestamps.astype("datetime64[D]"), selected_sessions
        )
        return np.flatnonzero(selected).astype(np.int64)

    def _assigned_symbols(
        self,
        panel: NormalizedSectorPanel,
        rows: NDArray[np.int64],
        sessions: NDArray[np.datetime64],
    ) -> NDArray[np.bool_]:
        assigned = np.zeros(len(panel.symbols), dtype=np.bool_)
        for column in range(len(panel.symbols)):
            complete = 0
            for session in np.unique(sessions):
                local = sessions == session
                if int(panel.valid[rows[local], column].sum()) >= self.config.minimum_minutes_per_session:
                    complete += 1
            assigned[column] = complete >= self.config.minimum_sessions
        return assigned

    def _similarities(
        self, panel: NormalizedSectorPanel
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        rows = np.arange(len(panel.timestamps), dtype=np.int64)
        factor_model = BayesianSectorFactor()
        parameters = factor_model.fit(panel, rows)
        flow = factor_model.filter(panel, parameters)
        residual_values = (
            panel.velocity
            - flow.factor_mean[:, None] * parameters.loadings[None, :]
        )
        common = self._pairwise(panel.velocity, panel)
        residual = self._pairwise(residual_values, panel)
        return common, residual

    def _pairwise(
        self, values: NDArray[np.float64], panel: NormalizedSectorPanel
    ) -> NDArray[np.float64]:
        symbols = values.shape[1]
        result = np.eye(symbols, dtype=np.float64)
        sessions = panel.timestamps.astype("datetime64[D]")
        reliability = np.zeros_like(panel.activity)
        for column in range(symbols):
            valid = panel.activity_valid[:, column] & np.isfinite(
                panel.activity[:, column]
            )
            median = float(np.median(panel.activity[valid, column])) if valid.any() else 1.0
            reliability[valid, column] = np.clip(
                panel.activity[valid, column] / max(median, 1e-12),
                0.0,
                self.config.activity_cap,
            )

        for first in range(symbols):
            for second in range(first + 1, symbols):
                valid = (
                    panel.valid[:, first]
                    & panel.valid[:, second]
                    & panel.activity_valid[:, first]
                    & panel.activity_valid[:, second]
                    & np.isfinite(values[:, first])
                    & np.isfinite(values[:, second])
                )
                effective_sessions = sum(
                    int(np.sum(valid & (sessions == session)))
                    >= self.config.minimum_minutes_per_session
                    for session in np.unique(sessions)
                )
                if effective_sessions < self.config.minimum_pair_sessions:
                    correlation = 0.0
                else:
                    weights = np.sqrt(
                        reliability[:, first] * reliability[:, second]
                    )
                    correlation = self._weighted_spearman(
                        values[:, first], values[:, second], weights, valid
                    )
                    shrinkage = effective_sessions / (
                        effective_sessions + self.config.shrinkage_sessions
                    )
                    correlation *= shrinkage
                result[first, second] = correlation
                result[second, first] = correlation
        return result

    @staticmethod
    def _weighted_spearman(
        first: NDArray[np.float64],
        second: NDArray[np.float64],
        weights: NDArray[np.float64],
        valid: NDArray[np.bool_],
    ) -> float:
        indices = np.flatnonzero(valid & (weights > 0.0))
        if len(indices) < 3:
            return 0.0
        x = rankdata(first[indices])
        y = rankdata(second[indices])
        weight = weights[indices]
        weight /= max(float(weight.sum()), 1e-12)
        x -= float(np.sum(weight * x))
        y -= float(np.sum(weight * y))
        denominator = np.sqrt(
            float(np.sum(weight * np.square(x)))
            * float(np.sum(weight * np.square(y)))
        )
        if denominator <= 1e-12:
            return 0.0
        return float(np.clip(np.sum(weight * x * y) / denominator, -1.0, 1.0))

    def _bands(
        self,
        similarity: NDArray[np.float64],
        assigned: NDArray[np.bool_],
        symbols: tuple[str, ...],
        previous: MonthlyRiverTopology | None,
    ) -> NDArray[np.int64]:
        result = np.full(len(symbols), -1, dtype=np.int64)
        columns = np.flatnonzero(assigned)
        if not len(columns):
            return result
        if len(columns) < 2 * self.config.minimum_band_size:
            result[columns] = 0
            return result
        distance = np.clip(
            1.0 - similarity[np.ix_(columns, columns)], 0.0, 2.0
        )
        np.fill_diagonal(distance, 0.0)
        condensed = squareform(distance, checks=False)
        tree = linkage(condensed, method="average")
        maximum = min(
            self.config.maximum_bands,
            len(columns) // self.config.minimum_band_size,
        )
        candidates: list[tuple[float, int, NDArray[np.int64]]] = []
        for count in range(2, maximum + 1):
            labels = fcluster(tree, count, criterion="maxclust").astype(np.int64)
            if len(np.unique(labels)) < 2:
                continue
            score = self._silhouette(distance, labels)
            candidates.append((score, -count, labels))
        labels = max(candidates, key=lambda item: (item[0], item[1]))[2]
        labels = self._canonical_labels(labels, columns, symbols)
        if previous is not None:
            labels = self._align_previous(labels, columns, symbols, previous)
        result[columns] = labels
        return result

    @staticmethod
    def _silhouette(
        distance: NDArray[np.float64], labels: NDArray[np.int64]
    ) -> float:
        scores: list[float] = []
        for row, label in enumerate(labels):
            same = np.flatnonzero(labels == label)
            same = same[same != row]
            if not len(same):
                scores.append(0.0)
                continue
            within = float(np.mean(distance[row, same]))
            outside = [
                float(np.mean(distance[row, labels == other]))
                for other in np.unique(labels)
                if other != label
            ]
            nearest = min(outside)
            scores.append((nearest - within) / max(nearest, within, 1e-12))
        return float(np.mean(scores))

    @staticmethod
    def _canonical_labels(
        labels: NDArray[np.int64],
        columns: NDArray[np.int64],
        symbols: tuple[str, ...],
    ) -> NDArray[np.int64]:
        ordered = sorted(
            np.unique(labels),
            key=lambda label: min(
                symbols[columns[index]]
                for index in np.flatnonzero(labels == label)
            ),
        )
        mapping = {int(label): index for index, label in enumerate(ordered)}
        return np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)

    @staticmethod
    def _align_previous(
        labels: NDArray[np.int64],
        columns: NDArray[np.int64],
        symbols: tuple[str, ...],
        previous: MonthlyRiverTopology,
    ) -> NDArray[np.int64]:
        previous_index = {symbol: index for index, symbol in enumerate(previous.symbols)}
        current_labels = np.unique(labels)
        previous_labels = np.unique(previous.band_ids[previous.band_ids >= 0])
        if not len(previous_labels):
            return labels
        overlap = np.zeros((len(current_labels), len(previous_labels)), dtype=np.float64)
        for row, label in enumerate(current_labels):
            members = [symbols[columns[index]] for index in np.flatnonzero(labels == label)]
            for column, old in enumerate(previous_labels):
                overlap[row, column] = sum(
                    symbol in previous_index
                    and previous.band_ids[previous_index[symbol]] == old
                    for symbol in members
                )
        rows, matches = linear_sum_assignment(-overlap)
        mapping = {
            int(current_labels[row]): int(previous_labels[match])
            for row, match in zip(rows, matches, strict=True)
            if overlap[row, match] > 0
        }
        next_label = int(previous_labels.max()) + 1
        for label in current_labels:
            if int(label) not in mapping:
                mapping[int(label)] = next_label
                next_label += 1
        return np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)

    def _row_order(
        self,
        similarity: NDArray[np.float64],
        band_ids: NDArray[np.int64],
        symbols: tuple[str, ...],
        previous: MonthlyRiverTopology | None,
    ) -> NDArray[np.int64]:
        result: list[int] = []
        for band in sorted(np.unique(band_ids[band_ids >= 0])):
            members = np.flatnonzero(band_ids == band)
            if len(members) > 2:
                distance = np.clip(
                    1.0 - similarity[np.ix_(members, members)], 0.0, 2.0
                )
                np.fill_diagonal(distance, 0.0)
                condensed = squareform(distance, checks=False)
                tree = optimal_leaf_ordering(
                    linkage(condensed, method="average"), condensed
                )
                ordered = members[leaves_list(tree)]
            else:
                ordered = members[np.argsort([symbols[index] for index in members])]
            ordered = self._orient(ordered, symbols, previous)
            result.extend(int(item) for item in ordered)
        result.extend(
            sorted(
                (int(item) for item in np.flatnonzero(band_ids < 0)),
                key=lambda item: symbols[item],
            )
        )
        return np.asarray(result, dtype=np.int64)

    @staticmethod
    def _orient(
        ordered: NDArray[np.int64],
        symbols: tuple[str, ...],
        previous: MonthlyRiverTopology | None,
    ) -> NDArray[np.int64]:
        if previous is None:
            forward = tuple(symbols[index] for index in ordered)
            reverse = tuple(reversed(forward))
            return ordered if forward <= reverse else ordered[::-1]
        old_positions = {
            previous.symbols[index]: position
            for position, index in enumerate(previous.row_order)
        }

        def displacement(items: NDArray[np.int64]) -> float:
            positions = [
                old_positions[symbols[index]]
                for index in items
                if symbols[index] in old_positions
            ]
            return float(sum(abs(value - rank) for rank, value in enumerate(positions)))

        return ordered if displacement(ordered) <= displacement(ordered[::-1]) else ordered[::-1]

    @staticmethod
    def _cohesion(
        similarity: NDArray[np.float64], band_ids: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        result = np.zeros(len(band_ids), dtype=np.float64)
        for index, band in enumerate(band_ids):
            peers = np.flatnonzero(band_ids == band) if band >= 0 else np.empty(0, dtype=np.int64)
            peers = peers[peers != index]
            if len(peers):
                result[index] = float(np.mean(similarity[index, peers]))
        return result

    @staticmethod
    def _canonical_panel(panel: NormalizedSectorPanel) -> NormalizedSectorPanel:
        columns = np.argsort(np.asarray(panel.symbols))
        return RiverTopologyBuilder._take(
            panel,
            np.arange(len(panel.timestamps), dtype=np.int64),
            columns,
        )

    @staticmethod
    def _take(
        panel: NormalizedSectorPanel,
        rows: NDArray[np.integer],
        columns: NDArray[np.integer],
    ) -> NormalizedSectorPanel:
        row_index = np.asarray(rows, dtype=np.int64)
        column_index = np.asarray(columns, dtype=np.int64)
        selected = np.ix_(row_index, column_index)
        return NormalizedSectorPanel(
            symbols=tuple(panel.symbols[index] for index in column_index),
            timestamps=panel.timestamps[row_index],
            raw_velocity=panel.raw_velocity[selected],
            velocity=panel.velocity[selected],
            activity=panel.activity[selected],
            velocity_scale=panel.velocity_scale[selected],
            volume_baseline=panel.volume_baseline[selected],
            valid=panel.valid[selected],
            activity_valid=panel.activity_valid[selected],
        )
