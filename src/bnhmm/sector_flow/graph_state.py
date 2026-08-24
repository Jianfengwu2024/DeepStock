"""Small Bayesian graph-state model for completed sector river slices."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .river import daily_statistics
from .types import (
    DailyGraphState,
    DailyRiverSlice,
    GraphStateParameters,
    MonthlyRiverTopology,
    SectorGraphPrediction,
)


MODEL_VERSION = "sector-flow-river-v2"


@dataclass(frozen=True, slots=True)
class GraphStateConfig:
    graph_strength: float = 1.0
    state_ridge: float = 0.5
    coefficient_prior_precision: float = 1.0
    minimum_variance: float = 1e-8

    def __post_init__(self) -> None:
        if self.graph_strength < 0.0 or self.state_ridge <= 0.0:
            raise ValueError("graph-state regularization must be nonnegative")
        if self.coefficient_prior_precision <= 0.0 or self.minimum_variance <= 0.0:
            raise ValueError("Bayesian ridge settings must be positive")


class SectorGraphStateModel:
    def __init__(self, config: GraphStateConfig | None = None) -> None:
        self.config = config or GraphStateConfig()

    def infer_state(
        self,
        topology: MonthlyRiverTopology,
        river_slice: DailyRiverSlice,
    ) -> DailyGraphState:
        indices = self._topology_indices(topology, river_slice.symbols)
        statistics = daily_statistics(river_slice, topology)
        observed = statistics[:, 2]
        activity = np.clip(statistics[:, 5], 0.25, 4.0)
        graph = topology.graph[np.ix_(indices, indices)]
        laplacian = np.diag(graph.sum(axis=1)) - graph
        observation_precision = np.diag(activity)
        posterior_precision = (
            observation_precision
            + self.config.graph_strength * laplacian
            + self.config.state_ridge * np.eye(len(indices))
        )
        posterior_covariance = np.linalg.inv(posterior_precision)
        posterior_mean = posterior_covariance @ observation_precision @ observed
        bands = topology.band_ids[indices]
        band_mean = np.zeros(len(indices), dtype=np.float64)
        for row, band in enumerate(bands):
            peers = bands == band if band >= 0 else np.zeros(len(indices), dtype=np.bool_)
            if peers.any():
                band_mean[row] = float(np.mean(posterior_mean[peers]))
        return DailyGraphState(
            session=river_slice.session,
            symbols=river_slice.symbols,
            sector_mean=float(np.mean(statistics[:, 0])),
            band_mean=band_mean,
            stock_mean=posterior_mean,
            stock_variance=np.maximum(
                np.diag(posterior_covariance), self.config.minimum_variance
            ),
        )

    def fit(
        self,
        samples: Sequence[
            tuple[MonthlyRiverTopology, DailyRiverSlice, NDArray[np.float64]]
        ],
        *,
        training_cutoff: np.datetime64,
    ) -> GraphStateParameters:
        if not samples:
            raise ValueError("graph-state fitting requires training samples")
        feature_rows: list[NDArray[np.float64]] = []
        targets: list[NDArray[np.float64]] = []
        for topology, river_slice, target in samples:
            features, _ = self._features(topology, river_slice)
            if target.shape != (len(river_slice.symbols),):
                raise ValueError("graph-state targets must match slice symbols")
            indices = self._topology_indices(topology, river_slice.symbols)
            valid = topology.assigned[indices] & np.isfinite(target)
            feature_rows.append(features[valid])
            targets.append(np.asarray(target[valid], dtype=np.float64))
        matrix = np.vstack(feature_rows)
        response = np.concatenate(targets)
        if len(response) < 2:
            raise ValueError("graph-state fitting requires at least two targets")
        center = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale[scale < 1e-8] = 1.0
        standardized = (matrix - center) / scale
        design = np.column_stack((np.ones(len(standardized)), standardized))
        identity = np.eye(design.shape[1])
        identity[0, 0] = 1e-8
        initial_precision = (
            design.T @ design
            + self.config.coefficient_prior_precision * identity
        )
        initial_mean = np.linalg.solve(initial_precision, design.T @ response)
        residual = response - design @ initial_mean
        residual_variance = max(
            float(np.mean(np.square(residual))), self.config.minimum_variance
        )
        precision = (
            design.T @ design / residual_variance
            + self.config.coefficient_prior_precision * identity
        )
        covariance = np.linalg.inv(precision)
        coefficient_mean = covariance @ design.T @ response / residual_variance
        return GraphStateParameters(
            feature_center=center,
            feature_scale=scale,
            coefficient_mean=coefficient_mean,
            coefficient_covariance=covariance,
            residual_variance=residual_variance,
            graph_strength=self.config.graph_strength,
            training_cutoff=np.datetime64(training_cutoff, "ns"),
            model_version=MODEL_VERSION,
        )

    def predict(
        self,
        parameters: GraphStateParameters,
        topology: MonthlyRiverTopology,
        river_slice: DailyRiverSlice,
        *,
        available_at: np.datetime64,
    ) -> SectorGraphPrediction:
        signal_time = np.datetime64(river_slice.timestamps[-1], "ns")
        if parameters.training_cutoff >= signal_time:
            raise ValueError("graph-state parameters are not sample-out")
        features, state = self._features(topology, river_slice)
        standardized = (features - parameters.feature_center) / parameters.feature_scale
        design = np.column_stack((np.ones(len(standardized)), standardized))
        mean = design @ parameters.coefficient_mean
        mean -= float(np.mean(mean))
        parameter_variance = np.einsum(
            "ij,jk,ik->i",
            design,
            parameters.coefficient_covariance,
            design,
        )
        graph_coefficient = parameters.coefficient_mean[7]
        variance = (
            parameters.residual_variance
            + parameter_variance
            + np.square(graph_coefficient) * state.stock_variance
        )
        return SectorGraphPrediction(
            topology_month=topology.month,
            symbols=river_slice.symbols,
            expected_relative_return=mean,
            predictive_variance=np.maximum(variance, self.config.minimum_variance),
            signal_time=signal_time,
            available_at=np.datetime64(available_at, "ns"),
            training_cutoff=parameters.training_cutoff,
            model_version=parameters.model_version,
        )

    def _features(
        self,
        topology: MonthlyRiverTopology,
        river_slice: DailyRiverSlice,
    ) -> tuple[NDArray[np.float64], DailyGraphState]:
        statistics = daily_statistics(river_slice, topology)
        state = self.infer_state(topology, river_slice)
        indices = self._topology_indices(topology, river_slice.symbols)
        features = np.column_stack(
            (
                statistics,
                state.stock_mean,
                np.sqrt(state.stock_variance),
                topology.cohesion[indices],
                topology.centrality[indices],
            )
        )
        return features, state

    @staticmethod
    def _topology_indices(
        topology: MonthlyRiverTopology, symbols: tuple[str, ...]
    ) -> NDArray[np.int64]:
        index = {symbol: position for position, symbol in enumerate(topology.symbols)}
        if set(index) != set(symbols):
            raise ValueError("graph topology and river slice symbols do not align")
        return np.asarray([index[symbol] for symbol in symbols], dtype=np.int64)
