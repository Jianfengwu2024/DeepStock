"""Bayesian distributed-lag attribution of sector leaders and responders."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
from numpy.typing import NDArray

from .types import (
    FlowAttributionParameters,
    FlowAttributionPosterior,
    NormalizedSectorPanel,
    SectorFlowPosterior,
    StockFlowAttribution,
)


@dataclass(frozen=True, slots=True)
class AttributionConfig:
    max_lag_minutes: int = 30
    ridge_precision: float = 4.0
    minimum_rows: int = 120
    minimum_residual_variance: float = 1e-6

    def __post_init__(self) -> None:
        if self.max_lag_minutes < 0:
            raise ValueError("maximum attribution lag cannot be negative")
        if self.ridge_precision <= 0.0:
            raise ValueError("ridge precision must be positive")
        if self.minimum_rows < 2:
            raise ValueError("attribution requires at least two rows")
        if self.minimum_residual_variance <= 0.0:
            raise ValueError("minimum residual variance must be positive")


class FlowAttributionModel:
    def __init__(self, config: AttributionConfig | None = None) -> None:
        self.config = config or AttributionConfig()

    def fit(
        self,
        panel: NormalizedSectorPanel,
        flow: SectorFlowPosterior,
        rows: NDArray[np.integer],
    ) -> FlowAttributionParameters:
        self._validate_alignment(panel, flow)
        selected = np.asarray(rows, dtype=np.int64)
        if len(selected) < self.config.minimum_rows:
            raise ValueError("too few selected rows for flow attribution")
        if np.any((selected < 0) | (selected >= len(panel.timestamps))):
            raise ValueError("attribution training rows are outside the panel")
        selected_mask = np.zeros(len(panel.timestamps), dtype=np.bool_)
        selected_mask[selected] = True
        lags = np.arange(self.config.max_lag_minutes + 1, dtype=np.int64)
        leader_inputs: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = []
        response_inputs: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = []
        for symbol in range(len(panel.symbols)):
            leader_inputs.append(
                self._regression_inputs(
                    predictor=panel.velocity[:, symbol],
                    predictor_valid=panel.valid[:, symbol],
                    target=flow.leave_one_out_mean[:, symbol],
                    target_valid=flow.valid,
                    selected=selected_mask,
                    weights=panel.activity[:, symbol],
                    weights_valid=panel.activity_valid[:, symbol],
                )
            )
            response_inputs.append(
                self._regression_inputs(
                    predictor=flow.leave_one_out_mean[:, symbol],
                    predictor_valid=flow.valid,
                    target=panel.velocity[:, symbol],
                    target_valid=panel.valid[:, symbol],
                    selected=selected_mask,
                    weights=panel.activity[:, symbol],
                    weights_valid=panel.activity_valid[:, symbol],
                )
            )

        zero_prior = np.zeros(len(lags), dtype=np.float64)
        initial_leader = [self._fit_ridge(*inputs, zero_prior) for inputs in leader_inputs]
        initial_response = [self._fit_ridge(*inputs, zero_prior) for inputs in response_inputs]
        leader_prior = np.median(
            np.stack([item[0] for item in initial_leader]), axis=0
        )
        response_prior = np.median(
            np.stack([item[0] for item in initial_response]), axis=0
        )
        leader_fits = [self._fit_ridge(*inputs, leader_prior) for inputs in leader_inputs]
        response_fits = [self._fit_ridge(*inputs, response_prior) for inputs in response_inputs]
        return FlowAttributionParameters(
            symbols=panel.symbols,
            lags=lags,
            leader_coefficient_mean=np.stack([item[0] for item in leader_fits]),
            leader_coefficient_covariance=np.stack([item[1] for item in leader_fits]),
            response_coefficient_mean=np.stack([item[0] for item in response_fits]),
            response_coefficient_covariance=np.stack([item[1] for item in response_fits]),
            leader_residual_scale=np.asarray(
                [sqrt(item[2]) for item in leader_fits], dtype=np.float64
            ),
            response_residual_scale=np.asarray(
                [sqrt(item[2]) for item in response_fits], dtype=np.float64
            ),
            leader_prior_mean=leader_prior,
            response_prior_mean=response_prior,
        )

    def summarize(
        self, parameters: FlowAttributionParameters
    ) -> tuple[StockFlowAttribution, ...]:
        summaries: list[StockFlowAttribution] = []
        for symbol, name in enumerate(parameters.symbols):
            leader_std = np.sqrt(
                np.maximum(
                    np.diag(parameters.leader_coefficient_covariance[symbol]),
                    1e-15,
                )
            )
            response_std = np.sqrt(
                np.maximum(
                    np.diag(parameters.response_coefficient_covariance[symbol]),
                    1e-15,
                )
            )
            leader_quality = (
                parameters.leader_coefficient_mean[symbol] / leader_std
            )
            response_quality = (
                parameters.response_coefficient_mean[symbol] / response_std
            )
            leader_index = int(np.argmax(leader_quality))
            response_index = int(np.argmax(response_quality))
            leader_mean = float(
                parameters.leader_coefficient_mean[symbol, leader_index]
            )
            response_mean = float(
                parameters.response_coefficient_mean[symbol, response_index]
            )
            summaries.append(
                StockFlowAttribution(
                    symbol=name,
                    leader_probability=self._normal_cdf(
                        leader_mean / float(leader_std[leader_index])
                    ),
                    leader_horizon_minutes=int(parameters.lags[leader_index]),
                    leader_coefficient_mean=leader_mean,
                    leader_coefficient_std=float(leader_std[leader_index]),
                    response_probability=self._normal_cdf(
                        response_mean / float(response_std[response_index])
                    ),
                    response_delay_minutes=int(parameters.lags[response_index]),
                    response_coefficient_mean=response_mean,
                    response_coefficient_std=float(response_std[response_index]),
                    coupling_mean=response_mean,
                    coupling_std=float(response_std[response_index]),
                    residual_scale=float(parameters.response_residual_scale[symbol]),
                )
            )
        return tuple(summaries)

    def score(
        self,
        panel: NormalizedSectorPanel,
        flow: SectorFlowPosterior,
        parameters: FlowAttributionParameters,
    ) -> FlowAttributionPosterior:
        self._validate_alignment(panel, flow)
        if parameters.symbols != panel.symbols:
            raise ValueError("attribution parameters do not match panel symbols")
        shape = panel.velocity.shape
        leader_mean = np.full(shape, np.nan, dtype=np.float64)
        leader_variance = np.full(shape, np.nan, dtype=np.float64)
        response_mean = np.full(shape, np.nan, dtype=np.float64)
        response_variance = np.full(shape, np.nan, dtype=np.float64)
        valid = np.zeros(shape, dtype=np.bool_)
        lag_count = len(parameters.lags)
        for row in range(lag_count - 1, shape[0]):
            source_rows = row - parameters.lags
            for symbol in range(shape[1]):
                leader_valid = bool(np.all(panel.valid[source_rows, symbol]))
                response_valid = bool(np.all(flow.valid[source_rows]))
                if leader_valid:
                    design = panel.velocity[source_rows, symbol]
                    leader_mean[row, symbol] = float(
                        design @ parameters.leader_coefficient_mean[symbol]
                    )
                    leader_variance[row, symbol] = float(
                        design
                        @ parameters.leader_coefficient_covariance[symbol]
                        @ design
                    )
                if response_valid:
                    design = flow.leave_one_out_mean[source_rows, symbol]
                    response_mean[row, symbol] = float(
                        design @ parameters.response_coefficient_mean[symbol]
                    )
                    response_variance[row, symbol] = float(
                        design
                        @ parameters.response_coefficient_covariance[symbol]
                        @ design
                    )
                valid[row, symbol] = leader_valid and response_valid
        return FlowAttributionPosterior(
            symbols=panel.symbols,
            timestamps=panel.timestamps,
            leader_mean=leader_mean,
            leader_variance=leader_variance,
            response_mean=response_mean,
            response_variance=response_variance,
            valid=valid,
        )

    def _regression_inputs(
        self,
        *,
        predictor: NDArray[np.float64],
        predictor_valid: NDArray[np.bool_],
        target: NDArray[np.float64],
        target_valid: NDArray[np.bool_],
        selected: NDArray[np.bool_],
        weights: NDArray[np.float64],
        weights_valid: NDArray[np.bool_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        lag_count = self.config.max_lag_minutes + 1
        designs: list[NDArray[np.float64]] = []
        targets: list[float] = []
        row_weights: list[float] = []
        for row in range(lag_count - 1, len(target)):
            source_rows = row - np.arange(lag_count)
            if not selected[row] or not np.all(selected[source_rows]):
                continue
            if not target_valid[row] or not np.all(predictor_valid[source_rows]):
                continue
            designs.append(predictor[source_rows])
            targets.append(float(target[row]))
            confidence = float(weights[row]) if weights_valid[row] else 1.0
            row_weights.append(float(np.clip(confidence, 0.1, 10.0)))
        if len(designs) < self.config.minimum_rows:
            raise ValueError("too few valid rows for flow attribution")
        return (
            np.stack(designs).astype(np.float64),
            np.asarray(targets, dtype=np.float64),
            np.asarray(row_weights, dtype=np.float64),
        )

    def _fit_ridge(
        self,
        design: NDArray[np.float64],
        target: NDArray[np.float64],
        weights: NDArray[np.float64],
        prior_mean: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_target = target * np.sqrt(weights)
        precision = (
            weighted_design.T @ weighted_design
            + np.eye(design.shape[1]) * self.config.ridge_precision
        )
        information = (
            weighted_design.T @ weighted_target
            + self.config.ridge_precision * prior_mean
        )
        precision_inverse = np.linalg.inv(precision)
        coefficient = precision_inverse @ information
        residual = target - design @ coefficient
        residual_variance = max(
            float(np.average(np.square(residual), weights=weights)),
            self.config.minimum_residual_variance,
        )
        covariance = precision_inverse * residual_variance
        return coefficient, covariance, residual_variance

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + erf(value / sqrt(2.0)))

    @staticmethod
    def _validate_alignment(
        panel: NormalizedSectorPanel, flow: SectorFlowPosterior
    ) -> None:
        if panel.symbols != flow.symbols:
            raise ValueError("sector panel and flow symbols do not align")
        if not np.array_equal(panel.timestamps, flow.timestamps):
            raise ValueError("sector panel and flow timestamps do not align")
