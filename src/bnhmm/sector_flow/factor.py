"""Robust causal dynamic-factor inference for synchronized sector velocity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .types import NormalizedSectorPanel, SectorFactorParameters, SectorFlowPosterior


@dataclass(frozen=True, slots=True)
class SectorFactorConfig:
    persistence: float | None = None
    process_variance: float | None = None
    observation_variance_floor: float = 0.05
    robust_degrees_of_freedom: float = 5.0
    minimum_symbols: int = 5

    def __post_init__(self) -> None:
        if self.persistence is not None and not -0.999 <= self.persistence <= 0.999:
            raise ValueError("configured persistence is outside the stable range")
        if self.process_variance is not None and self.process_variance <= 0.0:
            raise ValueError("configured process variance must be positive")
        if self.observation_variance_floor <= 0.0:
            raise ValueError("observation variance floor must be positive")
        if self.robust_degrees_of_freedom <= 2.0:
            raise ValueError("robust degrees of freedom must exceed two")
        if self.minimum_symbols < 2:
            raise ValueError("at least two symbols are required for a sector factor")


class BayesianSectorFactor:
    def __init__(self, config: SectorFactorConfig | None = None) -> None:
        self.config = config or SectorFactorConfig()

    def fit(
        self, panel: NormalizedSectorPanel, rows: NDArray[np.integer]
    ) -> SectorFactorParameters:
        selected = np.asarray(rows, dtype=np.int64)
        if not len(selected):
            raise ValueError("factor fitting requires training rows")
        values = panel.velocity[selected]
        valid = panel.valid[selected]
        if values.shape[1] < self.config.minimum_symbols:
            raise ValueError("sector contains too few symbols for factor fitting")
        matrix = np.where(valid, values, 0.0)
        coverage = np.maximum(valid.sum(axis=0), 1)
        symbol_scale = np.sqrt(np.sum(np.square(matrix), axis=0) / coverage)
        symbol_scale = np.maximum(symbol_scale, np.sqrt(self.config.observation_variance_floor))
        standardized = matrix / symbol_scale
        left, singular, right = np.linalg.svd(standardized, full_matrices=False)
        factor = left[:, 0] * singular[0]
        factor_scale = max(float(factor.std()), 1e-8)
        factor /= factor_scale
        loadings = (right[0] * symbol_scale * factor_scale).astype(np.float64)
        observed_mean = np.divide(
            matrix.sum(axis=1),
            np.maximum(valid.sum(axis=1), 1),
        )
        if float(np.dot(factor, observed_mean)) < 0.0:
            factor *= -1.0
            loadings *= -1.0

        denominator = float(np.dot(factor[:-1], factor[:-1])) if len(factor) > 1 else 0.0
        fitted_persistence = (
            float(np.dot(factor[1:], factor[:-1]) / denominator)
            if denominator > 1e-12
            else 0.0
        )
        persistence = (
            self.config.persistence
            if self.config.persistence is not None
            else float(np.clip(fitted_persistence, -0.99, 0.99))
        )
        innovations = factor[1:] - persistence * factor[:-1]
        fitted_process_variance = float(np.var(innovations)) if len(innovations) else 1.0
        process_variance = max(
            self.config.process_variance or fitted_process_variance,
            1e-6,
        )
        fitted = factor[:, None] * loadings[None, :]
        residual = np.where(valid, values - fitted, np.nan)
        residual_variance = np.asarray(
            [
                np.nanmean(np.square(residual[:, column]))
                if np.any(valid[:, column])
                else 1.0
                for column in range(values.shape[1])
            ],
            dtype=np.float64,
        )
        residual_variance = np.maximum(
            np.nan_to_num(residual_variance, nan=1.0),
            self.config.observation_variance_floor,
        )
        return SectorFactorParameters(
            loadings=loadings,
            residual_variance=residual_variance,
            persistence=persistence,
            process_variance=process_variance,
        )

    def filter(
        self, panel: NormalizedSectorPanel, parameters: SectorFactorParameters
    ) -> SectorFlowPosterior:
        rows, symbols = panel.velocity.shape
        if parameters.loadings.shape != (symbols,):
            raise ValueError("factor parameters do not match panel symbols")
        factor_mean = np.zeros(rows, dtype=np.float64)
        factor_variance = np.zeros(rows, dtype=np.float64)
        loo_mean = np.zeros((rows, symbols), dtype=np.float64)
        loo_variance = np.zeros((rows, symbols), dtype=np.float64)
        activity = np.zeros(rows, dtype=np.float64)
        breadth = np.zeros(rows, dtype=np.float64)
        coherence = np.zeros(rows, dtype=np.float64)
        dispersion = np.zeros(rows, dtype=np.float64)
        flow_strength = np.zeros(rows, dtype=np.float64)
        flow_valid = np.zeros(rows, dtype=np.bool_)
        unconditional = parameters.process_variance / max(
            1.0 - parameters.persistence**2, 1e-3
        )
        previous_mean = 0.0
        previous_variance = unconditional
        previous_loo_mean = np.zeros(symbols, dtype=np.float64)
        previous_loo_variance = np.full(symbols, unconditional, dtype=np.float64)

        for row in range(rows):
            observed = panel.valid[row]
            y = np.where(observed, panel.velocity[row], 0.0)
            prior_mean = parameters.persistence * previous_mean
            prior_variance = (
                parameters.persistence**2 * previous_variance
                + parameters.process_variance
            )
            prior_loo_mean = parameters.persistence * previous_loo_mean
            prior_loo_variance = (
                parameters.persistence**2 * previous_loo_variance
                + parameters.process_variance
            )
            standardized_square = np.square(y) / parameters.residual_variance
            robust_weight = (
                self.config.robust_degrees_of_freedom + 1.0
            ) / (
                self.config.robust_degrees_of_freedom + standardized_square
            )
            robust_weight *= observed
            precision_contribution = (
                robust_weight
                * np.square(parameters.loadings)
                / parameters.residual_variance
            )
            information_contribution = (
                robust_weight
                * parameters.loadings
                * y
                / parameters.residual_variance
            )
            total_precision = float(precision_contribution.sum())
            total_information = float(information_contribution.sum())
            enough = int(observed.sum()) >= self.config.minimum_symbols
            if enough and total_precision > 0.0:
                posterior_precision = 1.0 / prior_variance + total_precision
                current_variance = 1.0 / posterior_precision
                current_mean = current_variance * (
                    prior_mean / prior_variance + total_information
                )
                flow_valid[row] = True
            else:
                current_mean = prior_mean
                current_variance = prior_variance
            factor_mean[row] = current_mean
            factor_variance[row] = current_variance

            loo_precision = (
                1.0 / prior_loo_variance
                + total_precision
                - precision_contribution
            )
            loo_variance[row] = 1.0 / np.maximum(loo_precision, 1e-12)
            loo_mean[row] = loo_variance[row] * (
                prior_loo_mean / prior_loo_variance
                + total_information
                - information_contribution
            )

            active = panel.activity_valid[row]
            if np.any(active):
                activity[row] = float(np.mean(panel.activity[row, active]))
            if enough:
                observed_values = y[observed]
                predicted = parameters.loadings[observed] * current_mean
                residual = observed_values - predicted
                direction = np.sign(current_mean)
                breadth[row] = float(
                    np.mean(np.sign(observed_values) == direction)
                ) if direction else 0.5
                observed_energy = float(np.mean(np.square(observed_values)))
                residual_energy = float(np.mean(np.square(residual)))
                coherence[row] = float(
                    np.clip(
                        1.0 - residual_energy / max(observed_energy, 1e-12),
                        0.0,
                        1.0,
                    )
                )
                dispersion[row] = float(np.std(residual))
                flow_strength[row] = current_mean * activity[row] * coherence[row]

            previous_mean = current_mean
            previous_variance = current_variance
            previous_loo_mean = loo_mean[row]
            previous_loo_variance = loo_variance[row]

        return SectorFlowPosterior(
            symbols=panel.symbols,
            timestamps=panel.timestamps,
            factor_mean=factor_mean,
            factor_variance=factor_variance,
            leave_one_out_mean=loo_mean,
            leave_one_out_variance=loo_variance,
            activity=activity,
            breadth=breadth,
            coherence=coherence,
            dispersion=dispersion,
            flow_strength=flow_strength,
            valid=flow_valid,
        )
