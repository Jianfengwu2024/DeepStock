"""Low-parameter Bayesian dynamics over named daily boundary states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import BoundaryDailyStates


_DRIFT_BASE = (
    "velocity_last60",
    "momentum_last60",
    "coherence_last60",
    "decoupling_last60",
    "thickness_last60",
    "thickness_slope_close",
)
_DIFFUSION = (
    "turbulence_last60",
    "thickness_last60",
    "abs_decoupling_last60",
    "low_coherence_last60",
    "velocity_quadratic_variation",
    "historical_index_volatility",
)
_POTENTIAL = (
    "thickness_last60",
    "thickness_slope_close",
    "turbulence_last60",
    "abs_decoupling_last60",
    "coherence_last60",
    "viscosity_close",
)


@dataclass(frozen=True, slots=True)
class BoundaryStateLabels:
    signal_sessions: np.ndarray
    target_time: np.ndarray
    next_return: np.ndarray
    historical_volatility: np.ndarray
    survival: np.ndarray
    return_valid: np.ndarray
    survival_valid: np.ndarray

    def __post_init__(self) -> None:
        shape = (len(self.signal_sessions),)
        arrays = (
            self.target_time,
            self.next_return,
            self.historical_volatility,
            self.survival,
            self.return_valid,
            self.survival_valid,
        )
        if any(item.shape != shape for item in arrays):
            raise ValueError("boundary labels must align with signal sessions")
        if np.any(self.return_valid & ~np.isfinite(self.next_return)):
            raise ValueError("valid next returns must be finite")
        if np.any(self.return_valid & (~np.isfinite(self.historical_volatility) | (self.historical_volatility <= 0.0))):
            raise ValueError("valid historical volatility must be finite and positive")
        if np.any(self.survival_valid & ~np.isin(self.survival, (0.0, 1.0))):
            raise ValueError("valid survival labels must be binary")


@dataclass(frozen=True, slots=True)
class BayesianBoundaryPosterior:
    drift_feature_names: tuple[str, ...]
    drift_center: np.ndarray
    drift_scale: np.ndarray
    drift_mean: np.ndarray
    drift_covariance: np.ndarray
    drift_noise_variance: float
    diffusion_feature_names: tuple[str, ...]
    diffusion_center: np.ndarray
    diffusion_scale: np.ndarray
    diffusion_mean: np.ndarray
    diffusion_covariance: np.ndarray
    diffusion_noise_variance: float
    potential_feature_names: tuple[str, ...]
    potential_center: np.ndarray
    potential_scale: np.ndarray
    potential_mean: np.ndarray
    potential_covariance: np.ndarray
    training_cutoff: np.datetime64
    fit_rows: int
    survival_fit_rows: int

    @property
    def diffusion_variance(self) -> float:
        return self.diffusion_noise_variance

    def design_vector(
        self,
        component: str,
        states: BoundaryDailyStates,
        row: int,
        labels: BoundaryStateLabels,
    ) -> np.ndarray:
        if component == "drift":
            names, center, scale = (
                self.drift_feature_names,
                self.drift_center,
                self.drift_scale,
            )
        elif component == "diffusion":
            names, center, scale = (
                self.diffusion_feature_names,
                self.diffusion_center,
                self.diffusion_scale,
            )
        elif component == "potential":
            names, center, scale = (
                self.potential_feature_names,
                self.potential_center,
                self.potential_scale,
            )
        else:
            raise ValueError(f"unknown boundary component: {component}")
        raw = _feature_matrix(states, labels, names)[row]
        return np.concatenate((np.ones(1), (raw - center) / scale))


@dataclass(frozen=True, slots=True)
class BayesianBoundaryStateConfig:
    prior_precision: float = 2.0
    variance_floor: float = 1e-8
    minimum_rows: int = 30
    maximum_newton_steps: int = 50
    convergence_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.prior_precision <= 0.0 or self.variance_floor <= 0.0:
            raise ValueError("Bayesian boundary priors must be positive")
        if self.minimum_rows < 10 or self.maximum_newton_steps < 1:
            raise ValueError("Bayesian boundary fitting budget is invalid")
        if self.convergence_tolerance <= 0.0:
            raise ValueError("convergence tolerance must be positive")


class BayesianBoundaryStateModel:
    def __init__(self, config: BayesianBoundaryStateConfig | None = None) -> None:
        self.config = config or BayesianBoundaryStateConfig()

    def fit(
        self,
        states: BoundaryDailyStates,
        labels: BoundaryStateLabels,
        *,
        training_cutoff: np.datetime64,
    ) -> BayesianBoundaryPosterior:
        cutoff = np.datetime64(training_cutoff, "ns")
        self._validate_alignment(states, labels)
        labelled = labels.return_valid & states.valid
        survival_labelled = labels.survival_valid & labelled
        if np.any(labels.target_time[labelled] > cutoff):
            raise ValueError("target_time exceeds the training cutoff")
        if int(labelled.sum()) < self.config.minimum_rows:
            raise ValueError("insufficient matured daily returns for Bayesian boundary fit")
        if int(survival_labelled.sum()) < self.config.minimum_rows:
            raise ValueError("insufficient matured survival labels for Bayesian boundary fit")

        drift_names = tuple(
            (*_DRIFT_BASE, *sorted(
                name
                for name in states.feature_names
                if name.startswith("branch_") and name.endswith("_cumulative_contribution")
            ))
        )
        drift_x, drift_center, drift_scale = self._design(
            states, labels, drift_names, labelled
        )
        y = labels.next_return[labelled]
        drift_mean, drift_covariance, drift_noise = self._gaussian_fit(drift_x, y)
        drift_residual = y - drift_x @ drift_mean

        diffusion_x, diffusion_center, diffusion_scale = self._design(
            states, labels, _DIFFUSION, labelled
        )
        log_variance = np.log(np.square(drift_residual) + self.config.variance_floor)
        diffusion_mean, diffusion_covariance, diffusion_noise = self._gaussian_fit(
            diffusion_x, log_variance
        )

        potential_x, potential_center, potential_scale = self._design(
            states, labels, _POTENTIAL, survival_labelled
        )
        survival_beta, potential_covariance = self._logistic_fit(
            potential_x, labels.survival[survival_labelled]
        )
        # sigmoid(X beta) = exp(-softplus(-X beta)); theta = -beta.
        potential_mean = -survival_beta

        return BayesianBoundaryPosterior(
            drift_feature_names=drift_names,
            drift_center=drift_center,
            drift_scale=drift_scale,
            drift_mean=drift_mean,
            drift_covariance=drift_covariance,
            drift_noise_variance=drift_noise,
            diffusion_feature_names=_DIFFUSION,
            diffusion_center=diffusion_center,
            diffusion_scale=diffusion_scale,
            diffusion_mean=diffusion_mean,
            diffusion_covariance=diffusion_covariance,
            diffusion_noise_variance=diffusion_noise,
            potential_feature_names=_POTENTIAL,
            potential_center=potential_center,
            potential_scale=potential_scale,
            potential_mean=potential_mean,
            potential_covariance=potential_covariance,
            training_cutoff=cutoff,
            fit_rows=int(labelled.sum()),
            survival_fit_rows=int(survival_labelled.sum()),
        )

    @staticmethod
    def _validate_alignment(
        states: BoundaryDailyStates, labels: BoundaryStateLabels
    ) -> None:
        if not np.array_equal(states.sessions, labels.signal_sessions):
            raise ValueError("daily states and labels must share signal sessions")

    def _design(
        self,
        states: BoundaryDailyStates,
        labels: BoundaryStateLabels,
        names: tuple[str, ...],
        selected: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrix = _feature_matrix(states, labels, names)[selected]
        if np.any(~np.isfinite(matrix)):
            raise ValueError("Bayesian boundary features must be finite")
        center = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = (matrix - center) / scale
        return np.column_stack((np.ones(len(matrix)), standardized)), center, scale

    def _gaussian_fit(
        self, design: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        prior = np.eye(design.shape[1]) * self.config.prior_precision
        prior[0, 0] = self.config.prior_precision * 1e-4
        initial = np.linalg.solve(design.T @ design + prior, design.T @ target)
        residual = target - design @ initial
        noise = max(float(np.mean(np.square(residual))), self.config.variance_floor)
        precision = design.T @ design / noise + prior
        covariance = _inverse_positive_definite(precision)
        mean = covariance @ (design.T @ target / noise)
        residual = target - design @ mean
        noise = max(float(np.mean(np.square(residual))), self.config.variance_floor)
        return mean, covariance, noise

    def _logistic_fit(
        self, design: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        coefficient = np.zeros(design.shape[1], dtype=np.float64)
        prior = np.eye(design.shape[1]) * self.config.prior_precision
        prior[0, 0] = self.config.prior_precision * 1e-4
        hessian = prior.copy()
        for _ in range(self.config.maximum_newton_steps):
            linear = np.clip(design @ coefficient, -30.0, 30.0)
            probability = 1.0 / (1.0 + np.exp(-linear))
            weights = np.maximum(probability * (1.0 - probability), 1e-8)
            hessian = design.T @ (design * weights[:, None]) + prior
            gradient = design.T @ (target - probability) - prior @ coefficient
            step = np.linalg.solve(hessian, gradient)
            coefficient += step
            if float(np.max(np.abs(step))) < self.config.convergence_tolerance:
                break
        covariance = _inverse_positive_definite(hessian)
        return coefficient, covariance


def _feature_matrix(
    states: BoundaryDailyStates,
    labels: BoundaryStateLabels,
    names: tuple[str, ...],
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for name in names:
        if name == "historical_index_volatility":
            values = labels.historical_volatility
        elif name == "abs_decoupling_last60":
            values = np.abs(states.feature("decoupling_last60"))
        elif name == "low_coherence_last60":
            values = 1.0 - states.feature("coherence_last60")
        else:
            values = states.feature(name)
        columns.append(np.asarray(values, dtype=np.float64))
    return np.column_stack(columns)


def _inverse_positive_definite(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) * 0.5
    try:
        factor = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError:
        jitter = max(float(np.trace(symmetric)) / max(len(symmetric), 1), 1.0) * 1e-8
        try:
            factor = np.linalg.cholesky(symmetric + np.eye(len(symmetric)) * jitter)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Bayesian posterior precision is not positive definite") from exc
    inverse_factor = np.linalg.solve(factor, np.eye(len(factor)))
    covariance = inverse_factor.T @ inverse_factor
    return (covariance + covariance.T) * 0.5
