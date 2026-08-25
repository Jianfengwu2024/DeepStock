"""Low-parameter Bayesian dynamics fitted independently for each branch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import TwoBranchDailyStates, TwoBranchLabels


DRIFT_FEATURE_NAMES = (
    "return_1d",
    "return_mean_5",
    "return_mean_20",
    "coherence_1d",
)
POTENTIAL_FEATURE_NAMES = (
    "abs_return_1d",
    "volatility_20",
    "low_coherence_1d",
)


@dataclass(frozen=True, slots=True)
class BayesianBranchModelConfig:
    prior_precision: float = 2.0
    variance_floor: float = 1e-8
    minimum_rows: int = 30
    maximum_newton_steps: int = 50
    convergence_tolerance: float = 1e-8
    neutral_sigma: float = 0.35
    drift_half_life_sessions: float | None = 60.0

    def __post_init__(self) -> None:
        if self.prior_precision <= 0.0 or self.variance_floor <= 0.0:
            raise ValueError("branch Bayesian priors must be positive")
        if self.minimum_rows < 10 or self.maximum_newton_steps < 1:
            raise ValueError("branch Bayesian fitting budget is invalid")
        if self.convergence_tolerance <= 0.0 or self.neutral_sigma <= 0.0:
            raise ValueError("branch Bayesian tolerances must be positive")
        if (
            self.drift_half_life_sessions is not None
            and self.drift_half_life_sessions <= 0.0
        ):
            raise ValueError("drift half-life must be positive when provided")


@dataclass(frozen=True, slots=True)
class BayesianBranchPosterior:
    branch: int
    drift_feature_names: tuple[str, ...]
    drift_center: np.ndarray
    drift_scale: np.ndarray
    drift_mean: np.ndarray
    drift_covariance: np.ndarray
    diffusion_variance: float
    potential_feature_names: tuple[str, ...]
    potential_center: np.ndarray
    potential_scale: np.ndarray
    potential_mean: np.ndarray
    potential_covariance: np.ndarray
    training_cutoff: np.datetime64
    fit_rows: int

    def __post_init__(self) -> None:
        if self.branch not in (0, 1):
            raise ValueError("branch posterior index must be zero or one")
        if not np.isfinite(self.diffusion_variance) or self.diffusion_variance <= 0.0:
            raise ValueError("branch diffusion variance must be finite and positive")

    def drift_design(self, states: TwoBranchDailyStates, row: int) -> np.ndarray:
        raw = np.asarray(
            [states.feature(name)[row, self.branch] for name in self.drift_feature_names]
        )
        return np.concatenate((np.ones(1), (raw - self.drift_center) / self.drift_scale))

    def potential_design(self, states: TwoBranchDailyStates, row: int) -> np.ndarray:
        raw = _potential_matrix(states, self.branch)[row]
        return np.concatenate(
            (np.ones(1), (raw - self.potential_center) / self.potential_scale)
        )


class BayesianBranchModel:
    def __init__(self, config: BayesianBranchModelConfig | None = None) -> None:
        self.config = config or BayesianBranchModelConfig()

    def fit(
        self,
        states: TwoBranchDailyStates,
        labels: TwoBranchLabels,
        *,
        branch: int,
        training_cutoff: np.datetime64,
    ) -> BayesianBranchPosterior:
        if branch not in (0, 1):
            raise ValueError("branch must be zero or one")
        if not np.array_equal(states.sessions, labels.signal_sessions):
            raise ValueError("branch states and labels must align")
        cutoff = np.datetime64(training_cutoff, "ns")
        selected = (
            states.valid[:, branch]
            & labels.valid[:, branch]
            & (labels.target_time <= cutoff)
        )
        if int(selected.sum()) < self.config.minimum_rows:
            raise ValueError("insufficient matured branch returns for Bayesian fit")

        drift_raw = np.column_stack(
            [states.feature(name)[:, branch] for name in DRIFT_FEATURE_NAMES]
        )
        drift_x, drift_center, drift_scale = _standardized_design(
            drift_raw[selected]
        )
        target = labels.next_return[selected, branch]
        sample_weight = self._recent_weights(len(target))
        drift_mean, drift_covariance = self._gaussian_fit(
            drift_x, target, sample_weight
        )
        residual = target - drift_x @ drift_mean
        diffusion_variance = max(
            float(np.average(np.square(residual), weights=sample_weight)),
            self.config.variance_floor,
        )

        potential_raw = _potential_matrix(states, branch)
        potential_x, potential_center, potential_scale = _standardized_design(
            potential_raw[selected]
        )
        direction = states.feature("return_mean_5")[selected, branch]
        threshold = self.config.neutral_sigma * labels.historical_volatility[
            selected, branch
        ]
        survival = (
            (np.sign(target) == np.sign(direction))
            & (np.abs(target) > threshold)
            & (np.abs(direction) > 1e-12)
        ).astype(np.float64)
        survival_beta, potential_covariance = self._logistic_fit(
            potential_x, survival
        )

        return BayesianBranchPosterior(
            branch=branch,
            drift_feature_names=DRIFT_FEATURE_NAMES,
            drift_center=drift_center,
            drift_scale=drift_scale,
            drift_mean=drift_mean,
            drift_covariance=drift_covariance,
            diffusion_variance=diffusion_variance,
            potential_feature_names=POTENTIAL_FEATURE_NAMES,
            potential_center=potential_center,
            potential_scale=potential_scale,
            potential_mean=-survival_beta,
            potential_covariance=potential_covariance,
            training_cutoff=cutoff,
            fit_rows=int(selected.sum()),
        )

    def _gaussian_fit(
        self,
        design: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        weighted_design = design * sample_weight[:, None]
        prior = np.eye(design.shape[1]) * self.config.prior_precision
        prior[0, 0] *= 1e-4
        initial = np.linalg.solve(
            design.T @ weighted_design + prior,
            weighted_design.T @ target,
        )
        residual = target - design @ initial
        noise = max(
            float(np.average(np.square(residual), weights=sample_weight)),
            self.config.variance_floor,
        )
        precision = design.T @ weighted_design / noise + prior
        covariance = _inverse_positive_definite(precision)
        mean = covariance @ (weighted_design.T @ target / noise)
        return mean, covariance

    def _recent_weights(self, rows: int) -> np.ndarray:
        half_life = self.config.drift_half_life_sessions
        if half_life is None:
            return np.ones(rows, dtype=np.float64)
        age = (rows - 1) - np.arange(rows, dtype=np.float64)
        weights = np.exp2(-age / half_life)
        return weights / float(np.mean(weights))

    def _logistic_fit(
        self, design: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        coefficient = np.zeros(design.shape[1], dtype=np.float64)
        prior = np.eye(design.shape[1]) * self.config.prior_precision
        prior[0, 0] *= 1e-4
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
        return coefficient, _inverse_positive_definite(hessian)


def _potential_matrix(states: TwoBranchDailyStates, branch: int) -> np.ndarray:
    return np.column_stack(
        (
            np.abs(states.feature("return_1d")[:, branch]),
            states.feature("volatility_20")[:, branch],
            1.0 - states.feature("coherence_1d")[:, branch],
        )
    )


def _standardized_design(
    raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.any(~np.isfinite(raw)):
        raise ValueError("branch Bayesian features must be finite")
    center = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return np.column_stack((np.ones(len(raw)), (raw - center) / scale)), center, scale


def _inverse_positive_definite(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) * 0.5
    jitter = max(float(np.trace(symmetric)) / max(len(symmetric), 1), 1.0) * 1e-10
    try:
        factor = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError:
        factor = np.linalg.cholesky(symmetric + np.eye(len(symmetric)) * jitter)
    inverse_factor = np.linalg.solve(factor, np.eye(len(factor)))
    covariance = inverse_factor.T @ inverse_factor
    return (covariance + covariance.T) * 0.5
