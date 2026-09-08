"""Low-parameter Bayesian dynamics fitted independently for each branch."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .convection import CONVECTION_POTENTIAL_FEATURE_NAMES
from .types import TwoBranchDailyStates, TwoBranchLabels


DRIFT_FEATURE_NAMES = (
    "rolling_log_position_60d",
    "rolling_path_efficiency_60d",
)
CAUSAL_DRIFT_FEATURE_NAMES = DRIFT_FEATURE_NAMES


@dataclass(frozen=True, slots=True)
class BayesianBranchModelConfig:
    prior_precision: float = 2.0
    variance_floor: float = 1e-8
    minimum_rows: int = 30
    maximum_newton_steps: int = 50
    convergence_tolerance: float = 1e-8
    neutral_sigma: float = 0.35
    drift_half_life_sessions: float | None = 60.0
    drift_feature_names: tuple[str, ...] = DRIFT_FEATURE_NAMES

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
        if self.drift_feature_names != DRIFT_FEATURE_NAMES:
            raise ValueError(
                "direction model must use the rolling 60-session drift features"
            )

    @property
    def direction_label_sigma(self) -> float:
        return self.neutral_sigma


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
    potential_mode: str = "beta0_c1"
    drift_model: str = "gaussian_return"
    drift_baseline_mode: str = "bayesian_intercept"
    drift_return_scale: float = 1.0
    direction_label_sigma: float = 0.0
    direction_fit_rows: int = 0

    def __post_init__(self) -> None:
        if self.branch < 0:
            raise ValueError("branch posterior index must be non-negative")
        if not np.isfinite(self.diffusion_variance) or self.diffusion_variance <= 0.0:
            raise ValueError("branch diffusion variance must be finite and positive")
        if self.drift_model not in {"gaussian_return", "bayesian_direction_logit"}:
            raise ValueError("unknown branch drift model")
        if self.drift_baseline_mode != "bayesian_intercept":
            raise ValueError("unknown drift baseline mode")
        if not np.isfinite(self.drift_return_scale) or self.drift_return_scale <= 0.0:
            raise ValueError("branch drift return scale must be finite and positive")

    def drift_design(self, states: TwoBranchDailyStates, row: int) -> np.ndarray:
        raw = np.asarray(
            [states.feature(name)[row, self.branch] for name in self.drift_feature_names]
        )
        return np.concatenate((np.ones(1), (raw - self.drift_center) / self.drift_scale))

    def potential_design(self, raw_features: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_features, dtype=np.float64)
        if raw.shape != self.potential_center.shape or np.any(~np.isfinite(raw)):
            raise ValueError("convection potential features are invalid")
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
        potential_features: np.ndarray,
        potential_valid: np.ndarray,
        branch: int,
        training_cutoff: np.datetime64,
    ) -> BayesianBranchPosterior:
        if branch < 0 or branch >= states.values.shape[1]:
            raise ValueError("branch index is outside the state tensor")
        if not np.array_equal(states.sessions, labels.signal_sessions):
            raise ValueError("branch states and labels must align")
        potential_features = np.asarray(potential_features, dtype=np.float64)
        potential_valid = np.asarray(potential_valid, dtype=np.bool_)
        expected_shape = (
            len(states.sessions),
            len(CONVECTION_POTENTIAL_FEATURE_NAMES),
        )
        if potential_features.shape != expected_shape:
            raise ValueError(
                f"convection potential features must have shape {expected_shape}"
            )
        if potential_valid.shape != (len(states.sessions),):
            raise ValueError("convection potential validity must align with sessions")
        cutoff = np.datetime64(training_cutoff, "ns")
        selected = (
            states.valid[:, branch]
            & labels.valid[:, branch]
            & (labels.target_time <= cutoff)
        )
        if int(selected.sum()) < self.config.minimum_rows:
            raise ValueError("insufficient matured branch returns for Bayesian fit")

        drift_names = self.config.drift_feature_names
        drift_raw = np.column_stack(
            [states.feature(name)[:, branch] for name in drift_names]
        )
        selected_target = labels.next_return[selected, branch]
        direction_selected = selected.copy()
        if self.config.direction_label_sigma > 0.0:
            direction_selected &= np.abs(labels.next_return[:, branch]) > (
                self.config.direction_label_sigma
                * labels.historical_volatility[:, branch]
            )
        if int(direction_selected.sum()) < self.config.minimum_rows:
            raise ValueError("insufficient unambiguous branch returns for direction fit")
        drift_x, drift_center, drift_scale = _odd_symmetric_design(
            drift_raw[direction_selected]
        )
        direction_target = labels.next_return[direction_selected, branch]
        direction_mask = direction_selected[selected]
        sample_weight = self._recent_weights(len(selected_target))[direction_mask]
        sample_weight /= float(np.mean(sample_weight))
        drift_mean, drift_covariance = self._constrained_logistic_fit(
            drift_x,
            (direction_target > 0.0).astype(np.float64),
            bounds=[(None, None)] * (len(drift_names) + 1),
            intercept_position=0,
            intercept_prior_multiplier=1.0,
            sample_weight=sample_weight,
        )
        drift_return_scale = max(
            float(np.average(np.abs(direction_target), weights=sample_weight)),
            self.config.variance_floor**0.5,
        )
        selected_design = np.column_stack(
            (np.ones(len(selected_target)), drift_raw[selected] / drift_scale)
        )
        fitted_logit = selected_design @ drift_mean
        fitted_return = np.tanh(fitted_logit * 0.5) * drift_return_scale
        residual = selected_target - fitted_return
        diffusion_variance = max(
            float(
                np.average(
                    np.square(residual),
                    weights=self._recent_weights(len(selected_target)),
                )
            ),
            self.config.variance_floor,
        )

        selected_rows = np.flatnonzero(selected)
        calibration_positions, causal_direction = self._causal_drift_predictions(
            drift_raw[selected],
            selected_target,
            labels.historical_volatility[selected, branch],
        )
        calibration_rows = selected_rows[calibration_positions]
        usable = potential_valid[calibration_rows]
        calibration_rows = calibration_rows[usable]
        causal_direction = causal_direction[usable]
        if len(calibration_rows) < 5:
            raise ValueError("insufficient matured convection rows for potential fit")
        potential_x, potential_center, potential_scale = _standardized_design(
            potential_features[calibration_rows]
        )
        threshold = self.config.neutral_sigma * labels.historical_volatility[
            calibration_rows, branch
        ]
        calibration_target = labels.next_return[calibration_rows, branch]
        drift_survival = (
            (np.sign(calibration_target) == np.sign(causal_direction))
            & (np.abs(calibration_target) > threshold)
            & (np.abs(causal_direction) > 1e-12)
        ).astype(np.float64)
        disruption = 1.0 - drift_survival
        potential_mean, potential_covariance = self._fit_potential(
            potential_x, disruption
        )

        return BayesianBranchPosterior(
            branch=branch,
            drift_feature_names=drift_names,
            drift_center=drift_center,
            drift_scale=drift_scale,
            drift_mean=drift_mean,
            drift_covariance=drift_covariance,
            diffusion_variance=diffusion_variance,
            potential_feature_names=CONVECTION_POTENTIAL_FEATURE_NAMES,
            potential_center=potential_center,
            potential_scale=potential_scale,
            potential_mean=potential_mean,
            potential_covariance=potential_covariance,
            training_cutoff=cutoff,
            fit_rows=int(selected.sum()),
            drift_model="bayesian_direction_logit",
            drift_baseline_mode="bayesian_intercept",
            drift_return_scale=drift_return_scale,
            direction_label_sigma=self.config.direction_label_sigma,
            direction_fit_rows=int(direction_selected.sum()),
        )

    def _fit_potential(
        self, design: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._constrained_logistic_fit(
            design,
            target,
            bounds=[(None, None), *[(0.0, None)] * (design.shape[1] - 1)],
            intercept_position=0,
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

    def _causal_drift_predictions(
        self,
        drift_raw: np.ndarray,
        target: np.ndarray,
        historical_volatility: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        calibration_start = min(
            self.config.minimum_rows,
            max(5, len(target) // 3),
            len(target) - 1,
        )
        positions: list[int] = []
        predictions: list[float] = []
        for position in range(calibration_start, len(target)):
            direction_mask = np.ones(position, dtype=np.bool_)
            if self.config.direction_label_sigma > 0.0:
                direction_mask &= np.abs(target[:position]) > (
                    self.config.direction_label_sigma
                    * historical_volatility[:position]
                )
            if int(direction_mask.sum()) < calibration_start:
                continue
            design, _, scale = _odd_symmetric_design(
                drift_raw[:position][direction_mask]
            )
            weight = self._recent_weights(position)[direction_mask]
            weight /= float(np.mean(weight))
            mean, _ = self._constrained_logistic_fit(
                design,
                (target[:position][direction_mask] > 0.0).astype(np.float64),
                bounds=[(None, None)] * (drift_raw.shape[1] + 1),
                intercept_position=0,
                intercept_prior_multiplier=1.0,
                sample_weight=weight,
            )
            current = np.concatenate(
                (
                    np.ones(1),
                    drift_raw[position] / scale,
                )
            )
            return_scale = max(
                float(
                    np.average(
                        np.abs(target[:position][direction_mask]),
                        weights=weight,
                    )
                ),
                self.config.variance_floor**0.5,
            )
            positions.append(position)
            predictions.append(float(
                np.tanh(float(current @ mean) * 0.5) * return_scale
            ))
        return np.asarray(positions, dtype=np.int64), np.asarray(predictions)

    def _constrained_logistic_fit(
        self,
        design: np.ndarray,
        target: np.ndarray,
        *,
        bounds: list[tuple[float | None, float | None]],
        intercept_position: int | None,
        intercept_prior_multiplier: float = 1e-4,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        weights = (
            np.ones(len(target), dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        prior = np.eye(design.shape[1]) * self.config.prior_precision
        if intercept_position is not None:
            prior[intercept_position, intercept_position] *= (
                intercept_prior_multiplier
            )

        def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
            linear = design @ coefficient
            probability = expit(linear)
            loss = float(
                np.sum(weights * (np.logaddexp(0.0, linear) - target * linear))
                + 0.5 * coefficient @ prior @ coefficient
            )
            gradient = design.T @ (weights * (probability - target)) + prior @ coefficient
            return loss, gradient

        result = minimize(
            objective,
            np.zeros(design.shape[1], dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": self.config.maximum_newton_steps,
                "ftol": self.config.convergence_tolerance,
            },
        )
        if not result.success:
            raise ValueError(f"convection potential fit failed: {result.message}")
        coefficient = np.asarray(result.x, dtype=np.float64)
        linear = design @ coefficient
        probability = expit(linear)
        curvature = weights * np.maximum(probability * (1.0 - probability), 1e-8)
        hessian = design.T @ (design * curvature[:, None]) + prior
        return coefficient, _inverse_positive_definite(hessian)


def _standardized_design(
    raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.any(~np.isfinite(raw)):
        raise ValueError("branch Bayesian features must be finite")
    center = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return np.column_stack((np.ones(len(raw)), (raw - center) / scale)), center, scale


def _odd_symmetric_design(
    raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.any(~np.isfinite(raw)):
        raise ValueError("branch direction features must be finite")
    scale = np.sqrt(np.mean(np.square(raw), axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    center = np.zeros(raw.shape[1], dtype=np.float64)
    return np.column_stack((np.ones(len(raw)), raw / scale)), center, scale


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
