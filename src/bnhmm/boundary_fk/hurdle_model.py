"""Causal low-parameter Bayesian hurdle model for tradable sector moves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, ndtr

from .types import TwoBranchDailyStates, TwoBranchLabels


@dataclass(frozen=True, slots=True)
class BayesianHurdleModelConfig:
    prior_precision: float = 2.0
    variance_floor: float = 1e-8
    minimum_rows: int = 30
    minimum_direction_rows: int = 20
    maximum_steps: int = 50
    convergence_tolerance: float = 1e-8
    neutral_sigma: float = 0.35
    half_life_sessions: float | None = 60.0
    feature_names: tuple[str, ...] = ("signed_path_efficiency",)
    calibration_fraction: float = 0.20

    def __post_init__(self) -> None:
        if self.prior_precision <= 0.0 or self.variance_floor <= 0.0:
            raise ValueError("hurdle Bayesian priors must be positive")
        if self.minimum_rows < 10 or self.minimum_direction_rows < 10:
            raise ValueError("hurdle fitting requires at least ten rows")
        if self.maximum_steps < 1 or self.convergence_tolerance <= 0.0:
            raise ValueError("hurdle fitting budget is invalid")
        if self.neutral_sigma <= 0.0:
            raise ValueError("hurdle neutral sigma must be positive")
        if self.half_life_sessions is not None and self.half_life_sessions <= 0.0:
            raise ValueError("hurdle half-life must be positive when provided")
        if not self.feature_names or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("hurdle feature names must be unique and non-empty")
        if not 0.1 <= self.calibration_fraction <= 0.4:
            raise ValueError("hurdle calibration fraction must lie in [0.1, 0.4]")


@dataclass(frozen=True, slots=True)
class HurdleDirectionEstimate:
    significant_move_probability: float
    conditional_bullish_probability: float
    prior_bullish_probability: float
    feature_log_odds_delta: float


@dataclass(frozen=True, slots=True)
class BayesianHurdlePosterior:
    branch: int
    feature_names: tuple[str, ...]
    move_scale: np.ndarray
    move_mean: np.ndarray
    move_covariance: np.ndarray
    direction_scale: np.ndarray
    direction_mean: np.ndarray
    direction_covariance: np.ndarray
    return_scale: float
    training_cutoff: np.datetime64
    fit_rows: int
    direction_fit_rows: int
    direction_label_sigma: float
    minimum_feature_evidence_probability: float
    calibration_rows: int

    def __post_init__(self) -> None:
        width = len(self.feature_names) + 1
        if self.branch < 0 or self.fit_rows < self.direction_fit_rows:
            raise ValueError("hurdle posterior row counts are invalid")
        if self.direction_fit_rows < 1 or self.direction_label_sigma <= 0.0:
            raise ValueError("hurdle direction posterior is empty")
        if not 0.5 <= self.minimum_feature_evidence_probability <= 0.95:
            raise ValueError("hurdle feature evidence probability is invalid")
        if self.calibration_rows < 1:
            raise ValueError("hurdle posterior requires causal calibration rows")
        if self.move_mean.shape != (width,) or self.direction_mean.shape != (width,):
            raise ValueError("hurdle posterior coefficients do not match features")
        if self.move_covariance.shape != (width, width) or self.direction_covariance.shape != (
            width,
            width,
        ):
            raise ValueError("hurdle posterior covariance does not match features")
        if not np.isfinite(self.return_scale) or self.return_scale <= 0.0:
            raise ValueError("hurdle return scale must be finite and positive")

    def move_design(self, states: TwoBranchDailyStates, row: int) -> np.ndarray:
        raw = self._raw_features(states, row)
        return np.concatenate((np.ones(1), np.abs(raw) / self.move_scale))

    def direction_design(self, states: TwoBranchDailyStates, row: int) -> np.ndarray:
        raw = self._raw_features(states, row)
        return np.concatenate((np.ones(1), raw / self.direction_scale))

    def estimate(self, states: TwoBranchDailyStates, row: int) -> HurdleDirectionEstimate:
        move_design = self.move_design(states, row)
        direction_design = self.direction_design(states, row)
        return HurdleDirectionEstimate(
            significant_move_probability=float(expit(move_design @ self.move_mean)),
            conditional_bullish_probability=float(
                expit(direction_design @ self.direction_mean)
            ),
            prior_bullish_probability=float(expit(self.direction_mean[0])),
            feature_log_odds_delta=float(
                direction_design[1:] @ self.direction_mean[1:]
            ),
        )

    def _raw_features(self, states: TwoBranchDailyStates, row: int) -> np.ndarray:
        if row < 0 or row >= len(states.sessions):
            raise IndexError("hurdle forecast row is out of range")
        if not states.valid[row, self.branch]:
            raise ValueError("cannot estimate an invalid hurdle state")
        raw = np.asarray(
            [states.feature(name)[row, self.branch] for name in self.feature_names],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(raw)):
            raise ValueError("hurdle features must be finite")
        return raw


class BayesianHurdleModel:
    def __init__(self, config: BayesianHurdleModelConfig | None = None) -> None:
        self.config = config or BayesianHurdleModelConfig()

    def fit(
        self,
        states: TwoBranchDailyStates,
        labels: TwoBranchLabels,
        *,
        branch: int,
        training_cutoff: np.datetime64,
    ) -> BayesianHurdlePosterior:
        if branch < 0 or branch >= states.values.shape[1]:
            raise ValueError("hurdle branch index is outside the state tensor")
        if not np.array_equal(states.sessions, labels.signal_sessions):
            raise ValueError("hurdle states and labels must align")
        cutoff = np.datetime64(training_cutoff, "ns")
        selected = (
            states.valid[:, branch]
            & labels.valid[:, branch]
            & (labels.target_time <= cutoff)
        )
        if int(selected.sum()) < self.config.minimum_rows:
            raise ValueError("insufficient matured rows for hurdle fit")

        raw = np.column_stack(
            [states.feature(name)[:, branch] for name in self.config.feature_names]
        )[selected]
        target_return = labels.next_return[selected, branch]
        volatility = labels.historical_volatility[selected, branch]
        significant = np.abs(target_return) > self.config.neutral_sigma * volatility
        if int(significant.sum()) < self.config.minimum_direction_rows:
            raise ValueError("insufficient significant returns for hurdle direction fit")

        weights = self._recent_weights(len(target_return))
        move_scale = _root_mean_square_scale(raw)
        move_design = np.column_stack((np.ones(len(raw)), np.abs(raw) / move_scale))
        move_mean, move_covariance = self._fit_logistic(
            move_design,
            significant.astype(np.float64),
            weights,
        )

        direction_raw = raw[significant]
        direction_scale = _root_mean_square_scale(direction_raw)
        direction_design = np.column_stack(
            (np.ones(len(direction_raw)), direction_raw / direction_scale)
        )
        direction_weights = weights[significant]
        direction_weights /= float(np.mean(direction_weights))
        direction_target = (target_return[significant] > 0.0).astype(np.float64)
        evidence_probability, calibration_rows = self._calibrate_feature_evidence(
            raw,
            target_return,
            volatility,
            weights,
        )
        direction_mean, direction_covariance = self._fit_logistic(
            direction_design,
            direction_target,
            direction_weights,
        )
        return_scale = max(
            float(np.average(np.abs(target_return[significant]), weights=direction_weights)),
            self.config.variance_floor**0.5,
        )
        return BayesianHurdlePosterior(
            branch=branch,
            feature_names=self.config.feature_names,
            move_scale=move_scale,
            move_mean=move_mean,
            move_covariance=move_covariance,
            direction_scale=direction_scale,
            direction_mean=direction_mean,
            direction_covariance=direction_covariance,
            return_scale=return_scale,
            training_cutoff=cutoff,
            fit_rows=int(selected.sum()),
            direction_fit_rows=int(significant.sum()),
            direction_label_sigma=self.config.neutral_sigma,
            minimum_feature_evidence_probability=evidence_probability,
            calibration_rows=calibration_rows,
        )

    def _calibrate_feature_evidence(
        self,
        raw: np.ndarray,
        target_return: np.ndarray,
        volatility: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[float, int]:
        calibration_rows = max(5, int(np.ceil(len(raw) * self.config.calibration_fraction)))
        split = len(raw) - calibration_rows
        prefix_significant = np.abs(target_return[:split]) > (
            self.config.neutral_sigma * volatility[:split]
        )
        calibration_significant = np.abs(target_return[split:]) > (
            self.config.neutral_sigma * volatility[split:]
        )
        if int(prefix_significant.sum()) < 10 or int(calibration_significant.sum()) < 4:
            return 0.75, calibration_rows

        prefix_raw = raw[:split][prefix_significant]
        scale = _root_mean_square_scale(prefix_raw)
        design = np.column_stack((np.ones(len(prefix_raw)), prefix_raw / scale))
        prefix_weights = weights[:split][prefix_significant]
        prefix_weights /= float(np.mean(prefix_weights))
        mean, covariance = self._fit_logistic(
            design,
            (target_return[:split][prefix_significant] > 0.0).astype(np.float64),
            prefix_weights,
        )
        calibration_design = raw[split:] / scale
        delta_mean = calibration_design @ mean[1:]
        feature_covariance = covariance[1:, 1:]
        delta_variance = np.einsum(
            "ij,jk,ik->i",
            calibration_design,
            feature_covariance,
            calibration_design,
        )
        delta_std = np.sqrt(np.maximum(delta_variance, self.config.variance_floor))
        feature_bullish = ndtr(delta_mean / delta_std)
        actual_bullish = target_return[split:] > 0.0

        best_threshold = 0.75
        best_score: tuple[float, float, int] | None = None
        for threshold in (0.51, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
            predicted = np.zeros(len(feature_bullish), dtype=np.int8)
            predicted[feature_bullish >= threshold] = 1
            predicted[feature_bullish <= 1.0 - threshold] = -1
            usable = calibration_significant & (predicted != 0)
            if int(np.sum(predicted[usable] == 1)) < 1 or int(
                np.sum(predicted[usable] == -1)
            ) < 1:
                continue
            actual = actual_bullish[usable]
            if np.all(actual) or not np.any(actual):
                continue
            bull_recall = float(np.mean(predicted[usable][actual] == 1))
            bear_recall = float(np.mean(predicted[usable][~actual] == -1))
            balanced = 0.5 * (bull_recall + bear_recall)
            accuracy = float(
                np.mean(predicted[usable] == np.where(actual, 1, -1))
            )
            score = (balanced, accuracy, int(np.sum(usable)))
            if best_score is None or score > best_score:
                best_score = score
                best_threshold = threshold
        return best_threshold, calibration_rows

    def _recent_weights(self, rows: int) -> np.ndarray:
        half_life = self.config.half_life_sessions
        if half_life is None:
            return np.ones(rows, dtype=np.float64)
        age = (rows - 1) - np.arange(rows, dtype=np.float64)
        weights = np.exp2(-age / half_life)
        return weights / float(np.mean(weights))

    def _fit_logistic(
        self,
        design: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        prior = np.eye(design.shape[1]) * self.config.prior_precision
        prior[0, 0] *= 1e-4

        def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
            linear = design @ coefficient
            probability = expit(linear)
            loss = float(
                np.sum(
                    sample_weight
                    * (np.logaddexp(0.0, linear) - target * linear)
                )
                + 0.5 * coefficient @ prior @ coefficient
            )
            gradient = (
                design.T @ (sample_weight * (probability - target))
                + prior @ coefficient
            )
            return loss, gradient

        result = minimize(
            objective,
            np.zeros(design.shape[1], dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self.config.maximum_steps,
                "ftol": self.config.convergence_tolerance,
            },
        )
        if not result.success:
            raise ValueError(f"hurdle Bayesian fit failed: {result.message}")
        mean = np.asarray(result.x, dtype=np.float64)
        probability = expit(design @ mean)
        curvature = sample_weight * np.maximum(probability * (1.0 - probability), 1e-8)
        hessian = design.T @ (design * curvature[:, None]) + prior
        return mean, _inverse_positive_definite(hessian)


def _root_mean_square_scale(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 2 or np.any(~np.isfinite(raw)):
        raise ValueError("hurdle features must be a finite matrix")
    scale = np.sqrt(np.mean(np.square(raw), axis=0))
    return np.where(scale > 1e-8, scale, 1.0)


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
