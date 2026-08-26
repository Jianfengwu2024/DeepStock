"""Monte Carlo Feynman-Kac forecasts for Bayesian boundary dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from .bayesian_state import BayesianBoundaryPosterior, BoundaryStateLabels
from .branch_model import BayesianBranchPosterior, BayesianConvectionDriftPosterior
from .types import BoundaryDailyStates
from .types import TwoBranchDailyStates


@dataclass(frozen=True, slots=True)
class BoundaryFeynmanKacConfig:
    paths: int = 10_000
    seed: int = 7
    use_effective_potential: bool = True
    use_convection_drift_correction: bool = False

    def __post_init__(self) -> None:
        if not 1_000 <= self.paths <= 100_000:
            raise ValueError("Feynman-Kac paths must lie in [1000, 100000]")
        if self.seed < 0:
            raise ValueError("Feynman-Kac seed must be non-negative")


@dataclass(frozen=True, slots=True)
class BoundaryFeynmanKacForecast:
    raw_state_probabilities: tuple[float, float, float]
    survival_joint_probabilities: tuple[float, float, float]
    trend_survival_probability: float
    expected_return: float
    epistemic_return_variance: float
    aleatoric_return_variance: float
    total_return_variance: float
    expected_drift: float
    expected_diffusion_variance: float
    expected_effective_potential: float
    bullish_probability_std: float
    expected_attenuated_drift: float = 0.0
    bullish_drift_probability: float = 0.5
    drift_signal_to_noise: float = 0.0
    expected_base_drift: float = 0.0
    expected_convection_drift_correction: float = 0.0


class BoundaryFeynmanKacForecaster:
    def __init__(self, config: BoundaryFeynmanKacConfig | None = None) -> None:
        self.config = config or BoundaryFeynmanKacConfig()

    def predict(
        self,
        posterior: BayesianBoundaryPosterior,
        states: BoundaryDailyStates,
        row: int,
        labels: BoundaryStateLabels,
        *,
        neutral_threshold: float,
    ) -> BoundaryFeynmanKacForecast:
        if not np.isfinite(neutral_threshold) or neutral_threshold < 0.0:
            raise ValueError("neutral threshold must be finite and non-negative")
        resolved_row = row if row >= 0 else len(states.sessions) + row
        if resolved_row < 0 or resolved_row >= len(states.sessions):
            raise IndexError("boundary forecast row is out of range")
        if not states.valid[resolved_row]:
            raise ValueError("cannot forecast an invalid boundary state")

        rng = np.random.default_rng(self.config.seed)
        drift_design = posterior.design_vector("drift", states, resolved_row, labels)
        diffusion_design = posterior.design_vector("diffusion", states, resolved_row, labels)
        potential_design = posterior.design_vector("potential", states, resolved_row, labels)

        drift_coefficients = rng.multivariate_normal(
            posterior.drift_mean, posterior.drift_covariance, size=self.config.paths
        )
        diffusion_coefficients = rng.multivariate_normal(
            posterior.diffusion_mean,
            posterior.diffusion_covariance,
            size=self.config.paths,
        )
        potential_coefficients = rng.multivariate_normal(
            posterior.potential_mean,
            posterior.potential_covariance,
            size=self.config.paths,
        )

        drift = drift_coefficients @ drift_design
        log_variance = np.clip(diffusion_coefficients @ diffusion_design, -30.0, 10.0)
        diffusion_variance = np.exp(log_variance)
        if self.config.use_effective_potential:
            effective_potential = np.logaddexp(
                0.0, potential_coefficients @ potential_design
            )
            survival = np.exp(-effective_potential)
        else:
            effective_potential = np.zeros(self.config.paths, dtype=np.float64)
            survival = np.ones(self.config.paths, dtype=np.float64)

        returns = drift + np.sqrt(diffusion_variance) * rng.standard_normal(self.config.paths)
        bear = returns < -neutral_threshold
        bull = returns > neutral_threshold
        flat = ~(bear | bull)
        classes = (bear, flat, bull)

        raw = tuple(float(np.mean(mask)) for mask in classes)
        survival_joint = tuple(float(np.mean(survival * mask)) for mask in classes)
        epistemic = float(np.var(drift))
        aleatoric = float(np.mean(diffusion_variance))
        conditional_bullish = ndtr(
            (drift - neutral_threshold) / np.sqrt(diffusion_variance)
        )

        return BoundaryFeynmanKacForecast(
            raw_state_probabilities=raw,
            survival_joint_probabilities=survival_joint,
            trend_survival_probability=float(np.mean(survival)),
            expected_return=float(np.mean(returns)),
            epistemic_return_variance=epistemic,
            aleatoric_return_variance=aleatoric,
            total_return_variance=epistemic + aleatoric,
            expected_drift=float(np.mean(drift)),
            expected_diffusion_variance=aleatoric,
            expected_effective_potential=float(np.mean(effective_potential)),
            bullish_probability_std=float(np.std(conditional_bullish)),
            expected_attenuated_drift=float(np.mean(drift * survival)),
            bullish_drift_probability=float(np.mean(drift > 0.0)),
            drift_signal_to_noise=float(
                abs(np.mean(drift)) / np.sqrt(epistemic + aleatoric)
            ),
        )


class BranchFeynmanKacForecaster:
    """Forecast one branch with potential acting directly on its drift."""

    def __init__(self, config: BoundaryFeynmanKacConfig | None = None) -> None:
        self.config = config or BoundaryFeynmanKacConfig()

    def predict(
        self,
        posterior: BayesianBranchPosterior,
        states: TwoBranchDailyStates,
        row: int,
        *,
        potential_features: np.ndarray | None = None,
        convection_drift_posterior: BayesianConvectionDriftPosterior | None = None,
        convection_drift_features: np.ndarray | None = None,
        neutral_threshold: float,
    ) -> BoundaryFeynmanKacForecast:
        if not np.isfinite(neutral_threshold) or neutral_threshold < 0.0:
            raise ValueError("neutral threshold must be finite and non-negative")
        resolved_row = row if row >= 0 else len(states.sessions) + row
        if resolved_row < 0 or resolved_row >= len(states.sessions):
            raise IndexError("branch forecast row is out of range")
        if not states.valid[resolved_row, posterior.branch]:
            raise ValueError("cannot forecast an invalid branch state")

        rng = np.random.default_rng(self.config.seed)
        drift_coefficients = rng.multivariate_normal(
            posterior.drift_mean,
            posterior.drift_covariance,
            size=self.config.paths,
        )
        base_drift = drift_coefficients @ posterior.drift_design(states, resolved_row)
        convection_correction = np.zeros(self.config.paths, dtype=np.float64)
        if self.config.use_convection_drift_correction:
            if convection_drift_posterior is None or convection_drift_features is None:
                raise ValueError("convection drift posterior and features are required")
            correction_rng = np.random.default_rng(self.config.seed + 1_000_003)
            correction_coefficients = correction_rng.multivariate_normal(
                convection_drift_posterior.mean,
                convection_drift_posterior.covariance,
                size=self.config.paths,
            )
            convection_correction = (
                correction_coefficients
                @ convection_drift_posterior.design(convection_drift_features)
            )
        drift = base_drift + convection_correction
        return_shocks = rng.standard_normal(self.config.paths)
        if self.config.use_effective_potential:
            if potential_features is None:
                raise ValueError("convection potential features are required")
            potential_coefficients = rng.multivariate_normal(
                posterior.potential_mean,
                posterior.potential_covariance,
                size=self.config.paths,
            )
            potential_coefficients[:, 1:] = np.maximum(
                potential_coefficients[:, 1:], 0.0
            )
            potential = np.logaddexp(
                0.0,
                potential_coefficients @ posterior.potential_design(potential_features),
            )
            survival = np.exp(-potential)
        else:
            potential = np.zeros(self.config.paths, dtype=np.float64)
            survival = np.ones(self.config.paths, dtype=np.float64)
        attenuated_drift = drift * survival
        diffusion = posterior.diffusion_variance
        returns = drift + np.sqrt(diffusion) * return_shocks
        classes = (
            returns < -neutral_threshold,
            np.abs(returns) <= neutral_threshold,
            returns > neutral_threshold,
        )
        probabilities = tuple(float(np.mean(mask)) for mask in classes)
        survival_joint = tuple(float(np.mean(survival * mask)) for mask in classes)
        conditional_bullish = ndtr(
            (drift - neutral_threshold) / np.sqrt(diffusion)
        )
        epistemic = float(np.var(drift))
        expected_drift = float(np.mean(drift))

        return BoundaryFeynmanKacForecast(
            raw_state_probabilities=probabilities,
            survival_joint_probabilities=survival_joint,
            trend_survival_probability=float(np.mean(survival)),
            expected_return=expected_drift,
            epistemic_return_variance=epistemic,
            aleatoric_return_variance=float(diffusion),
            total_return_variance=float(epistemic + diffusion),
            expected_drift=expected_drift,
            expected_diffusion_variance=float(diffusion),
            expected_effective_potential=float(np.mean(potential)),
            bullish_probability_std=float(np.std(conditional_bullish)),
            expected_attenuated_drift=float(np.mean(attenuated_drift)),
            bullish_drift_probability=float(np.mean(drift > 0.0)),
            drift_signal_to_noise=float(
                abs(expected_drift) / np.sqrt(epistemic + diffusion)
            ),
            expected_base_drift=float(np.mean(base_drift)),
            expected_convection_drift_correction=float(
                np.mean(convection_correction)
            ),
        )
