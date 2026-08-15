"""Short-horizon Feynman-Kac posterior solver for FKStock."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from math import erf, sqrt

import numpy as np

from ..config import REGIME_SCORES
from ..evaluation import PredictionRecord
from .generator import BayesianGeneratorPosterior, FKFeatureRow


@dataclass(frozen=True, slots=True)
class FeynmanKacConfig:
    strong_threshold_sigma: float = 0.75

    def __post_init__(self) -> None:
        if self.strong_threshold_sigma <= 0:
            raise ValueError("strong-state threshold must be positive")


@dataclass(frozen=True, slots=True)
class FeynmanKacPotentialConfig:
    """Research-only effective-potential terms retained for reproducible ablation."""

    potential_floor: float = 0.0
    noise_potential_weight: float = 0.0
    uncertainty_potential_weight: float = 0.0
    incoherence_potential_weight: float = 0.0
    maximum_potential: float = 0.0
    mc_samples: int = 16
    seed: int = 7

    def __post_init__(self) -> None:
        if min(
            self.potential_floor,
            self.noise_potential_weight,
            self.uncertainty_potential_weight,
            self.incoherence_potential_weight,
        ) < 0:
            raise ValueError("effective-potential terms cannot be negative")
        if self.maximum_potential < self.potential_floor:
            raise ValueError("maximum potential cannot be below its floor")
        if self.mc_samples < 2:
            raise ValueError("Feynman-Kac posterior needs at least two samples")


class FeynmanKacSolver:
    """Apply an effective killing potential to a Bayesian local diffusion."""

    def __init__(
        self,
        config: FeynmanKacConfig | None = None,
        potential_config: FeynmanKacPotentialConfig | None = None,
    ) -> None:
        self.config = config or FeynmanKacConfig()
        self.potential_config = potential_config

    def predict(
        self, row: FKFeatureRow, posterior: BayesianGeneratorPosterior
    ) -> tuple[PredictionRecord, dict[str, float]]:
        design = posterior.design_vector(row)
        raw_drift = float(design @ posterior.coefficient_mean)
        raw_epistemic = max(
            float(design @ posterior.coefficient_covariance @ design), 0.0
        )
        total_raw_variance = posterior.noise_variance + raw_epistemic
        epistemic_fraction = raw_epistemic / max(total_raw_variance, 1e-12)
        noise_fraction = row.stochastic_current_fraction / (
            1.0 + row.stochastic_current_fraction
        )
        potential = 0.0
        if self.potential_config is not None:
            potential = float(
                np.clip(
                    self.potential_config.potential_floor
                    + self.potential_config.noise_potential_weight * noise_fraction
                    + self.potential_config.uncertainty_potential_weight
                    * epistemic_fraction
                    + self.potential_config.incoherence_potential_weight
                    * (1.0 - row.current_coherence),
                    self.potential_config.potential_floor,
                    self.potential_config.maximum_potential,
                )
            )
        survival = float(np.exp(-potential))
        drift = raw_drift * survival
        epistemic_variance = raw_epistemic * survival**2
        aleatoric_variance = posterior.noise_variance

        sample_count = (
            self.potential_config.mc_samples if self.potential_config is not None else 16
        )
        rng = np.random.default_rng(self._row_seed(row))
        coefficients = rng.multivariate_normal(
            posterior.coefficient_mean,
            posterior.coefficient_covariance,
            size=sample_count,
            check_valid="ignore",
        )
        sampled_drift = (coefficients @ design) * survival
        conditional = np.asarray(
            [
                _class_probabilities(
                    float(value),
                    sqrt(aleatoric_variance),
                    self.config.strong_threshold_sigma,
                )
                for value in sampled_drift
            ],
            dtype=np.float64,
        )
        probability_mean = conditional.mean(axis=0)
        probability_mean /= probability_mean.sum()
        probability_epistemic = conditional.var(axis=0, ddof=1)
        probability_aleatoric = (conditional * (1.0 - conditional)).mean(axis=0)
        total_variance = aleatoric_variance + epistemic_variance
        score = float(probability_mean @ np.asarray(REGIME_SCORES))
        record = PredictionRecord(
            symbol=row.symbol,
            timeframe_id=1,
            signal_time=row.signal_time,
            execution_time=row.execution_time,
            target_time=row.target_time,
            target=row.target,
            forward_return=row.forward_return,
            standardized_return=row.standardized_return,
            return_location=row.return_location,
            return_scale=row.return_scale,
            prior_probability=posterior.class_prior,
            probability_mean=tuple(float(value) for value in probability_mean),
            aleatoric_variance=tuple(float(value) for value in probability_aleatoric),
            epistemic_variance=tuple(float(value) for value in probability_epistemic),
            expected_score=score,
            expected_return=float(drift * row.return_scale),
            expected_standardized_return=drift,
            return_variance=float(total_variance * row.return_scale**2),
            aleatoric_return_variance=float(aleatoric_variance * row.return_scale**2),
            epistemic_return_variance=float(epistemic_variance * row.return_scale**2),
            context_length=row.matured_return_count,
        )
        diagnostics = {
            "raw_drift": raw_drift,
            "effective_drift": drift,
            "diffusion_variance": aleatoric_variance,
            "epistemic_variance": epistemic_variance,
            "effective_potential": potential,
            "survival_factor": survival,
            "stochastic_current_fraction": row.stochastic_current_fraction,
            "current_coherence": row.current_coherence,
            "continuity_error": row.continuity_error,
        }
        return record, diagnostics

    def _row_seed(self, row: FKFeatureRow) -> int:
        seed = self.potential_config.seed if self.potential_config is not None else 7
        payload = f"{seed}|{row.symbol}|{row.signal_time}".encode("utf-8")
        digest = blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "little")


def _class_probabilities(mean: float, standard_deviation: float, threshold: float) -> np.ndarray:
    standard_deviation = max(standard_deviation, 1e-8)

    def cdf(value: float) -> float:
        return 0.5 * (1.0 + erf((value - mean) / (standard_deviation * sqrt(2.0))))

    lower = cdf(-threshold)
    middle = cdf(0.0)
    upper = cdf(threshold)
    result = np.asarray(
        (lower, middle - lower, upper - middle, 1.0 - upper), dtype=np.float64
    )
    return np.maximum(result, 0.0)
