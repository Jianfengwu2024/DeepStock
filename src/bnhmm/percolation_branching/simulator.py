"""Posterior branching simulation for next-session four-phase probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .branching import spectral_radius
from .types import (
    PHASE_NAMES,
    BranchingPosterior,
    CriticalStateForecast,
    PercolationPrediction,
)


@dataclass(frozen=True, slots=True)
class PercolationSimulator:
    paths: int = 10_000
    maximum_generations: int = 240
    maximum_events: int = 10_000
    model_version: str = "percolation-branching-v1"

    def __post_init__(self) -> None:
        if self.paths < 1 or self.maximum_generations < 1 or self.maximum_events < 1:
            raise ValueError("simulation budgets must be positive")

    def predict(
        self,
        bull_posterior: BranchingPosterior,
        bear_posterior: BranchingPosterior,
        bull_forecast: CriticalStateForecast,
        bear_forecast: CriticalStateForecast,
        *,
        bull_threshold: float,
        bear_threshold: float,
        signal_time: np.datetime64,
        available_at: np.datetime64,
        target_session: np.datetime64,
        training_cutoff: np.datetime64 | None = None,
        rng: np.random.Generator,
    ) -> PercolationPrediction:
        if bull_posterior.symbols != bear_posterior.symbols:
            raise ValueError("directional posterior symbols must align")
        if bull_posterior.direction != "bull" or bear_posterior.direction != "bear":
            raise ValueError("directional posteriors are reversed")
        if not signal_time < available_at:
            raise ValueError("prediction timing must be causal")
        phase_counts = np.zeros(4, dtype=np.int64)
        bull_membership = np.zeros(len(bull_posterior.symbols), dtype=np.float64)
        bear_membership = np.zeros(len(bull_posterior.symbols), dtype=np.float64)
        bull_giant_sum = 0.0
        bear_giant_sum = 0.0
        for _ in range(self.paths):
            bull_giant, bull_seen = self._simulate_direction(
                bull_posterior, bull_forecast, rng
            )
            bear_giant, bear_seen = self._simulate_direction(
                bear_posterior, bear_forecast, rng
            )
            bull_percolates = bull_giant >= bull_threshold
            bear_percolates = bear_giant >= bear_threshold
            if bull_percolates and bear_percolates:
                phase = "conflict"
            elif bull_percolates:
                phase = "bull"
            elif bear_percolates:
                phase = "bear"
            else:
                phase = "none"
            phase_counts[PHASE_NAMES.index(phase)] += 1
            bull_membership += bull_seen
            bear_membership += bear_seen
            bull_giant_sum += bull_giant
            bear_giant_sum += bear_giant
        probabilities = tuple((phase_counts / self.paths).astype(float).tolist())
        effective_cutoff = (
            np.datetime64(training_cutoff)
            if training_cutoff is not None
            else max(bull_posterior.training_cutoff, bear_posterior.training_cutoff)
        )
        return PercolationPrediction(
            phase_probabilities=probabilities,
            bull_reproduction_mean=float(
                np.exp(
                    bull_forecast.log_reproduction_mean
                    + 0.5 * bull_forecast.log_reproduction_variance
                )
            ),
            bear_reproduction_mean=float(
                np.exp(
                    bear_forecast.log_reproduction_mean
                    + 0.5 * bear_forecast.log_reproduction_variance
                )
            ),
            bull_supercritical_probability=bull_forecast.probability_supercritical,
            bear_supercritical_probability=bear_forecast.probability_supercritical,
            bull_giant_fraction_mean=bull_giant_sum / self.paths,
            bear_giant_fraction_mean=bear_giant_sum / self.paths,
            bull_membership_probability=bull_membership / self.paths,
            bear_membership_probability=bear_membership / self.paths,
            signal_time=np.datetime64(signal_time),
            available_at=np.datetime64(available_at),
            target_session=np.datetime64(target_session, "D"),
            training_cutoff=effective_cutoff,
            model_version=self.model_version,
        )

    def _simulate_direction(
        self,
        posterior: BranchingPosterior,
        forecast: CriticalStateForecast,
        rng: np.random.Generator,
    ) -> tuple[float, np.ndarray]:
        count = len(posterior.symbols)
        reproduction = float(
            np.exp(
                rng.normal(
                    forecast.log_reproduction_mean,
                    np.sqrt(max(forecast.log_reproduction_variance, 0.0)),
                )
            )
        )
        shape = self._sample_shape(posterior, rng) * reproduction
        seed_total = int(rng.poisson(max(forecast.external_mean, 1e-9)))
        if seed_total == 0:
            return 0.0, np.zeros(count, dtype=np.float64)
        weights = np.maximum(posterior.external_rate, 0.0)
        weights = (
            weights / weights.sum()
            if weights.sum() > 0.0
            else np.full(count, 1 / count)
        )
        seeds = np.unique(rng.choice(count, size=seed_total, p=weights))
        largest = np.zeros(count, dtype=np.bool_)
        for seed in seeds:
            seen = self._simulate_cascade(shape, int(seed), rng)
            if int(seen.sum()) > int(largest.sum()):
                largest = seen
                if bool(np.all(largest)):
                    break
        return float(largest.mean()), largest.astype(np.float64)

    def _simulate_cascade(
        self, shape: np.ndarray, seed: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Simulate one source lineage so unrelated external rain stays separate."""
        active = np.zeros(shape.shape[0], dtype=np.int64)
        active[seed] = 1
        seen = active > 0
        event_count = 1
        for _ in range(self.maximum_generations):
            if event_count >= self.maximum_events or int(active.sum()) == 0:
                break
            offspring = rng.poisson(np.minimum(active @ shape, self.maximum_events))
            remaining = self.maximum_events - event_count
            if int(offspring.sum()) > remaining:
                scale = remaining / max(int(offspring.sum()), 1)
                offspring = rng.binomial(offspring, scale)
            active = offspring.astype(np.int64)
            seen |= active > 0
            event_count += int(active.sum())
        return seen

    @staticmethod
    def _sample_shape(
        posterior: BranchingPosterior, rng: np.random.Generator
    ) -> np.ndarray:
        mean = posterior.mean
        variance = posterior.variance
        sampled = mean.copy()
        uncertain = posterior.adjacency & (mean > 0.0) & (variance > 0.0)
        if np.any(uncertain):
            shape = np.square(mean[uncertain]) / variance[uncertain]
            scale = variance[uncertain] / mean[uncertain]
            sampled[uncertain] = rng.gamma(shape, scale)
        sampled[~posterior.adjacency] = 0.0
        radius = spectral_radius(sampled)
        if radius > 1e-12:
            sampled /= radius
        else:
            sampled = posterior.normalized_mean.copy()
        return sampled
