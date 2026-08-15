from __future__ import annotations

import unittest

import numpy as np

from bnhmm.fk_stock import (
    BayesianGeneratorPosterior,
    FKFeatureRow,
    FeynmanKacConfig,
    FeynmanKacSolver,
)


def row(*, stochastic_fraction: float = 0.0, coherence: float = 1.0) -> FKFeatureRow:
    return FKFeatureRow(
        symbol="000001",
        signal_time=np.datetime64("2024-01-02T15:00"),
        execution_time=np.datetime64("2024-01-03T09:30"),
        target_time=np.datetime64("2024-01-03T15:00"),
        target=3,
        forward_return=0.02,
        standardized_return=1.2,
        return_location=0.0,
        return_scale=0.01,
        features=np.zeros(9, dtype=np.float64),
        stochastic_current_fraction=stochastic_fraction,
        current_coherence=coherence,
        continuity_error=0.0,
        matured_return_count=10,
    )


def posterior(drift: float) -> BayesianGeneratorPosterior:
    mean = np.zeros(10, dtype=np.float64)
    mean[0] = drift
    return BayesianGeneratorPosterior(
        coefficient_mean=mean,
        coefficient_covariance=np.eye(10, dtype=np.float64) * 1e-5,
        feature_location=np.zeros(9, dtype=np.float64),
        feature_scale=np.ones(9, dtype=np.float64),
        noise_variance=0.15,
        class_prior=(0.25, 0.25, 0.25, 0.25),
        fit_rows=100,
    )


class FeynmanKacSolverTests(unittest.TestCase):
    def test_drift_sign_controls_direction_and_probabilities_sum_to_one(self) -> None:
        solver = FeynmanKacSolver(FeynmanKacConfig(mc_samples=128))
        bullish, _ = solver.predict(row(), posterior(1.0))
        bearish, _ = solver.predict(row(), posterior(-1.0))

        self.assertAlmostEqual(sum(bullish.probability_mean), 1.0, places=10)
        self.assertGreater(sum(bullish.probability_mean[2:]), 0.5)
        self.assertGreater(sum(bearish.probability_mean[:2]), 0.5)
        self.assertGreater(bullish.expected_score, 0.0)
        self.assertLess(bearish.expected_score, 0.0)

    def test_effective_potential_shrinks_directional_drift(self) -> None:
        solver = FeynmanKacSolver(
            FeynmanKacConfig(
                noise_potential_weight=2.0,
                incoherence_potential_weight=1.0,
                mc_samples=64,
            )
        )
        clean, clean_diagnostics = solver.predict(row(), posterior(1.0))
        noisy, noisy_diagnostics = solver.predict(
            row(stochastic_fraction=20.0, coherence=0.0), posterior(1.0)
        )

        self.assertGreater(
            noisy_diagnostics["effective_potential"],
            clean_diagnostics["effective_potential"],
        )
        self.assertLess(abs(noisy.expected_standardized_return), abs(clean.expected_standardized_return))
        self.assertLess(abs(noisy.expected_score), abs(clean.expected_score))


if __name__ == "__main__":
    unittest.main()
