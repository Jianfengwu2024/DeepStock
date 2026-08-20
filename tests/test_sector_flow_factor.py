from __future__ import annotations

import unittest

import numpy as np

from bnhmm.sector_flow import (
    BayesianSectorFactor,
    NormalizedSectorPanel,
    SectorFactorConfig,
)


def normalized_panel(velocity: np.ndarray, *, seed: int = 0) -> NormalizedSectorPanel:
    rows, columns = velocity.shape
    rng = np.random.default_rng(seed)
    start = np.datetime64("2025-01-02T09:31")
    timestamps = start + np.arange(rows).astype("timedelta64[m]")
    valid = np.isfinite(velocity)
    activity = np.clip(1.0 + rng.normal(0.0, 0.1, size=velocity.shape), 0.1, 2.0)
    return NormalizedSectorPanel(
        symbols=tuple(f"S{index:02d}" for index in range(columns)),
        timestamps=timestamps.astype("datetime64[ns]"),
        raw_velocity=velocity.copy(),
        velocity=velocity.copy(),
        activity=activity,
        velocity_scale=np.ones_like(velocity),
        volume_baseline=np.ones_like(velocity),
        valid=valid,
        activity_valid=np.ones_like(valid),
    )


class BayesianSectorFactorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SectorFactorConfig(
            persistence=None,
            process_variance=None,
            observation_variance_floor=0.02,
            robust_degrees_of_freedom=5.0,
            minimum_symbols=3,
        )

    def test_recovers_a_known_common_sector_factor(self) -> None:
        rng = np.random.default_rng(31)
        rows, columns = 600, 12
        factor = np.zeros(rows)
        for index in range(1, rows):
            factor[index] = 0.92 * factor[index - 1] + rng.normal(0.0, 0.35)
        loadings = np.linspace(0.7, 1.3, columns)
        velocity = factor[:, None] * loadings + rng.normal(0.0, 0.18, (rows, columns))
        panel = normalized_panel(velocity)

        model = BayesianSectorFactor(self.config)
        parameters = model.fit(panel, np.arange(400))
        posterior = model.filter(panel, parameters)

        correlation = np.corrcoef(factor[400:], posterior.factor_mean[400:])[0, 1]
        self.assertGreater(correlation, 0.9)
        self.assertGreater(float(np.nanmedian(posterior.coherence[400:])), 0.7)

    def test_leave_one_out_flow_is_invariant_to_the_excluded_stock(self) -> None:
        rng = np.random.default_rng(7)
        velocity = rng.normal(0.0, 1.0, (240, 8))
        panel = normalized_panel(velocity)
        model = BayesianSectorFactor(self.config)
        parameters = model.fit(panel, np.arange(120))
        original = model.filter(panel, parameters)
        changed_velocity = velocity.copy()
        changed_velocity[:, 3] += np.linspace(-20.0, 20.0, len(velocity))
        changed = model.filter(normalized_panel(changed_velocity), parameters)

        np.testing.assert_allclose(
            original.leave_one_out_mean[:, 3],
            changed.leave_one_out_mean[:, 3],
            rtol=0.0,
            atol=1e-12,
        )

    def test_filter_is_causal_and_handles_all_invalid_minutes(self) -> None:
        rng = np.random.default_rng(9)
        velocity = rng.normal(0.0, 1.0, (120, 6))
        panel = normalized_panel(velocity)
        model = BayesianSectorFactor(self.config)
        parameters = model.fit(panel, np.arange(60))
        original = model.filter(panel, parameters)
        modified = velocity.copy()
        modified[80:] *= -10.0
        modified[70] = np.nan
        changed = model.filter(normalized_panel(modified), parameters)

        np.testing.assert_allclose(original.factor_mean[:70], changed.factor_mean[:70])
        self.assertFalse(changed.valid[70])
        self.assertTrue(np.isfinite(changed.factor_mean[70]))


if __name__ == "__main__":
    unittest.main()
