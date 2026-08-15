from __future__ import annotations

import unittest

import numpy as np

from bnhmm.stock_transformer.conservation import (
    ConservationConfig,
    ConservationLawDiscoverer,
    DensityFrame,
    DensityProjector,
)


class StockConservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ConservationConfig(grid_points=32, ridge=1e-4)
        self.discoverer = ConservationLawDiscoverer(self.config)

    def frame(self, day: int, center: float) -> DensityFrame:
        grid = self.discoverer.grid
        density = np.exp(-0.5 * ((grid - center) / 0.2) ** 2)
        density /= density.sum()
        return DensityFrame("000001", np.datetime64(f"2024-01-{day:02d}"), density)

    def test_discovery_preserves_mass_and_mean_zero_noise(self) -> None:
        frames = [self.frame(day, -0.25 + day * 0.03) for day in range(1, 9)]
        law = self.discoverer.fit([frames])
        transitions = self.discoverer.transform_series(frames, law)

        self.assertEqual(law.transition_count, 7)
        self.assertLess(law.continuity_rmse, 1e-12)
        self.assertTrue(all(item.mass_error < 1e-12 for item in transitions))
        residual = np.concatenate([item.stochastic_current for item in transitions])
        self.assertLess(abs(float(residual.mean())), law.noise_scale * 0.05)

    def test_projector_maps_variable_bins_to_unit_density(self) -> None:
        projector = DensityProjector(self.config)
        frame = projector.project(
            "000001",
            np.datetime64("2024-01-02"),
            np.asarray([-0.8, -0.2, 0.1, 0.7]),
            np.asarray([0.1, 0.4, 0.3, 0.2]),
        )
        self.assertEqual(frame.density.shape, (32,))
        self.assertAlmostEqual(float(frame.density.sum()), 1.0)

    def test_rejects_cross_symbol_transition(self) -> None:
        first = self.frame(1, -0.1)
        second = DensityFrame("000002", np.datetime64("2024-01-02"), first.density)
        with self.assertRaisesRegex(ValueError, "cross symbols"):
            self.discoverer.fit([[first, second]])


if __name__ == "__main__":
    unittest.main()
