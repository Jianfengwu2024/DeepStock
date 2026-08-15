from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from bnhmm import LabelConfig, PriceVolumeProfileBuilder
from bnhmm.data.dataset import ProfileLabelBuilder
from bnhmm.fk_stock import FKGenerator, FKGeneratorConfig
from bnhmm.fk_stock.generator import FEATURE_NAMES
from bnhmm.stock_transformer import (
    ConservationConfig,
    ConservationLawDiscoverer,
    DensityProjector,
)
from test_profile import synthetic_sessions


class FKGeneratorTests(unittest.TestCase):
    def test_rows_use_only_same_symbol_and_matured_returns(self) -> None:
        states = PriceVolumeProfileBuilder().build(synthetic_sessions(28))
        observations = ProfileLabelBuilder(
            LabelConfig(standardization_min_observations=2)
        ).build(states)
        second = replace(observations, symbol="000002")
        conservation = ConservationConfig(grid_points=24)
        projector = DensityProjector(conservation)
        series = []
        for item in (observations, second):
            series.append(
                [
                    projector.project(
                        item.symbol,
                        item.signal_times[index],
                        item.profiles[index][:, 3],
                        item.profiles[index][:, 1],
                    )
                    for index in range(len(item))
                ]
            )
        law = ConservationLawDiscoverer(conservation).fit(series)
        generator = FKGenerator(
            FKGeneratorConfig(
                drift_lookback_sessions=5,
                minimum_return_history=2,
                minimum_fit_rows=4,
            )
        )
        rows = generator.build_rows((observations, second))

        self.assertGreater(len(rows), 4)
        self.assertEqual({row.symbol for row in rows}, {"000001", "000002"})
        for row in rows:
            source = observations if row.symbol == "000001" else second
            matured = source.target_times < row.signal_time
            self.assertEqual(
                row.matured_return_count,
                min(int(matured.sum()), generator.config.drift_lookback_sessions),
            )
            self.assertTrue(np.all(source.target_times[matured] < row.signal_time))

        posterior = generator.fit(rows)
        self.assertEqual(posterior.fit_rows, len(rows))
        self.assertEqual(posterior.feature_names, ("rolling_diffusion",))
        self.assertEqual(posterior.coefficient_mean.shape, (2,))
        self.assertAlmostEqual(sum(posterior.class_prior), 1.0)

        full_generator = FKGenerator(
            replace(
                generator.config,
                active_features=FEATURE_NAMES,
            )
        )
        full_rows = full_generator.build_rows(
            (observations, second), law, conservation
        )
        full = full_generator.fit(full_rows)
        self.assertEqual(full.feature_names, FEATURE_NAMES)
        self.assertEqual(full.coefficient_mean.shape, (10,))
        self.assertEqual(full.design_vector(full_rows[0]).shape, (10,))


if __name__ == "__main__":
    unittest.main()
