from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from bnhmm import LabelConfig, PriceVolumeProfileBuilder
from bnhmm.data.dataset import ProfileLabelBuilder
from bnhmm.fk_stock import (
    FKGeneratorConfig,
    FKStockPipelineConfig,
    FeynmanKacConfig,
    run_fk_stock_walk_forward,
)
from bnhmm.fk_stock.pipeline import _fixed_training_moments
from bnhmm.stock_transformer import ConservationConfig
from bnhmm.walk_forward import WalkForwardConfig
from test_profile import synthetic_sessions


class FKPipelineTests(unittest.TestCase):
    def test_new_constituent_uses_training_only_sector_moments(self) -> None:
        states = PriceVolumeProfileBuilder().build(synthetic_sessions(24))
        established = ProfileLabelBuilder(
            LabelConfig(standardization_min_observations=2)
        ).build(states)
        newcomer = replace(
            established.take(np.arange(len(established) - 3, len(established))),
            symbol="000099",
        )
        cutoff = newcomer.signal_times[0]
        means, variances = _fixed_training_moments(
            (established, newcomer), cutoff
        )

        self.assertIn("000099", means)
        self.assertIn("000099", variances)
        self.assertEqual(len(means["000099"]), 4)

    def test_walk_forward_persists_generator_and_field_diagnostics(self) -> None:
        states = PriceVolumeProfileBuilder().build(synthetic_sessions(38))
        first = ProfileLabelBuilder(
            LabelConfig(standardization_min_observations=2)
        ).build(states)
        second = replace(first, symbol="000002")
        with TemporaryDirectory() as directory:
            result = run_fk_stock_walk_forward(
                (first, second),
                ConservationConfig(grid_points=24),
                FKGeneratorConfig(
                    drift_lookback_sessions=5,
                    minimum_return_history=2,
                    minimum_fit_rows=8,
                ),
                FeynmanKacConfig(),
                FKStockPipelineConfig(),
                WalkForwardConfig(
                    minimum_train_sessions=12,
                    tuning_sessions=4,
                    test_sessions=4,
                    step_sessions=4,
                    max_folds=1,
                ),
                output_dir=directory,
                dataset_id="test-sector",
            )
            model = json.loads(
                Path(directory, "fold-00", "model.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(result.folds), 1)
        self.assertGreater(len(result.predictions), 0)
        self.assertEqual(model["strategy"], "fk-stock")
        self.assertEqual(model["generator_posterior"]["feature_names"], ["rolling_diffusion"])
        self.assertIsNone(model["conservation_law"])
        self.assertIn("generator_posterior", model)
        self.assertIn("outer_field_diagnostics", result.folds[0])
        self.assertTrue(
            all(
                prediction.signal_time < prediction.target_time
                for prediction in result.predictions
            )
        )


if __name__ == "__main__":
    unittest.main()
