from __future__ import annotations

import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory

from bnhmm import LabelConfig, PriceVolumeProfileBuilder
from bnhmm.data.dataset import ProfileLabelBuilder
from bnhmm.stock_transformer import (
    ConservationConfig,
    StockPipelineConfig,
    StockTrainerConfig,
    StockTransformerConfig,
    TokenizationConfig,
    run_stock_transformer_walk_forward,
)
from bnhmm.walk_forward import WalkForwardConfig
from test_profile import synthetic_sessions


class StockPipelineTests(unittest.TestCase):
    def test_small_sector_walk_forward_keeps_three_layer_artifacts(self) -> None:
        states = PriceVolumeProfileBuilder().build(synthetic_sessions(34))
        first = ProfileLabelBuilder(
            LabelConfig(standardization_min_observations=2)
        ).build(states)
        second = replace(first, symbol="000002")
        with TemporaryDirectory() as directory:
            result = run_stock_transformer_walk_forward(
                [first, second],
                ConservationConfig(grid_points=24),
                TokenizationConfig(maximum_tokens=6, boundary_threshold=0.8),
                StockTransformerConfig(
                    d_model=16,
                    heads=4,
                    layers=1,
                    feedforward_dim=32,
                    maximum_days=3,
                    maximum_tokens=6,
                ),
                StockTrainerConfig(epochs=1, batch_size=8),
                StockPipelineConfig(sequence_length=3, mc_samples=2),
                WalkForwardConfig(
                    minimum_train_sessions=10,
                    tuning_sessions=3,
                    test_sessions=3,
                    step_sessions=3,
                    max_folds=1,
                ),
                device="cpu",
                output_dir=directory,
                dataset_id="test-sector",
            )

        self.assertEqual(len(result.folds), 1)
        self.assertGreater(len(result.predictions), 0)
        self.assertIn("conservation_law", result.folds[0])
        self.assertIn("token_diagnostics", result.folds[0])
        self.assertEqual(result.folds[0]["strategy"], "stock-transformer")


if __name__ == "__main__":
    unittest.main()
