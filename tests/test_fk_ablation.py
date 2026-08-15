from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from bnhmm import LabelConfig, PriceVolumeProfileBuilder
from bnhmm.data.dataset import ProfileLabelBuilder
from bnhmm.fk_stock import (
    FKGeneratorConfig,
    FeynmanKacConfig,
    FeynmanKacPotentialConfig,
    run_fk_stock_ablation,
)
from bnhmm.stock_transformer import ConservationConfig
from bnhmm.walk_forward import WalkForwardConfig
from test_profile import synthetic_sessions


class FKAblationTests(unittest.TestCase):
    def test_nested_ablation_selects_before_outer_evaluation(self) -> None:
        states = PriceVolumeProfileBuilder().build(synthetic_sessions(34))
        first = ProfileLabelBuilder(
            LabelConfig(standardization_min_observations=2)
        ).build(states)
        second = replace(first, symbol="000002")
        with TemporaryDirectory() as directory:
            result = run_fk_stock_ablation(
                (first, second),
                ConservationConfig(grid_points=24),
                FKGeneratorConfig(
                    drift_lookback_sessions=5,
                    minimum_return_history=2,
                    minimum_fit_rows=6,
                ),
                FeynmanKacConfig(),
                FeynmanKacPotentialConfig(mc_samples=2, maximum_potential=4.0),
                WalkForwardConfig(
                    minimum_train_sessions=12,
                    tuning_sessions=4,
                    test_sessions=3,
                    step_sessions=3,
                    max_folds=1,
                ),
                output_dir=directory,
                dataset_id="test-sector",
            )
            payload = json.loads(
                Path(directory, "ablation.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["selected_features"], list(result.selected_features))
        self.assertIn(
            result.selected_potential,
            {"no-potential", "epistemic-only", "flow-only", "full-v1"},
        )
        self.assertGreater(len(result.tuning_feature_results), 10)
        self.assertGreater(len(result.selected_predictions), 0)


if __name__ == "__main__":
    unittest.main()
