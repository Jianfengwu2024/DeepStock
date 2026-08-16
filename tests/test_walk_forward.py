from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from bnhmm import LabelConfig, PriceVolumeProfileBuilder, SequenceConfig
from bnhmm.data.dataset import ProfileLabelBuilder
from bnhmm.training import ModelConfig, TrainerConfig
from bnhmm.walk_forward import (
    WalkForwardConfig,
    _state_diagnostics,
    build_walk_forward_folds,
    run_walk_forward,
)
from test_profile import synthetic_sessions
from test_training import sequence


class WalkForwardTests(unittest.TestCase):
    def test_state_diagnostics_aggregate_local_price_region_flow(self) -> None:
        model = ModelConfig(
            max_profile_bins=32,
            d_model=16,
            motif_dim=4,
            heads=4,
            layers=1,
            feedforward_dim=32,
            max_sequence_length=8,
        ).build(seed=7)
        item = sequence(4, length=4, bins=8)
        prediction = SimpleNamespace(
            probability_mean=(0.25, 0.25, 0.25, 0.25),
            prior_probability=(0.25, 0.25, 0.25, 0.25),
        )

        diagnostics = _state_diagnostics(
            model, [item], [prediction], device="cpu"
        )

        self.assertGreater(diagnostics["observed_local_flow_bins"], 0)
        self.assertGreaterEqual(diagnostics["mean_z2_flow_order"], 0.0)
        self.assertLessEqual(diagnostics["mean_z2_flow_order"], 1.0)
        self.assertGreaterEqual(diagnostics["active_laminar_ratio"], 0.0)
        self.assertLessEqual(diagnostics["active_laminar_ratio"], 1.0)

    def test_folds_keep_inner_tuning_before_outer_test(self) -> None:
        states = PriceVolumeProfileBuilder().build(synthetic_sessions(33))
        observations = ProfileLabelBuilder(
            LabelConfig(standardization_min_observations=2)
        ).build(states)
        config = WalkForwardConfig(
            minimum_train_sessions=10,
            tuning_sessions=3,
            test_sessions=4,
            step_sessions=4,
            max_folds=2,
        )

        folds = build_walk_forward_folds([observations], config)

        self.assertEqual(len(folds), 2)
        for fold in folds:
            self.assertLess(fold.tuning_start, fold.test_start)
            self.assertLess(fold.test_start, fold.test_end)
        self.assertLessEqual(folds[0].test_end, folds[1].test_start)

    def test_multi_symbol_run_builds_one_model_pipeline_per_symbol(self) -> None:
        states = PriceVolumeProfileBuilder().build(synthetic_sessions(33))
        first = ProfileLabelBuilder(
            LabelConfig(standardization_min_observations=2)
        ).build(states)
        second = replace(first, symbol="000002")
        destinations: list[Path] = []

        def fake_symbol_run(observation, *args, output_dir, **kwargs):
            destinations.append(Path(output_dir))
            return SimpleNamespace(
                folds=({"symbol": observation.symbol, "fold": 0},),
                predictions=(),
                regime_return_means=(-0.03, -0.005, 0.005, 0.03),
                regime_return_variances=(0.0001,) * 4,
            )

        with TemporaryDirectory() as directory, patch(
            "bnhmm.walk_forward._run_symbol_walk_forward",
            side_effect=fake_symbol_run,
        ) as symbol_run:
            result = run_walk_forward(
                [first, second],
                ModelConfig(),
                TrainerConfig(),
                SequenceConfig(),
                WalkForwardConfig(),
                device="cpu",
                output_dir=directory,
            )

        self.assertEqual(symbol_run.call_count, 2)
        self.assertEqual(
            [call.args[0].symbol for call in symbol_run.call_args_list],
            [first.symbol, second.symbol],
        )
        self.assertEqual(set(result.regime_return_means), {first.symbol, second.symbol})
        self.assertEqual(
            destinations,
            [
                Path(directory, "symbols", first.symbol),
                Path(directory, "symbols", second.symbol),
            ],
        )

    def test_sector_scope_builds_one_shared_pipeline(self) -> None:
        states = PriceVolumeProfileBuilder().build(synthetic_sessions(33))
        first = ProfileLabelBuilder(
            LabelConfig(standardization_min_observations=2)
        ).build(states)
        second = replace(first, symbol="000002")
        expected = SimpleNamespace(
            folds=({"model_scope": "sector-pooled", "fold": 0},),
            predictions=(),
            regime_return_means={first.symbol: (-0.03, -0.005, 0.005, 0.03)},
            regime_return_variances={first.symbol: (0.0001,) * 4},
        )

        with TemporaryDirectory() as directory, patch(
            "bnhmm.walk_forward._run_pooled_walk_forward",
            return_value=expected,
        ) as pooled_run, patch(
            "bnhmm.walk_forward._run_symbol_walk_forward"
        ) as symbol_run:
            result = run_walk_forward(
                [first, second],
                ModelConfig(),
                TrainerConfig(),
                SequenceConfig(),
                WalkForwardConfig(),
                device="cpu",
                output_dir=directory,
                training_scope="sector-pooled",
                dataset_id="semiconductor",
            )

        self.assertIs(result, expected)
        pooled_run.assert_called_once()
        self.assertEqual(
            pooled_run.call_args.kwargs["dataset_id"], "semiconductor"
        )
        symbol_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
