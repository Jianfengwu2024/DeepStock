from __future__ import annotations

import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bnhmm import LabelConfig, PriceVolumeProfileBuilder, SequenceConfig
from bnhmm.ablation import BNHMMAblationConfig, run_bnhmm_ablation
from bnhmm.data.dataset import ProfileLabelBuilder
from bnhmm.training import ModelConfig, TrainerConfig
from bnhmm.walk_forward import WalkForwardConfig
from test_profile import synthetic_sessions


class BNHMMAblationTests(unittest.TestCase):
    def test_outer_test_is_called_once_after_candidate_selection(self) -> None:
        states = PriceVolumeProfileBuilder().build(synthetic_sessions(34))
        first = ProfileLabelBuilder(
            LabelConfig(standardization_min_observations=2)
        ).build(states)
        second = replace(first, symbol="000002")

        def fake_candidate(candidate, seeds, *args, **kwargs):
            del args, kwargs
            penalty = 0.0 if "price-only" in candidate.name else 0.01
            passes_gate = "price-only" in candidate.name
            seed_results = [
                {
                    "seed": seed,
                    "best_epoch": 1,
                    "log_loss": 1.0 + penalty,
                    "brier_score": 0.7 + penalty,
                    "accuracy": 0.3,
                    "balanced_accuracy": 0.3,
                }
                for seed in seeds
            ]
            return {
                "name": candidate.name,
                "stage": candidate.stage,
                "active_parameter_count": 100,
                "seed_results": seed_results,
                "mean_log_loss": 1.0 + penalty,
                "mean_brier_score": 0.7 + penalty,
                "mean_accuracy": 0.3,
                "mean_balanced_accuracy": 0.3,
                "mean_prior_log_loss": 1.1,
                "mean_prior_brier_score": 0.75,
                "mean_majority_baseline": 0.25,
                "mean_random_balanced_baseline": 0.25,
                "passes_tuning_gate": passes_gate,
            }

        outer_metrics = {
            "observations": 10,
            "accuracy": 0.3,
            "balanced_accuracy": 0.3,
            "log_loss": 1.0,
            "brier_score": 0.7,
            "prior_log_loss": 1.1,
            "prior_brier_score": 0.75,
        }
        with TemporaryDirectory() as directory, patch(
            "bnhmm.ablation._evaluate_candidate", side_effect=fake_candidate
        ), patch(
            "bnhmm.ablation._evaluate_outer", return_value=([], outer_metrics)
        ) as outer:
            result = run_bnhmm_ablation(
                (first, second),
                ModelConfig(),
                TrainerConfig(epochs=1, pretrain_epochs=0),
                SequenceConfig(length=2),
                WalkForwardConfig(
                    minimum_train_sessions=12,
                    tuning_sessions=4,
                    test_sessions=3,
                    step_sessions=3,
                    max_folds=1,
                ),
                device="cpu",
                output_dir=directory,
                dataset_id="test-sector",
                ablation_config=BNHMMAblationConfig(
                    confirmation_seeds=(7, 17, 29)
                ),
            )

        outer.assert_called_once()
        self.assertFalse(result.fold["selection_used_outer_test"])
        self.assertEqual(outer.call_args.args[0].name, result.selected_candidate.name)
        self.assertTrue(result.promotion_recommended)

    def test_noninferior_reduction_requires_two_of_three_seeds(self) -> None:
        from bnhmm.ablation import _noninferior

        baseline_rows = [
            {
                "seed": seed,
                "log_loss": 1.0,
                "brier_score": 0.7,
                "accuracy": 0.3,
                "balanced_accuracy": 0.3,
            }
            for seed in (7, 17, 29)
        ]
        reduced_rows = [
            {
                "seed": 7,
                "log_loss": 1.0,
                "brier_score": 0.7,
                "accuracy": 0.3,
                "balanced_accuracy": 0.3,
            },
            {
                "seed": 17,
                "log_loss": 1.0,
                "brier_score": 0.7,
                "accuracy": 0.3,
                "balanced_accuracy": 0.3,
            },
            {
                "seed": 29,
                "log_loss": 1.1,
                "brier_score": 0.8,
                "accuracy": 0.2,
                "balanced_accuracy": 0.2,
            },
        ]
        baseline = {
            "mean_log_loss": 1.0,
            "mean_brier_score": 0.7,
            "mean_accuracy": 0.3,
            "mean_balanced_accuracy": 0.3,
            "seed_results": baseline_rows,
        }
        reduced = {
            "mean_log_loss": 1.0,
            "mean_brier_score": 0.7,
            "mean_accuracy": 0.3,
            "mean_balanced_accuracy": 0.3,
            "seed_results": reduced_rows,
        }

        self.assertTrue(_noninferior(reduced, baseline, BNHMMAblationConfig()))

    def test_promotion_gate_rejects_below_majority_accuracy(self) -> None:
        from bnhmm.ablation import _promotion_reasons

        reasons = _promotion_reasons(
            {
                "mean_accuracy": 0.31,
                "mean_majority_baseline": 0.33,
                "mean_balanced_accuracy": 0.27,
                "mean_random_balanced_baseline": 0.25,
                "mean_log_loss": 1.30,
                "mean_prior_log_loss": 1.35,
                "mean_brier_score": 0.70,
                "mean_prior_brier_score": 0.73,
            }
        )

        self.assertIn("accuracy did not beat the tuning majority baseline", reasons)


if __name__ == "__main__":
    unittest.main()
