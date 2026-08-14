from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date

import numpy as np

from bnhmm import REGIME_SCORES
from bnhmm.evaluation import (
    DecisionPolicy,
    DecisionPolicyConfig,
    PredictionRecord,
    _format_bimonthly_date_axis,
    evaluate_predictions,
)


def prediction(day: int, target: int, probability: tuple[float, ...]) -> PredictionRecord:
    signal = np.datetime64("2026-01-01") + np.timedelta64(day, "D")
    scores = np.asarray(REGIME_SCORES)
    returns = (-0.03, -0.005, 0.005, 0.03)
    return PredictionRecord(
        symbol="000001",
        timeframe_id=1,
        signal_time=signal.astype("datetime64[ns]"),
        execution_time=(signal + np.timedelta64(1, "D")).astype("datetime64[ns]"),
        target_time=(signal + np.timedelta64(1, "D") + np.timedelta64(6, "h")).astype(
            "datetime64[ns]"
        ),
        target=target,
        forward_return=returns[target],
        standardized_return=returns[target] / 0.02,
        return_location=0.0,
        return_scale=0.02,
        prior_probability=(0.25,) * 4,
        probability_mean=probability,
        aleatoric_variance=(0.01,) * 4,
        epistemic_variance=(0.001,) * 4,
        expected_score=float(np.asarray(probability) @ scores),
        expected_return=returns[target],
        expected_standardized_return=returns[target] / 0.02,
        return_variance=0.0001,
        aleatoric_return_variance=0.00009,
        epistemic_return_variance=0.00001,
        context_length=8,
    )


class EvaluationTests(unittest.TestCase):
    def test_report_contains_probabilistic_and_strategy_diagnostics(self) -> None:
        predictions = [
            prediction(index, index % 4, tuple(np.eye(4)[index % 4] * 0.7 + 0.075))
            for index in range(16)
        ]
        report = evaluate_predictions(
            predictions,
            np.asarray([-0.03, -0.005, 0.005, 0.03]),
            np.full(4, 0.0001),
        )

        self.assertLess(report["log_loss"], 1.0)
        self.assertEqual(len(report["residual_autocorrelation"]), 20)
        self.assertIn("max_drawdown", report["backtest"])
        self.assertIn("average_abs_position", report["backtest"])
        self.assertGreaterEqual(report["backtest"]["active_position_ratio"], 0.0)
        self.assertEqual(report["observation_count"], 16)
        self.assertEqual(report["balanced_accuracy"], 1.0)
        self.assertEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["class_support"], [4, 4, 4, 4])
        self.assertEqual(report["confusion_matrix"], (np.eye(4, dtype=int) * 4).tolist())
        self.assertLess(report["ranked_probability_score"], 0.1)
        self.assertEqual(report["baselines"]["comparison_count"], 15)
        self.assertAlmostEqual(report["baselines"]["persistence_accuracy"], 0.0)
        self.assertAlmostEqual(report["baselines"]["reversal_accuracy"], 7 / 15)
        self.assertAlmostEqual(report["baselines"]["class_prior_brier"], 0.75)
        self.assertGreater(report["baselines"]["log_information_gain"], 0.0)
        self.assertGreater(report["baselines"]["brier_skill"], 0.0)
        self.assertIn(
            "beats_dynamic_prior_return_rmse", report["validation_gate"]["checks"]
        )
        self.assertFalse(report["validation_gate"]["passed"])

    def test_uncertainty_reduces_position(self) -> None:
        item = prediction(1, 3, (0.02, 0.03, 0.05, 0.9))
        cautious = DecisionPolicy(DecisionPolicyConfig(uncertainty_aversion=4.0))
        relaxed = DecisionPolicy(DecisionPolicyConfig(uncertainty_aversion=0.0))
        self.assertLess(cautious.target_position(item), relaxed.target_position(item))

    def test_aleatoric_class_overlap_does_not_suppress_bullish_position(self) -> None:
        item = replace(
            prediction(1, 3, (0.02, 0.03, 0.05, 0.9)),
            aleatoric_variance=(0.2,) * 4,
            epistemic_variance=(0.001,) * 4,
        )
        policy = DecisionPolicy(DecisionPolicyConfig(uncertainty_aversion=2.0))
        self.assertGreater(policy.target_position(item), 0.8)

    def test_three_state_predictions_are_rejected(self) -> None:
        item = prediction(1, 3, (0.02, 0.03, 0.05, 0.9))
        legacy = replace(
            item,
            probability_mean=(0.1, 0.2, 0.7),
            aleatoric_variance=(0.01,) * 3,
            epistemic_variance=(0.001,) * 3,
        )
        with self.assertRaisesRegex(ValueError, "four-state"):
            evaluate_predictions([legacy], np.zeros(4), np.ones(4))

    def test_long_only_policy_opens_holds_and_closes_by_state(self) -> None:
        policy = DecisionPolicy(DecisionPolicyConfig(uncertainty_aversion=0.0))
        weak_bull = prediction(1, 2, (0.02, 0.03, 0.9, 0.05))
        strong_bull = prediction(2, 3, (0.02, 0.03, 0.05, 0.9))
        weak_bear = prediction(3, 1, (0.03, 0.9, 0.05, 0.02))

        opened = policy.target_position(weak_bull, 0.0)
        increased = policy.target_position(strong_bull, opened)
        closed = policy.target_position(weak_bear, increased)

        self.assertEqual(opened, 0.5)
        self.assertEqual(increased, 1.0)
        self.assertEqual(closed, 0.0)

        uncertain_bull = replace(
            strong_bull,
            aleatoric_variance=(0.08,) * 4,
            epistemic_variance=(0.02,) * 4,
        )
        cautious = DecisionPolicy(DecisionPolicyConfig(uncertainty_aversion=4.0))
        self.assertEqual(cautious.target_position(uncertain_bull, 0.7), 0.7)

    def test_continuous_posterior_return_is_used_directly(self) -> None:
        first = replace(
            prediction(1, 3, (0.0, 0.0, 0.0, 1.0)), symbol="000001"
        )
        second = replace(first, symbol="000002")
        report = evaluate_predictions(
            [first, second],
            {
                "000001": (-0.03, -0.005, 0.005, 0.03),
                "000002": (-0.04, -0.01, 0.01, 0.05),
            },
            {"000001": (0.0001,) * 4, "000002": (0.0001,) * 4},
        )

        np.testing.assert_allclose(report["series"]["return_residuals"], [0.0, 0.0])

    def test_report_chart_uses_bimonthly_date_labels(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
        except ImportError:
            self.skipTest("matplotlib is an optional dependency")

        figure, axis = plt.subplots()
        axis.plot([date(2025, 1, 1), date(2025, 12, 31)], [1.0, 1.1])
        _format_bimonthly_date_axis(axis)
        figure.canvas.draw()
        labels = [label.get_text() for label in axis.get_xticklabels() if label.get_text()]
        plt.close(figure)

        self.assertGreaterEqual(len(labels), 5)
        self.assertLessEqual(len(labels), 7)
        self.assertTrue(all(len(label) == 7 and label[4] == "-" for label in labels))
        month_indexes = [int(label[:4]) * 12 + int(label[5:]) for label in labels]
        self.assertTrue(
            all(later - earlier == 2 for earlier, later in zip(month_indexes, month_indexes[1:]))
        )


if __name__ == "__main__":
    unittest.main()
