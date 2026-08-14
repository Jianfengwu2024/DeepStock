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
    return PredictionRecord(
        symbol="000001",
        timeframe_id=1,
        signal_time=signal.astype("datetime64[ns]"),
        execution_time=(signal + np.timedelta64(1, "D")).astype("datetime64[ns]"),
        target_time=(signal + np.timedelta64(1, "D") + np.timedelta64(6, "h")).astype(
            "datetime64[ns]"
        ),
        target=target,
        forward_return=(-0.01, 0.0, 0.01)[target],
        probability_mean=probability,
        aleatoric_variance=(0.01,) * 3,
        epistemic_variance=(0.001,) * 3,
        expected_score=float(np.asarray(probability) @ scores),
        expected_return=(-0.01, 0.0, 0.01)[target],
        return_variance=0.0001,
        aleatoric_return_variance=0.00009,
        epistemic_return_variance=0.00001,
        context_length=8,
    )


class EvaluationTests(unittest.TestCase):
    def test_report_contains_probabilistic_and_strategy_diagnostics(self) -> None:
        predictions = [
            prediction(index, index % 3, tuple(np.eye(3)[index % 3] * 0.7 + 0.1))
            for index in range(12)
        ]
        report = evaluate_predictions(
            predictions,
            np.asarray([-0.01, 0.0, 0.01]),
            np.full(3, 0.0001),
        )

        self.assertLess(report["log_loss"], 1.0)
        self.assertEqual(len(report["residual_autocorrelation"]), 20)
        self.assertIn("max_drawdown", report["backtest"])
        self.assertIn("average_abs_position", report["backtest"])
        self.assertGreaterEqual(report["backtest"]["active_position_ratio"], 0.0)
        self.assertEqual(report["observation_count"], 12)
        self.assertEqual(report["balanced_accuracy"], 1.0)
        self.assertEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["class_support"], [4, 4, 4])
        self.assertEqual(report["confusion_matrix"], (np.eye(3, dtype=int) * 4).tolist())
        self.assertLess(report["ranked_probability_score"], 0.1)
        self.assertEqual(report["baselines"]["comparison_count"], 11)
        self.assertAlmostEqual(report["baselines"]["persistence_accuracy"], 0.0)
        self.assertAlmostEqual(report["baselines"]["reversal_accuracy"], 3 / 11)
        self.assertIn("beats_zero_return_rmse", report["validation_gate"]["checks"])
        self.assertFalse(report["validation_gate"]["passed"])

    def test_uncertainty_reduces_position(self) -> None:
        item = prediction(1, 2, (0.05, 0.1, 0.85))
        cautious = DecisionPolicy(DecisionPolicyConfig(uncertainty_aversion=4.0))
        relaxed = DecisionPolicy(DecisionPolicyConfig(uncertainty_aversion=0.0))
        self.assertLess(cautious.target_position(item), relaxed.target_position(item))

    def test_four_state_predictions_are_rejected(self) -> None:
        item = prediction(1, 2, (0.05, 0.1, 0.85))
        legacy = replace(
            item,
            probability_mean=(0.1, 0.1, 0.7, 0.1),
            aleatoric_variance=(0.01,) * 4,
            epistemic_variance=(0.001,) * 4,
        )
        with self.assertRaisesRegex(ValueError, "three-state"):
            evaluate_predictions([legacy], np.zeros(3), np.ones(3))

    def test_long_only_policy_opens_holds_and_closes_by_state(self) -> None:
        policy = DecisionPolicy(DecisionPolicyConfig(uncertainty_aversion=0.0))
        bullish = prediction(1, 2, (0.05, 0.1, 0.85))
        flat = prediction(2, 1, (0.05, 0.9, 0.05))
        bearish = prediction(3, 0, (0.85, 0.1, 0.05))

        opened = policy.target_position(bullish, 0.0)
        held = policy.target_position(flat, opened)
        closed = policy.target_position(bearish, held)

        self.assertEqual(opened, 1.0)
        self.assertEqual(held, opened)
        self.assertEqual(closed, 0.0)

        uncertain_bull = replace(
            bullish,
            aleatoric_variance=(0.08,) * 3,
            epistemic_variance=(0.02,) * 3,
        )
        cautious = DecisionPolicy(DecisionPolicyConfig(uncertainty_aversion=4.0))
        self.assertEqual(cautious.target_position(uncertain_bull, 0.7), 0.7)

    def test_continuous_posterior_return_is_used_directly(self) -> None:
        first = replace(
            prediction(1, 2, (0.0, 0.0, 1.0)), symbol="000001"
        )
        second = replace(first, symbol="000002")
        report = evaluate_predictions(
            [first, second],
            {
                "000001": (-0.01, 0.0, 0.01),
                "000002": (-0.02, 0.0, 0.03),
            },
            {"000001": (0.0001,) * 3, "000002": (0.0001,) * 3},
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
