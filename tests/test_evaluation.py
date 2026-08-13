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
        forward_return=(-0.03, -0.008, 0.008, 0.03)[target],
        probability_mean=probability,
        aleatoric_variance=(0.01,) * 4,
        epistemic_variance=(0.001,) * 4,
        expected_score=float(np.asarray(probability) @ scores),
        context_length=8,
    )


class EvaluationTests(unittest.TestCase):
    def test_report_contains_probabilistic_and_strategy_diagnostics(self) -> None:
        predictions = [
            prediction(index, index % 4, tuple(np.eye(4)[index % 4] * 0.6 + 0.1))
            for index in range(12)
        ]
        report = evaluate_predictions(
            predictions,
            np.asarray([-0.03, -0.01, 0.01, 0.03]),
            np.full(4, 0.0001),
        )

        self.assertLess(report["log_loss"], 1.0)
        self.assertEqual(len(report["residual_autocorrelation"]), 20)
        self.assertIn("max_drawdown", report["backtest"])
        self.assertIn("average_abs_position", report["backtest"])
        self.assertGreaterEqual(report["backtest"]["active_position_ratio"], 0.0)
        self.assertEqual(report["observation_count"], 12)
        self.assertEqual(report["balanced_accuracy"], 1.0)
        self.assertEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["class_support"], [3, 3, 3, 3])
        self.assertEqual(report["confusion_matrix"], (np.eye(4, dtype=int) * 3).tolist())
        self.assertLess(report["ranked_probability_score"], 0.1)

    def test_uncertainty_reduces_position(self) -> None:
        item = prediction(1, 3, (0.02, 0.03, 0.2, 0.75))
        cautious = DecisionPolicy(DecisionPolicyConfig(uncertainty_aversion=4.0))
        relaxed = DecisionPolicy(DecisionPolicyConfig(uncertainty_aversion=0.0))
        self.assertLess(cautious.target_position(item), relaxed.target_position(item))

    def test_five_state_predictions_are_rejected(self) -> None:
        item = prediction(1, 3, (0.02, 0.03, 0.2, 0.75))
        legacy = replace(
            item,
            probability_mean=(0.1, 0.1, 0.6, 0.1, 0.1),
            aleatoric_variance=(0.01,) * 5,
            epistemic_variance=(0.001,) * 5,
        )
        with self.assertRaisesRegex(ValueError, "four-state"):
            evaluate_predictions([legacy], np.zeros(4), np.ones(4))

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
