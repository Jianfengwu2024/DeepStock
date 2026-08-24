from __future__ import annotations

import unittest

import numpy as np

from bnhmm.sector_flow.regime_data import SectorRegimeState
from bnhmm.sector_flow.regime_evaluation import (
    SectorRegimePredictionRecord,
    evaluate_sector_regimes,
)


def record(
    index: int,
    realized: SectorRegimeState,
    previous: SectorRegimeState,
) -> SectorRegimePredictionRecord:
    probability = np.full(3, 0.05)
    probability[int(realized)] = 0.9
    signal = np.datetime64("2025-02-01", "D") + np.timedelta64(index, "D")
    return SectorRegimePredictionRecord(
        signal_time=signal.astype("datetime64[ns]") + np.timedelta64(15, "h"),
        available_at=(signal + 1).astype("datetime64[ns]") + np.timedelta64(9, "h"),
        target_time=(signal + 1).astype("datetime64[ns]") + np.timedelta64(15, "h"),
        training_cutoff=np.datetime64("2025-01-31T15:00", "ns"),
        model_version="sector-regime-bnhmm-v1",
        probability_mean=probability,
        probability_variance=np.full(3, 0.001),
        previous_state=previous,
        realized_state=realized,
        realized_return=(-0.01, 0.0, 0.01)[int(realized)],
        return_scale=0.01,
        next_image_mean=np.zeros((4, 2, 3)),
        next_image_variance=np.zeros((4, 2, 3)),
    )


class SectorRegimeEvaluationTests(unittest.TestCase):
    def test_reports_three_state_skill_and_causal_baselines(self) -> None:
        records = (
            record(0, SectorRegimeState.BEAR, SectorRegimeState.BULL),
            record(1, SectorRegimeState.FLAT, SectorRegimeState.BEAR),
            record(2, SectorRegimeState.BULL, SectorRegimeState.FLAT),
        )

        report = evaluate_sector_regimes(
            records, training_class_prior=np.asarray([0.7, 0.2, 0.1])
        )

        self.assertAlmostEqual(report["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(report["support"], [1, 1, 1])
        self.assertAlmostEqual(report["majority_baseline_accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(report["persistence_baseline_accuracy"], 0.0)
        self.assertLess(report["brier_score"], 0.02)
        self.assertLess(report["log_loss"], 0.2)


if __name__ == "__main__":
    unittest.main()
