from __future__ import annotations

import unittest

import numpy as np

from bnhmm.sector_flow.evaluation import (
    SectorFlowPrediction,
    build_monthly_folds,
    evaluate_predictions,
)


class SectorFlowFoldTests(unittest.TestCase):
    def test_monthly_folds_are_expanding_and_cutoff_precedes_test(self) -> None:
        sessions = np.arange(
            np.datetime64("2025-01-02"),
            np.datetime64("2025-04-01"),
            dtype="datetime64[D]",
        )
        timestamps = (
            sessions[:, None].astype("datetime64[m]")
            + np.asarray([9 * 60 + 31, 9 * 60 + 32], dtype="timedelta64[m]")
        ).reshape(-1).astype("datetime64[ns]")

        folds = build_monthly_folds(timestamps, minimum_train_sessions=20)

        self.assertEqual(len(folds), 2)
        self.assertEqual(str(folds[0].test_start.astype("datetime64[D]")), "2025-02-01")
        self.assertEqual(str(folds[1].test_start.astype("datetime64[D]")), "2025-03-01")
        for fold in folds:
            self.assertTrue(np.all(timestamps[fold.training_rows] < fold.test_start))
            self.assertTrue(np.all(timestamps[fold.test_rows] >= fold.test_start))
            self.assertTrue(np.all(timestamps[fold.test_rows] < fold.test_end))

    def test_prediction_becomes_available_on_following_minute(self) -> None:
        signal_time = np.datetime64("2025-02-03T10:00", "ns")
        prediction = SectorFlowPrediction(
            symbol="S0",
            signal_time=signal_time,
            available_at=signal_time + np.timedelta64(1, "m"),
            training_cutoff=np.datetime64("2025-02-01", "ns"),
            model_version="sector-flow-bayes-v1",
            fold=0,
            source_probability=0.8,
            direction_probability=0.7,
            realized_return=0.01,
            interval_lower=-0.01,
            interval_upper=0.02,
            leader_horizon_minutes=3,
            response_delay_minutes=2,
        )

        self.assertGreater(prediction.available_at, prediction.signal_time)


class SectorFlowMetricTests(unittest.TestCase):
    def test_metrics_cover_probability_ranking_costs_and_posterior(self) -> None:
        base = np.datetime64("2025-02-03T10:00", "ns")
        records = []
        for minute in range(4):
            for symbol, probability, realized in (
                ("A", 0.9, 0.02),
                ("B", 0.6, 0.005),
                ("C", 0.2, -0.01),
            ):
                signal = base + np.timedelta64(minute, "m")
                records.append(
                    SectorFlowPrediction(
                        symbol=symbol,
                        signal_time=signal,
                        available_at=signal + np.timedelta64(1, "m"),
                        training_cutoff=np.datetime64("2025-02-01", "ns"),
                        model_version="sector-flow-bayes-v1",
                        fold=0,
                        source_probability=probability,
                        direction_probability=probability,
                        realized_return=realized,
                        interval_lower=realized - 0.01,
                        interval_upper=realized + 0.01,
                        leader_horizon_minutes=3 if symbol == "A" else 5,
                        response_delay_minutes=2 if symbol == "A" else 4,
                    )
                )

        metrics = evaluate_predictions(records, top_k=1, transaction_cost_bps=10.0)

        self.assertLess(metrics["brier_score"], 0.1)
        self.assertLess(metrics["log_loss"], 0.4)
        self.assertGreater(metrics["rank_ic"], 0.9)
        self.assertGreater(metrics["top_bottom_return"], 0.0)
        self.assertGreaterEqual(metrics["net_top_bottom_return"], 0.0)
        self.assertEqual(metrics["posterior_coverage"], 1.0)
        self.assertGreaterEqual(metrics["leader_lag_stability"], 0.0)
        self.assertGreaterEqual(metrics["response_lag_stability"], 0.0)


if __name__ == "__main__":
    unittest.main()
