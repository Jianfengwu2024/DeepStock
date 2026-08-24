from __future__ import annotations

import unittest

import numpy as np

from bnhmm.sector_flow.regime_data import SectorRegimeState
from bnhmm.sector_flow.regime_evaluation import SectorRegimePredictionRecord
from bnhmm.sector_flow.regime_policy import (
    SectorRegimePolicyConfig,
    StockSelectionSnapshot,
    apply_sector_regime_policy,
    evaluate_sector_regime_policy,
)


def regime(
    index: int, probability: tuple[float, float, float], variance: float = 0.001
) -> SectorRegimePredictionRecord:
    signal = np.datetime64("2025-02-03", "D") + np.timedelta64(index, "D")
    return SectorRegimePredictionRecord(
        signal_time=signal.astype("datetime64[ns]") + np.timedelta64(15, "h"),
        available_at=(signal + np.timedelta64(1, "D")).astype("datetime64[ns]")
        + np.timedelta64(9, "h"),
        target_time=(signal + np.timedelta64(1, "D")).astype("datetime64[ns]")
        + np.timedelta64(15, "h"),
        training_cutoff=np.datetime64("2025-01-31T15:00", "ns"),
        model_version="sector-regime-bnhmm-v1",
        probability_mean=np.asarray(probability),
        probability_variance=np.full(3, variance),
        previous_state=SectorRegimeState.FLAT,
        realized_state=SectorRegimeState.BULL,
        realized_return=0.01,
        return_scale=0.01,
        next_image_mean=np.zeros((4, 2, 3)),
        next_image_variance=np.zeros((4, 2, 3)),
    )


def snapshot(item: SectorRegimePredictionRecord) -> StockSelectionSnapshot:
    return StockSelectionSnapshot(
        signal_time=item.signal_time,
        symbols=("A", "B", "C", "D"),
        relative_scores=np.asarray([0.01, 0.04, -0.02, 0.03]),
        next_open_close_returns=np.asarray([0.01, 0.03, -0.02, 0.02]),
    )


class SectorRegimePolicyTests(unittest.TestCase):
    def test_only_confident_low_uncertainty_bull_opens_top_k(self) -> None:
        regimes = (
            regime(0, (0.1, 0.2, 0.7)),
            regime(1, (0.1, 0.8, 0.1)),
            regime(2, (0.7, 0.2, 0.1)),
            regime(3, (0.1, 0.2, 0.7), variance=0.05),
        )
        config = SectorRegimePolicyConfig(
            top_k=2,
            bull_probability_threshold=0.6,
            maximum_bull_variance=0.01,
            transaction_cost_bps=10.0,
        )

        observations = apply_sector_regime_policy(
            regimes, tuple(snapshot(item) for item in regimes), config
        )

        self.assertEqual(observations[0].selected_symbols, ("B", "D"))
        self.assertAlmostEqual(observations[0].gross_return, 0.025)
        self.assertAlmostEqual(observations[0].transaction_cost, 0.002)
        self.assertAlmostEqual(observations[0].net_return, 0.023)
        self.assertEqual(observations[1].selected_symbols, ())
        self.assertEqual(observations[2].selected_symbols, ())
        self.assertEqual(observations[3].selected_symbols, ())
        self.assertTrue(all(item.transaction_cost == 0.0 for item in observations[1:]))

    def test_evaluation_separates_timing_and_conditional_selection(self) -> None:
        regimes = (
            regime(0, (0.1, 0.2, 0.7)),
            regime(1, (0.1, 0.8, 0.1)),
        )
        observations = apply_sector_regime_policy(
            regimes,
            tuple(snapshot(item) for item in regimes),
            SectorRegimePolicyConfig(
                top_k=2,
                bull_probability_threshold=0.6,
                maximum_bull_variance=0.01,
                transaction_cost_bps=10.0,
            ),
        )

        report = evaluate_sector_regime_policy(observations)

        self.assertEqual(report["exposed_sessions"], 1)
        self.assertAlmostEqual(report["exposure_rate"], 0.5)
        self.assertGreater(report["ending_equity"], 1.0)
        self.assertGreater(report["conditional_rank_ic"], 0.0)
        self.assertGreater(report["always_hold_index_equity"], 1.0)


if __name__ == "__main__":
    unittest.main()
