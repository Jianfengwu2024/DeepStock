from __future__ import annotations

import unittest

import numpy as np

from bnhmm.sector_flow import (
    FlowAttributionPosterior,
    NormalizedSectorPanel,
    PathEfficiencyAnalyzer,
    RankingConfig,
    SectorFlowPosterior,
    SourceProximityRanker,
    StockFlowAttribution,
)


def ranking_fixture(
    raw_velocity: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> tuple[NormalizedSectorPanel, SectorFlowPosterior, FlowAttributionPosterior]:
    rows, symbols = raw_velocity.shape
    mask = np.ones_like(raw_velocity, dtype=np.bool_) if valid is None else valid
    timestamps = (
        np.datetime64("2025-01-02T09:31")
        + np.arange(rows).astype("timedelta64[m]")
    ).astype("datetime64[ns]")
    normalized = np.where(mask, raw_velocity, np.nan)
    panel = NormalizedSectorPanel(
        symbols=tuple(f"S{index}" for index in range(symbols)),
        timestamps=timestamps,
        raw_velocity=normalized.copy(),
        velocity=normalized.copy(),
        activity=np.ones_like(raw_velocity),
        velocity_scale=np.ones_like(raw_velocity),
        volume_baseline=np.ones_like(raw_velocity),
        valid=mask,
        activity_valid=mask.copy(),
    )
    factor_valid = np.any(mask, axis=1)
    factor = np.divide(
        np.nansum(normalized, axis=1),
        np.maximum(mask.sum(axis=1), 1),
    )
    flow = SectorFlowPosterior(
        symbols=panel.symbols,
        timestamps=timestamps,
        factor_mean=factor,
        factor_variance=np.full(rows, 0.1),
        leave_one_out_mean=np.repeat(factor[:, None], symbols, axis=1),
        leave_one_out_variance=np.full_like(raw_velocity, 0.1),
        activity=np.ones(rows),
        breadth=np.ones(rows),
        coherence=np.ones(rows),
        dispersion=np.zeros(rows),
        flow_strength=factor,
        valid=factor_valid,
    )
    attribution = FlowAttributionPosterior(
        symbols=panel.symbols,
        timestamps=timestamps,
        leader_mean=normalized.copy(),
        leader_variance=np.full_like(raw_velocity, 0.1),
        response_mean=normalized.copy(),
        response_variance=np.full_like(raw_velocity, 0.1),
        valid=mask.copy(),
    )
    return panel, flow, attribution


class PathEfficiencyAnalyzerTests(unittest.TestCase):
    def test_monotone_and_oscillating_paths_have_expected_efficiency(self) -> None:
        monotone = np.full(60, 0.01)
        oscillating = np.tile([0.01, -0.01], 30)
        panel, flow, attribution = ranking_fixture(
            np.column_stack([monotone, oscillating])
        )

        metrics = PathEfficiencyAnalyzer().compute(panel, flow, attribution)
        by_symbol = {item.symbol: item for item in metrics}
        long_monotone = by_symbol["S0"].window(60)
        long_oscillating = by_symbol["S1"].window(60)

        self.assertAlmostEqual(long_monotone.efficiency, 1.0)
        self.assertAlmostEqual(long_oscillating.efficiency, 0.0)
        self.assertEqual(long_monotone.reversal_count, 0)
        self.assertEqual(long_oscillating.reversal_count, 59)

    def test_missing_minutes_are_masked_and_all_windows_are_returned(self) -> None:
        values = np.full((70, 1), 0.01)
        valid = np.ones_like(values, dtype=np.bool_)
        valid[-10:-5] = False
        panel, flow, attribution = ranking_fixture(values, valid=valid)

        result = PathEfficiencyAnalyzer().compute(panel, flow, attribution)[0]

        self.assertEqual(tuple(item.minutes for item in result.windows), (5, 15, 30, 60))
        self.assertEqual(result.window(15).valid_minutes, 10)
        self.assertAlmostEqual(result.window(15).efficiency, 1.0)

    def test_endpoint_metrics_ignore_future_rows(self) -> None:
        values = np.full((80, 1), 0.01)
        panel, flow, attribution = ranking_fixture(values)
        original = PathEfficiencyAnalyzer().compute(
            panel, flow, attribution, endpoint=59
        )[0]
        changed_values = values.copy()
        changed_values[60:] = -10.0
        changed_panel, changed_flow, changed_attribution = ranking_fixture(
            changed_values
        )
        changed = PathEfficiencyAnalyzer().compute(
            changed_panel, changed_flow, changed_attribution, endpoint=59
        )[0]

        self.assertEqual(original, changed)


class SourceProximityRankerTests(unittest.TestCase):
    def test_joint_posterior_is_reproducible_and_exposes_components(self) -> None:
        values = np.full((60, 1), 0.01)
        panel, flow, attribution = ranking_fixture(values)
        path = PathEfficiencyAnalyzer().compute(panel, flow, attribution)
        summaries = (
            StockFlowAttribution(
                symbol="S0",
                leader_probability=0.95,
                leader_horizon_minutes=2,
                leader_coefficient_mean=0.8,
                leader_coefficient_std=0.1,
                response_probability=0.9,
                response_delay_minutes=1,
                response_coefficient_mean=0.7,
                response_coefficient_std=0.1,
                coupling_mean=0.7,
                coupling_std=0.1,
                residual_scale=0.1,
            ),
        )
        ranker = SourceProximityRanker(
            RankingConfig(monte_carlo_samples=4000, random_seed=23)
        )

        first = ranker.rank(path, summaries)[0]
        second = ranker.rank(path, summaries)[0]

        self.assertEqual(first, second)
        self.assertGreater(first.source_proximity_probability, 0.8)
        self.assertGreater(first.lead_or_response_probability, 0.9)
        self.assertGreater(first.coupling_probability, 0.99)
        self.assertGreater(first.efficiency_probability, 0.99)
        self.assertGreater(first.low_residual_probability, 0.99)
        self.assertGreater(first.activity_confirmation_probability, 0.5)


if __name__ == "__main__":
    unittest.main()
