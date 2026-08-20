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
    def test_vectorized_series_matches_endpoint_metrics(self) -> None:
        rng = np.random.default_rng(17)
        values = rng.normal(0.0, 0.02, (80, 3))
        valid = rng.random(values.shape) > 0.12
        panel, flow, attribution = ranking_fixture(values, valid=valid)
        analyzer = PathEfficiencyAnalyzer((5, 15))

        series = analyzer.compute_series(panel, flow, attribution, minutes=15)

        for endpoint in (0, 4, 14, 39, 79):
            expected = analyzer.compute(
                panel, flow, attribution, endpoint=endpoint
            )
            actual = series.at(endpoint)
            for expected_stock, actual_stock in zip(expected, actual):
                expected_metric = expected_stock.window(15)
                actual_metric = actual_stock.window(15)
                self.assertEqual(
                    actual_metric.valid_minutes, expected_metric.valid_minutes
                )
                self.assertEqual(
                    actual_metric.reversal_count, expected_metric.reversal_count
                )
                self.assertAlmostEqual(
                    actual_metric.efficiency, expected_metric.efficiency
                )
                self.assertAlmostEqual(
                    actual_metric.directional_agreement,
                    expected_metric.directional_agreement,
                )
                self.assertAlmostEqual(
                    actual_metric.activity_confirmation,
                    expected_metric.activity_confirmation,
                )
                self.assertAlmostEqual(
                    actual_metric.residual_fraction,
                    expected_metric.residual_fraction,
                )

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

    def test_path_windows_do_not_cross_market_time_gaps(self) -> None:
        values = np.full((80, 1), 0.01)
        panel, flow, attribution = ranking_fixture(values)
        panel.timestamps[40:] += np.timedelta64(1, "D")
        analyzer = PathEfficiencyAnalyzer((15,))

        detailed = analyzer.compute(
            panel, flow, attribution, endpoint=40
        )[0].window(15)
        vectorized = analyzer.compute_series(
            panel, flow, attribution, minutes=15
        ).at(40)[0].window(15)

        self.assertEqual(detailed.valid_minutes, 1)
        self.assertEqual(vectorized.valid_minutes, 1)
        self.assertEqual(detailed.reversal_count, 0)
        self.assertEqual(vectorized.reversal_count, 0)


class SourceProximityRankerTests(unittest.TestCase):
    def test_missing_endpoint_path_cannot_receive_source_probability(self) -> None:
        values = np.full((60, 1), 0.01)
        valid = np.ones_like(values, dtype=np.bool_)
        valid[-1] = False
        panel, flow, attribution = ranking_fixture(values, valid=valid)
        analyzer = PathEfficiencyAnalyzer((1,))
        series = analyzer.compute_series(panel, flow, attribution, minutes=1)
        summaries = (
            StockFlowAttribution(
                symbol="S0",
                leader_probability=0.99,
                leader_horizon_minutes=1,
                leader_coefficient_mean=1.0,
                leader_coefficient_std=0.01,
                response_probability=0.99,
                response_delay_minutes=1,
                response_coefficient_mean=1.0,
                response_coefficient_std=0.01,
                coupling_mean=1.0,
                coupling_std=0.01,
                residual_scale=0.01,
            ),
        )
        ranker = SourceProximityRanker(
            RankingConfig(
                windows=(1,),
                ranking_window=1,
                monte_carlo_samples=1000,
            )
        )

        detailed = ranker.rank(series.at(59), summaries)[0]
        fast = ranker.rank_series(series, summaries)

        self.assertEqual(detailed.path_metrics.valid_minutes, 0)
        self.assertEqual(detailed.source_proximity_probability, 0.0)
        self.assertEqual(fast.source_proximity_probability[59, 0], 0.0)

    def test_series_posterior_approximates_detailed_monte_carlo(self) -> None:
        rng = np.random.default_rng(29)
        values = rng.normal(0.002, 0.01, (120, 2))
        panel, flow, attribution = ranking_fixture(values)
        analyzer = PathEfficiencyAnalyzer((60,))
        series = analyzer.compute_series(panel, flow, attribution, minutes=60)
        summaries = tuple(
            StockFlowAttribution(
                symbol=symbol,
                leader_probability=0.7 + index * 0.1,
                leader_horizon_minutes=2,
                leader_coefficient_mean=0.5,
                leader_coefficient_std=0.1,
                response_probability=0.65,
                response_delay_minutes=1,
                response_coefficient_mean=0.4,
                response_coefficient_std=0.1,
                coupling_mean=0.35,
                coupling_std=0.2,
                residual_scale=0.1,
            )
            for index, symbol in enumerate(panel.symbols)
        )
        ranker = SourceProximityRanker(
            RankingConfig(
                windows=(60,),
                ranking_window=60,
                monte_carlo_samples=100_000,
                random_seed=31,
            )
        )

        fast = ranker.rank_series(series, summaries)
        detailed = {
            item.symbol: item
            for item in ranker.rank(series.at(119), summaries)
        }

        for column, symbol in enumerate(panel.symbols):
            self.assertAlmostEqual(
                fast.source_proximity_probability[119, column],
                detailed[symbol].source_proximity_probability,
                delta=0.035,
            )
            component_product = (
                fast.lead_or_response_probability[column]
                * fast.coupling_probability[column]
                * fast.efficiency_probability[119, column]
                * fast.low_residual_probability[119, column]
                * fast.activity_confirmation_probability[119, column]
            )
            self.assertAlmostEqual(
                fast.source_proximity_probability[119, column],
                component_product,
            )

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
