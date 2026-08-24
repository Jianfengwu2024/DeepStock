from __future__ import annotations

import unittest

import numpy as np

from bnhmm.sector_flow import (
    DailyRiverSliceBuilder,
    DailyRiverSlice,
    GraphStateParameters,
    MonthlyRiverTopology,
    NormalizedSectorPanel,
    SectorFactorParameters,
    SectorFlowPosterior,
    SectorGraphPrediction,
    daily_statistics,
)


class RiverContractTests(unittest.TestCase):
    def test_daily_slice_is_symbols_by_minutes(self) -> None:
        item = DailyRiverSlice(
            session=np.datetime64("2025-02-05", "D"),
            topology_month=np.datetime64("2025-02", "M"),
            symbols=("B", "A"),
            timestamps=np.array(
                ["2025-02-05T09:31", "2025-02-05T09:32"],
                dtype="datetime64[m]",
            ),
            velocity=np.zeros((2, 2)),
            residual_velocity=np.zeros((2, 2)),
            activity=np.ones((2, 2)),
            valid=np.ones((2, 2), dtype=np.bool_),
        )

        self.assertEqual(item.velocity.shape, (2, 2))

    def test_prediction_requires_strict_causal_timestamps(self) -> None:
        with self.assertRaisesRegex(ValueError, "prediction timing"):
            SectorGraphPrediction(
                topology_month=np.datetime64("2025-02", "M"),
                symbols=("A", "B"),
                expected_relative_return=np.zeros(2),
                predictive_variance=np.ones(2),
                signal_time=np.datetime64("2025-02-05T15:00"),
                available_at=np.datetime64("2025-02-05T15:00"),
                training_cutoff=np.datetime64("2025-01-31T15:00"),
                model_version="sector-flow-river-v2",
            )


def river_fixture(*, missing_minute: int | None = None):
    slots = np.concatenate((np.arange(571, 691), np.arange(781, 901)))
    timestamps = np.asarray(
        [np.datetime64("2025-02-05", "D") + np.timedelta64(int(slot), "m") for slot in slots],
        dtype="datetime64[ns]",
    )
    velocity = np.column_stack((np.linspace(-1.0, 1.0, 240), np.linspace(1.0, -1.0, 240)))
    valid = np.ones((240, 2), dtype=np.bool_)
    if missing_minute is not None:
        valid[missing_minute, 0] = False
        velocity[missing_minute, 0] = np.nan
    panel = NormalizedSectorPanel(
        symbols=("A", "B"),
        timestamps=timestamps,
        raw_velocity=np.where(valid, velocity * 1e-3, np.nan),
        velocity=velocity,
        activity=np.where(valid, 1.5, np.nan),
        velocity_scale=np.ones((240, 2)),
        volume_baseline=np.ones((240, 2)),
        valid=valid,
        activity_valid=valid.copy(),
    )
    parameters = SectorFactorParameters(
        loadings=np.array([1.0, 0.5]),
        residual_variance=np.ones(2),
        persistence=0.0,
        process_variance=1.0,
    )
    flow = SectorFlowPosterior(
        symbols=panel.symbols,
        timestamps=timestamps,
        factor_mean=np.full(240, 0.25),
        factor_variance=np.ones(240),
        leave_one_out_mean=np.zeros((240, 2)),
        leave_one_out_variance=np.ones((240, 2)),
        activity=np.ones(240),
        breadth=np.ones(240),
        coherence=np.ones(240),
        dispersion=np.zeros(240),
        flow_strength=np.ones(240),
        valid=np.ones(240, dtype=np.bool_),
    )
    topology = MonthlyRiverTopology(
        month=np.datetime64("2025-02", "M"),
        training_cutoff=np.datetime64("2025-01-31T15:00"),
        symbols=panel.symbols,
        assigned=np.ones(2, dtype=np.bool_),
        band_ids=np.array([0, 0], dtype=np.int64),
        row_order=np.array([1, 0], dtype=np.int64),
        common_similarity=np.eye(2),
        residual_similarity=np.eye(2),
        fused_similarity=np.eye(2),
        graph=np.zeros((2, 2)),
        cohesion=np.ones(2),
        centrality=np.ones(2),
    )
    return panel, parameters, flow, topology


class DailyRiverSliceBuilderTests(unittest.TestCase):
    def test_uses_frozen_topology_order_and_real_minute_grid(self) -> None:
        panel, parameters, flow, topology = river_fixture()

        result = DailyRiverSliceBuilder().build(
            panel,
            parameters,
            flow,
            topology,
            np.datetime64("2025-02-05", "D"),
        )

        self.assertEqual(result.symbols, ("B", "A"))
        self.assertEqual(result.velocity.shape, (2, 240))
        self.assertNotIn(np.datetime64("2025-02-05T11:31"), result.timestamps)
        self.assertNotIn(np.datetime64("2025-02-05T13:00"), result.timestamps)
        np.testing.assert_allclose(
            result.residual_velocity[0],
            panel.velocity[:, 1] - 0.5 * flow.factor_mean,
        )

    def test_missing_bar_stays_masked(self) -> None:
        panel, parameters, flow, topology = river_fixture(missing_minute=44)

        result = DailyRiverSliceBuilder().build(
            panel,
            parameters,
            flow,
            topology,
            np.datetime64("2025-02-05", "D"),
        )

        row = result.symbols.index("A")
        self.assertFalse(result.valid[row, 44])
        self.assertTrue(np.isnan(result.velocity[row, 44]))
        self.assertTrue(np.isnan(result.residual_velocity[row, 44]))

    def test_daily_statistics_consume_full_minute_slice(self) -> None:
        panel, parameters, flow, topology = river_fixture()
        result = DailyRiverSliceBuilder().build(
            panel,
            parameters,
            flow,
            topology,
            np.datetime64("2025-02-05", "D"),
        )

        statistics = daily_statistics(result, topology)

        self.assertEqual(statistics.shape, (2, 6))
        self.assertTrue(np.all(np.isfinite(statistics)))

    def test_graph_parameters_require_positive_feature_scales(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature scales"):
            GraphStateParameters(
                feature_center=np.zeros(2),
                feature_scale=np.array([1.0, 0.0]),
                coefficient_mean=np.zeros(3),
                coefficient_covariance=np.eye(3),
                residual_variance=1.0,
                graph_strength=1.0,
                training_cutoff=np.datetime64("2025-01-31T15:00"),
                model_version="sector-flow-river-v2",
            )


if __name__ == "__main__":
    unittest.main()
