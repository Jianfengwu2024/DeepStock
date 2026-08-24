from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from bnhmm.sector_flow import (
    MonthlyRiverTopology,
    NormalizedSectorPanel,
    RiverTopologyBuilder,
    TopologyConfig,
)


def topology(**changes: object) -> MonthlyRiverTopology:
    values: dict[str, object] = {
        "month": np.datetime64("2025-02", "M"),
        "training_cutoff": np.datetime64("2025-01-27T15:00"),
        "symbols": ("A", "B"),
        "assigned": np.ones(2, dtype=np.bool_),
        "band_ids": np.array([0, 0], dtype=np.int64),
        "row_order": np.array([0, 1], dtype=np.int64),
        "common_similarity": np.eye(2),
        "residual_similarity": np.eye(2),
        "fused_similarity": np.eye(2),
        "graph": np.eye(2),
        "cohesion": np.ones(2),
        "centrality": np.ones(2),
    }
    values.update(changes)
    return MonthlyRiverTopology(**values)


class MonthlyRiverTopologyContractTests(unittest.TestCase):
    def test_requires_one_vector_value_per_symbol(self) -> None:
        with self.assertRaisesRegex(ValueError, "topology vectors"):
            topology(band_ids=np.array([0], dtype=np.int64))

    def test_requires_square_symbol_aligned_matrices(self) -> None:
        with self.assertRaisesRegex(ValueError, "topology matrices"):
            topology(graph=np.eye(3))

    def test_requires_a_complete_row_permutation(self) -> None:
        with self.assertRaisesRegex(ValueError, "row order"):
            topology(row_order=np.array([0, 0], dtype=np.int64))


def planted_panel(*, sessions: int = 65, symbols: int = 8) -> NormalizedSectorPanel:
    rng = np.random.default_rng(1729)
    days = np.busday_offset(np.datetime64("2025-01-02", "D"), np.arange(sessions))
    slots = np.concatenate((np.arange(571, 691), np.arange(781, 901)))
    timestamps = np.asarray(
        [day + np.timedelta64(int(slot), "m") for day in days for slot in slots],
        dtype="datetime64[ns]",
    )
    rows = len(timestamps)
    common = rng.normal(0.0, 0.45, rows)
    first = rng.normal(0.0, 0.9, rows)
    second = rng.normal(0.0, 0.9, rows)
    velocity = np.empty((rows, symbols), dtype=np.float64)
    for column in range(symbols):
        branch = first if column < symbols // 2 else second
        velocity[:, column] = common + branch + rng.normal(0.0, 0.18, rows)
    activity = rng.lognormal(0.0, 0.25, size=velocity.shape)
    valid = np.ones(velocity.shape, dtype=np.bool_)
    return NormalizedSectorPanel(
        symbols=tuple(f"S{column:02d}" for column in range(symbols)),
        timestamps=timestamps,
        raw_velocity=velocity * 1e-3,
        velocity=velocity,
        activity=activity,
        velocity_scale=np.ones_like(velocity),
        volume_baseline=np.ones_like(velocity),
        valid=valid,
        activity_valid=valid.copy(),
    )


class RiverTopologyBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.panel = planted_panel()
        sessions = np.unique(self.panel.timestamps.astype("datetime64[D]"))
        self.cutoff = sessions[59] + np.timedelta64(15 * 60, "m")
        self.month = (self.cutoff + np.timedelta64(32, "D")).astype("datetime64[M]")
        self.builder = RiverTopologyBuilder(
            TopologyConfig(minimum_sessions=40, minimum_pair_sessions=30)
        )

    def test_similarity_recovers_planted_residual_groups(self) -> None:
        result = self.builder.build(self.panel, self.cutoff, self.month)

        self.assertGreater(result.fused_similarity[0, 1], result.fused_similarity[0, 6])
        self.assertGreater(
            result.residual_similarity[4, 5],
            result.residual_similarity[4, 1],
        )
        self.assertEqual(result.band_ids[0], result.band_ids[1])
        self.assertNotEqual(result.band_ids[0], result.band_ids[6])

    def test_future_rows_do_not_change_topology(self) -> None:
        original = self.builder.build(self.panel, self.cutoff, self.month)
        future = self.panel.timestamps > self.cutoff
        velocity = self.panel.velocity.copy()
        velocity[future] = 1000.0
        changed = replace(self.panel, velocity=velocity)

        repeated = self.builder.build(changed, self.cutoff, self.month)

        np.testing.assert_allclose(original.fused_similarity, repeated.fused_similarity)
        np.testing.assert_array_equal(original.band_ids, repeated.band_ids)
        np.testing.assert_array_equal(original.row_order, repeated.row_order)

    def test_uniform_activity_rescaling_does_not_change_topology(self) -> None:
        original = self.builder.build(self.panel, self.cutoff, self.month)
        scaled = replace(self.panel, activity=self.panel.activity * 50.0)

        repeated = self.builder.build(scaled, self.cutoff, self.month)

        np.testing.assert_allclose(original.fused_similarity, repeated.fused_similarity)
        np.testing.assert_array_equal(original.band_ids, repeated.band_ids)

    def test_input_symbol_permutation_preserves_canonical_order(self) -> None:
        original = self.builder.build(self.panel, self.cutoff, self.month)
        permutation = np.array([7, 1, 6, 0, 4, 2, 5, 3])
        permuted = replace(
            self.panel,
            symbols=tuple(self.panel.symbols[index] for index in permutation),
            raw_velocity=self.panel.raw_velocity[:, permutation],
            velocity=self.panel.velocity[:, permutation],
            activity=self.panel.activity[:, permutation],
            velocity_scale=self.panel.velocity_scale[:, permutation],
            volume_baseline=self.panel.volume_baseline[:, permutation],
            valid=self.panel.valid[:, permutation],
            activity_valid=self.panel.activity_valid[:, permutation],
        )

        repeated = self.builder.build(permuted, self.cutoff, self.month)

        self.assertEqual(original.symbols, repeated.symbols)
        np.testing.assert_array_equal(original.band_ids, repeated.band_ids)
        np.testing.assert_array_equal(original.row_order, repeated.row_order)


if __name__ == "__main__":
    unittest.main()
