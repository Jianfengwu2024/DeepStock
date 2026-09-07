from __future__ import annotations

import numpy as np
import pytest

from bnhmm.boundary_fk.daily_state import (
    BoundaryDailyStateBuilder,
    BoundaryTopologyBuilder,
)
from bnhmm.boundary_fk.types import NormalizedBoundaryPanel
from bnhmm.sector_flow.topology import TopologyConfig


def test_boundary_flow_topology_preserves_multi_branch_cap() -> None:
    builder = BoundaryTopologyBuilder(
        TopologyConfig(maximum_bands=4, minimum_band_size=2)
    )

    assert builder.builder.config.maximum_bands == 4


def test_daily_state_builder_accepts_dynamic_branch_configuration() -> None:
    assert BoundaryDailyStateBuilder().config.branch_count == 2
    assert BoundaryDailyStateBuilder(branch_count=3).config.branch_count == 3


def test_boundary_topology_uses_only_rows_before_effective_month() -> None:
    rng = np.random.default_rng(7)
    symbols = tuple(f"S{i}" for i in range(6))
    days = np.asarray(
        ["2024-01-25", "2024-01-26", "2024-01-29", "2024-02-01"],
        dtype="datetime64[D]",
    )
    timestamps = np.asarray(
        [day + np.timedelta64(slot, "m") for day in days for slot in (571, 572)],
        dtype="datetime64[ns]",
    )
    values = rng.normal(size=(len(timestamps), len(symbols)))
    valid = np.ones(values.shape, dtype=np.bool_)
    panel = NormalizedBoundaryPanel(
        symbols=symbols,
        timestamps=timestamps,
        raw_velocity=values * 0.001,
        standardized_velocity=values,
        relative_volume=np.ones(values.shape),
        velocity_scale=np.ones(values.shape),
        minute_volume_baseline=np.ones(values.shape),
        raw_velocity_valid=valid,
        velocity_valid=valid,
        density_valid=valid,
    )
    builder = BoundaryTopologyBuilder(
        TopologyConfig(
            lookback_sessions=3,
            minimum_sessions=2,
            minimum_pair_sessions=2,
            minimum_minutes_per_session=2,
            maximum_bands=3,
            minimum_band_size=2,
        )
    )
    base = builder.build(panel, [np.datetime64("2024-02", "M")])[0]
    changed_values = panel.standardized_velocity.copy()
    changed_values[timestamps >= np.datetime64("2024-02-01")] = 1_000_000.0
    changed = NormalizedBoundaryPanel(
        symbols=panel.symbols,
        timestamps=panel.timestamps,
        raw_velocity=panel.raw_velocity,
        standardized_velocity=changed_values,
        relative_volume=panel.relative_volume,
        velocity_scale=panel.velocity_scale,
        minute_volume_baseline=panel.minute_volume_baseline,
        raw_velocity_valid=panel.raw_velocity_valid,
        velocity_valid=panel.velocity_valid,
        density_valid=panel.density_valid,
    )

    repeated = builder.build(changed, [np.datetime64("2024-02", "M")])[0]

    assert base.training_cutoff < np.datetime64("2024-02-01")
    np.testing.assert_allclose(base.fused_similarity, repeated.fused_similarity)
    np.testing.assert_array_equal(base.band_ids, repeated.band_ids)
