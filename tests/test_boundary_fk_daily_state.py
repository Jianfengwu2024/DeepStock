from __future__ import annotations

import numpy as np

from bnhmm.boundary_fk.daily_state import (
    BoundaryDailyStateBuilder,
    BoundaryTopologyBuilder,
)
from bnhmm.boundary_fk.types import (
    BoundaryMinuteMoments,
    MonthlyIndexWeights,
    NormalizedBoundaryPanel,
    SyntheticSectorIndex,
)
from bnhmm.sector_flow.types import MonthlyRiverTopology
from bnhmm.sector_flow.topology import TopologyConfig


def _inputs(
    *, band_ids: list[int] | None = None
) -> tuple[
    NormalizedBoundaryPanel,
    BoundaryMinuteMoments,
    SyntheticSectorIndex,
    tuple[MonthlyRiverTopology, ...],
]:
    symbols = ("A", "B", "C")
    timestamps = np.asarray(
        [
            "2024-01-02T09:31",
            "2024-01-02T09:32",
            "2024-01-03T09:31",
            "2024-01-03T09:32",
        ],
        dtype="datetime64[ns]",
    )
    raw = np.asarray(
        [[0.01, -0.01, 0.005], [0.02, -0.005, 0.01]] * 2,
        dtype=np.float64,
    )
    standardized = raw * 100.0
    density = np.asarray([[2.0, 1.0, 3.0]] * 4)
    valid_matrix = np.ones(raw.shape, dtype=np.bool_)
    normalized = NormalizedBoundaryPanel(
        symbols=symbols,
        timestamps=timestamps,
        raw_velocity=raw,
        standardized_velocity=standardized,
        relative_volume=density,
        velocity_scale=np.full(raw.shape, 0.01),
        minute_volume_baseline=np.ones(raw.shape),
        raw_velocity_valid=valid_matrix,
        velocity_valid=valid_matrix,
        density_valid=valid_matrix,
    )
    rows = len(timestamps)
    sequence = np.arange(1, rows + 1, dtype=np.float64)
    moments = BoundaryMinuteMoments(
        timestamps=timestamps,
        water=sequence,
        momentum=sequence * 0.01,
        velocity=sequence * 0.001,
        mean_standardized_velocity=sequence * 0.1,
        coherence=np.asarray([0.9, 0.7, 0.95, 0.85]),
        turbulence=np.asarray([0.2, 1.2, 0.1, 1.5]),
        breadth=np.asarray([0.5, 0.2, 0.8, 0.4]),
        concentration=np.asarray([0.4, 0.5, 0.35, 0.45]),
        effective_constituents=np.asarray([2.5, 2.0, 2.8, 2.2]),
        decoupling=np.asarray([0.1, 0.2, -0.1, 0.3]),
        thickness=np.asarray([0.5, 0.8, 0.4, 1.0]),
        thickness_slope=np.asarray([0.0, 0.3, -0.1, 0.6]),
        viscosity=np.asarray([1.0, 1.1, 0.9, 1.2]),
        active_weight=np.ones(rows),
        active_fraction=np.ones(rows),
        valid=np.ones(rows, dtype=np.bool_),
    )
    frozen = MonthlyIndexWeights(
        month=np.datetime64("2024-01", "M"),
        weight_date=np.datetime64("2023-12-29"),
        symbols=symbols,
        weights=np.asarray([0.2, 0.3, 0.5]),
        training_cutoff=np.datetime64("2023-12-29"),
        excluded_symbols=(),
    )
    index = SyntheticSectorIndex(
        timestamps=timestamps,
        velocity=np.asarray([0.001, 0.002, 0.001, 0.002]),
        valid=np.ones(rows, dtype=np.bool_),
        active_weight=np.ones(rows),
        active_fraction=np.ones(rows),
        monthly_weights=(frozen,),
        daily_sessions=np.asarray(["2024-01-02", "2024-01-03"], dtype="datetime64[D]"),
        daily_open_to_close_return=np.asarray([0.01, 0.02]),
        daily_valid=np.ones(2, dtype=np.bool_),
        daily_active_weight=np.ones(2),
        daily_active_fraction=np.ones(2),
        excluded_symbols=(),
    )
    labels = np.asarray(band_ids or [0, 1, 2], dtype=np.int64)
    topology = MonthlyRiverTopology(
        month=np.datetime64("2024-01", "M"),
        training_cutoff=np.datetime64("2023-12-29T15:00"),
        symbols=symbols,
        assigned=np.ones(3, dtype=np.bool_),
        band_ids=labels,
        row_order=np.arange(3, dtype=np.int64),
        common_similarity=np.eye(3),
        residual_similarity=np.eye(3),
        fused_similarity=np.eye(3),
        graph=np.zeros((3, 3)),
        cohesion=np.ones(3),
        centrality=np.ones(3),
    )
    return normalized, moments, index, (topology,)


def test_daily_schema_is_named_unique_and_bounded() -> None:
    normalized, moments, index, topologies = _inputs()

    states = BoundaryDailyStateBuilder(branch_count=3).build(
        normalized,
        moments,
        index,
        topologies,
        high_turbulence_threshold=1.0,
    )

    assert len(states.feature_names) <= 36
    assert len(states.feature_names) == len(set(states.feature_names))
    assert len(states.feature_names) == len(states.units)
    assert all(name and unit for name, unit in zip(states.feature_names, states.units))
    assert states.values.shape == (2, len(states.feature_names))
    assert states.valid.all()


def test_sessions_in_one_month_share_one_frozen_topology() -> None:
    normalized, moments, index, topologies = _inputs()

    states = BoundaryDailyStateBuilder(branch_count=3).build(
        normalized,
        moments,
        index,
        topologies,
        high_turbulence_threshold=1.0,
    )

    assert np.all(states.topology_months == np.datetime64("2024-01", "M"))
    assert not states.branch_degraded.any()


def test_stable_branches_add_three_named_features_per_branch() -> None:
    normalized, moments, index, topologies = _inputs()

    states = BoundaryDailyStateBuilder(branch_count=3).build(
        normalized,
        moments,
        index,
        topologies,
        high_turbulence_threshold=1.0,
    )

    branch_names = [name for name in states.feature_names if name.startswith("branch_")]
    assert len(branch_names) == 9
    assert np.isfinite(states.values[:, -9:]).all()


def test_insufficient_branches_fall_back_to_main_flow_without_fake_slots() -> None:
    normalized, moments, index, topologies = _inputs(band_ids=[0, 0, 1])

    states = BoundaryDailyStateBuilder(branch_count=3).build(
        normalized,
        moments,
        index,
        topologies,
        high_turbulence_threshold=1.0,
    )

    assert states.branch_degraded.all()
    assert all(not name.startswith("branch_") for name in states.feature_names)
    assert states.values.shape[1] == 24


def test_future_day_changes_do_not_rewrite_prior_daily_state() -> None:
    normalized, moments, index, topologies = _inputs()
    base = BoundaryDailyStateBuilder(branch_count=3).build(
        normalized,
        moments,
        index,
        topologies,
        high_turbulence_threshold=1.0,
    )
    changed_water = moments.water.copy()
    changed_water[2:] = 1_000_000.0
    changed = BoundaryMinuteMoments(
        **{
            name: getattr(moments, name)
            for name in moments.__dataclass_fields__
            if name != "water"
        },
        water=changed_water,
    )

    repeated = BoundaryDailyStateBuilder(branch_count=3).build(
        normalized,
        changed,
        index,
        topologies,
        high_turbulence_threshold=1.0,
    )

    np.testing.assert_allclose(base.values[0], repeated.values[0])


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
        **{
            name: getattr(panel, name)
            for name in panel.__dataclass_fields__
            if name != "standardized_velocity"
        },
        standardized_velocity=changed_values,
    )

    repeated = builder.build(changed, [np.datetime64("2024-02", "M")])[0]

    assert base.training_cutoff < np.datetime64("2024-02-01")
    np.testing.assert_allclose(base.fused_similarity, repeated.fused_similarity)
    np.testing.assert_array_equal(base.band_ids, repeated.band_ids)
