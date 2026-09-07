from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bnhmm.boundary_fk.branch_state import (
    BRANCH_FEATURE_NAMES,
    TwoBranchDailyStateBuilder,
)
from bnhmm.boundary_fk.types import NormalizedBoundaryPanel
from bnhmm.sector_flow.types import MonthlyRiverTopology


def _inputs(days: int = 26) -> tuple[NormalizedBoundaryPanel, tuple[MonthlyRiverTopology, ...]]:
    symbols = ("A", "B", "C", "D")
    sessions = np.busday_offset(np.datetime64("2024-01-02"), np.arange(days))
    timestamps = np.asarray(
        [day + np.timedelta64(minute, "m") for day in sessions for minute in (571, 572, 573)],
        dtype="datetime64[ns]",
    )
    rows = len(timestamps)
    raw = np.zeros((rows, 4), dtype=np.float64)
    for day in range(days):
        value = (day + 1) * 1e-4
        raw[day * 3 : day * 3 + 3] = (value, value * 2, -value, -value * 2)
    valid = np.ones_like(raw, dtype=np.bool_)
    panel = NormalizedBoundaryPanel(
        symbols=symbols,
        timestamps=timestamps,
        raw_velocity=raw,
        standardized_velocity=raw / 1e-4,
        relative_volume=np.ones_like(raw),
        velocity_scale=np.full_like(raw, 1e-4),
        minute_volume_baseline=np.ones_like(raw),
        raw_velocity_valid=valid,
        velocity_valid=valid,
        density_valid=valid,
    )
    months = np.unique(sessions.astype("datetime64[M]"))
    topologies = tuple(
        MonthlyRiverTopology(
            month=month,
            training_cutoff=month.astype("datetime64[ns]") - np.timedelta64(1, "ns"),
            symbols=symbols,
            assigned=np.ones(4, dtype=np.bool_),
            band_ids=np.asarray([0, 0, 1, 1]),
            row_order=np.arange(4),
            common_similarity=np.eye(4),
            residual_similarity=np.eye(4),
            fused_similarity=np.eye(4),
            graph=np.zeros((4, 4)),
            cohesion=np.ones(4),
            centrality=np.ones(4),
        )
        for month in months
    )
    return panel, topologies


def test_builder_emits_two_branch_feature_tensor_and_frozen_memberships() -> None:
    panel, topologies = _inputs()

    result = TwoBranchDailyStateBuilder(minimum_valid_minutes=3).build(panel, topologies)

    assert result.states.feature_names == BRANCH_FEATURE_NAMES
    assert result.states.values.shape == (26, 2, len(BRANCH_FEATURE_NAMES))
    assert result.states.valid.shape == (26, 2)
    assert result.memberships[0].branches == (("A", "B"), ("C", "D"))
    assert result.states.valid[-1].all()
    assert result.convection_lookbacks == (1, 2, 3, 5)
    assert result.convection_strength.shape == (26, 2, 4)
    assert result.convection_valid.shape == (26, 2, 4)


def test_builder_emits_causal_direction_features() -> None:
    panel, topologies = _inputs()

    result = TwoBranchDailyStateBuilder(minimum_valid_minutes=3).build(panel, topologies)

    assert result.states.feature_names[-4:] == (
        "signed_path_efficiency",
        "path_efficiency_delta_1d",
        "path_direction_consistency_3d",
        "signed_member_synchronization",
    )
    assert result.states.feature("signed_path_efficiency")[-1, 0] == 1.0
    assert result.states.feature("signed_path_efficiency")[-1, 1] == -1.0
    assert result.states.feature("path_efficiency_delta_1d")[-1, 0] == 0.0
    assert result.states.feature("path_efficiency_delta_1d")[-1, 1] == 0.0
    assert result.states.feature("path_direction_consistency_3d")[-1, 0] == 1.0
    assert result.states.feature("path_direction_consistency_3d")[-1, 1] == -1.0
    assert result.states.feature("signed_member_synchronization")[-1, 0] == 1.0
    assert result.states.feature("signed_member_synchronization")[-1, 1] == -1.0


def test_first_membership_does_not_require_prior_sessions_to_have_topology() -> None:
    panel, topologies = _inputs()
    sessions = np.unique(panel.timestamps.astype("datetime64[D]"))
    first = replace(
        topologies[0],
        effective_session=sessions[5],
        training_cutoff=sessions[4].astype("datetime64[ns]"),
    )

    result = TwoBranchDailyStateBuilder(
        minimum_valid_minutes=3, history_sessions=5
    ).build(
        panel,
        (first,),
    )

    assert not result.states.valid[5].any()
    assert result.states.valid[9].all()


def test_path_efficiency_uses_displacement_over_total_variation() -> None:
    panel, topologies = _inputs()
    changed_raw = panel.raw_velocity.copy()
    changed_raw[-3:, :2] = np.asarray(
        (
            (0.03, 0.03),
            (-0.01, -0.01),
            (-0.01, -0.01),
        )
    )

    result = TwoBranchDailyStateBuilder(minimum_valid_minutes=3).build(
        replace(panel, raw_velocity=changed_raw), topologies
    )

    assert result.states.feature("signed_path_efficiency")[-1, 0] == pytest.approx(0.2)


def test_convection_strength_is_joint_reverse_flow_concentration() -> None:
    panel, topologies = _inputs()
    concentrated_raw = panel.raw_velocity.copy()
    concentrated_raw[-3:, :2] = np.asarray(
        (
            (0.04, 0.02),
            (-0.02, 0.0),
            (0.0, 0.0),
        )
    )
    dispersed_raw = panel.raw_velocity.copy()
    dispersed_raw[-3:, :2] = np.asarray(
        (
            (0.04, 0.02),
            (-0.005, -0.005),
            (-0.005, -0.005),
        )
    )

    concentrated = TwoBranchDailyStateBuilder(minimum_valid_minutes=3).build(
        replace(panel, raw_velocity=concentrated_raw), topologies
    )
    dispersed = TwoBranchDailyStateBuilder(minimum_valid_minutes=3).build(
        replace(panel, raw_velocity=dispersed_raw), topologies
    )

    assert concentrated.convection_valid[-1, 0, 0]
    assert dispersed.convection_valid[-1, 0, 0]
    assert concentrated.convection_strength[-1, 0, 0] == pytest.approx(1.0)
    assert dispersed.convection_strength[-1, 0, 0] == pytest.approx(0.1)
    assert (
        concentrated.convection_strength[-1, 0, 0]
        > dispersed.convection_strength[-1, 0, 0]
    )
    assert concentrated.convection_total_flow[-1, 0, 0] == pytest.approx(0.02)
    assert dispersed.convection_total_flow[-1, 0, 0] == pytest.approx(0.02)
    assert concentrated.convection_observations[-1, 0, 0] == 6


def test_convection_windows_pool_only_trailing_completed_sessions() -> None:
    panel, topologies = _inputs()
    changed_raw = panel.raw_velocity.copy()
    changed_raw[-6:-3, :2] = np.asarray(
        (
            (0.04, 0.02),
            (-0.005, -0.005),
            (-0.005, -0.005),
        )
    )
    changed_raw[-3:, :2] = np.asarray(
        (
            (0.04, 0.02),
            (-0.02, 0.0),
            (0.0, 0.0),
        )
    )

    result = TwoBranchDailyStateBuilder(minimum_valid_minutes=3).build(
        replace(panel, raw_velocity=changed_raw), topologies
    )

    assert result.convection_strength[-1, 0, 0] == pytest.approx(1.0)
    assert result.convection_strength[-1, 0, 1] == pytest.approx(0.25)
    assert result.convection_observations[-1, 0].tolist() == [6, 12, 18, 30]


def test_convection_is_invalid_without_reverse_flow() -> None:
    panel, topologies = _inputs()

    result = TwoBranchDailyStateBuilder(minimum_valid_minutes=3).build(panel, topologies)

    assert not result.convection_valid[-1, 0].any()
    assert np.isnan(result.convection_strength[-1, 0]).all()
    assert np.all(result.convection_total_flow[-1, 0] == 0.0)


def test_convection_is_invalid_when_the_branch_has_no_net_direction() -> None:
    panel, topologies = _inputs()
    changed_raw = panel.raw_velocity.copy()
    changed_raw[-3:, :2] = np.asarray(
        (
            (0.01, 0.01),
            (-0.01, -0.01),
            (0.0, 0.0),
        )
    )

    result = TwoBranchDailyStateBuilder(minimum_valid_minutes=3).build(
        replace(panel, raw_velocity=changed_raw), topologies
    )

    assert not result.convection_valid[-1, 0, 0]
    assert np.isnan(result.convection_strength[-1, 0, 0])


def test_labels_and_member_returns_use_only_the_next_completed_session() -> None:
    panel, topologies = _inputs()
    result = TwoBranchDailyStateBuilder(minimum_valid_minutes=3).build(panel, topologies)
    row = 20

    expected_a = np.expm1(np.sum(panel.raw_velocity[(row + 1) * 3 : (row + 2) * 3, 0]))
    expected_b = np.expm1(np.sum(panel.raw_velocity[(row + 1) * 3 : (row + 2) * 3, 1]))

    assert result.labels.target_time[row] == result.states.sessions[row + 1]
    assert result.labels.next_return[row, 0] == np.mean((expected_a, expected_b))
    assert result.member_next_returns[row, 0] == expected_a
    assert result.member_next_returns[row, 1] == expected_b


def test_future_minutes_do_not_change_an_earlier_state() -> None:
    panel, topologies = _inputs()
    builder = TwoBranchDailyStateBuilder(minimum_valid_minutes=3)
    original = builder.build(panel, topologies)
    changed_raw = panel.raw_velocity.copy()
    changed_raw[-3:] = 100.0

    repeated = builder.build(replace(panel, raw_velocity=changed_raw), topologies)

    np.testing.assert_allclose(original.states.values[-2], repeated.states.values[-2])


def test_partial_member_gap_keeps_branch_day_and_marks_suspended_return_zero() -> None:
    panel, topologies = _inputs()
    raw_valid = panel.raw_velocity_valid.copy()
    velocity_valid = panel.velocity_valid.copy()
    raw_valid[21 * 3 : 22 * 3, 1] = False
    velocity_valid[21 * 3 : 22 * 3, 1] = False
    changed = replace(
        panel,
        raw_velocity_valid=raw_valid,
        velocity_valid=velocity_valid,
    )

    result = TwoBranchDailyStateBuilder(
        minimum_valid_minutes=3,
        minimum_member_fraction=0.5,
    ).build(changed, topologies)

    assert result.labels.valid[20, 0]
    assert result.member_next_valid[20, 1]
    assert result.member_next_returns[20, 1] == 0.0
