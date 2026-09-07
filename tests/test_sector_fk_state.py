from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bnhmm.boundary_fk.sector_state import SectorDailyStateBuilder
from bnhmm.boundary_fk.types import NormalizedBoundaryPanel


def _panel(days: int = 26) -> NormalizedBoundaryPanel:
    symbols = ("A", "B", "C", "D")
    sessions = np.busday_offset(np.datetime64("2024-01-02"), np.arange(days))
    timestamps = np.asarray(
        [
            day + np.timedelta64(minute, "m")
            for day in sessions
            for minute in (571, 572, 573)
        ],
        dtype="datetime64[ns]",
    )
    raw = np.zeros((len(timestamps), len(symbols)), dtype=np.float64)
    for day in range(days):
        value = (day + 1) * 1e-4
        raw[day * 3 : day * 3 + 3] = (2 * value, value, -0.5 * value, -0.5 * value)
    valid = np.ones_like(raw, dtype=np.bool_)
    return NormalizedBoundaryPanel(
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


def test_sector_builder_uses_one_whole_sector_state_without_memberships() -> None:
    result = SectorDailyStateBuilder(minimum_valid_minutes=3).build(_panel())

    assert result.states.values.shape == (26, 1, 9)
    assert result.labels.next_return.shape == (26, 1)
    assert result.memberships == ()
    assert result.states.valid[-1, 0]
    assert result.states.feature("signed_path_efficiency")[-1, 0] == pytest.approx(1.0)
    assert result.states.feature("signed_member_synchronization")[-1, 0] == pytest.approx(0.5)


def test_sector_label_is_next_day_equal_weight_constituent_return() -> None:
    panel = _panel()
    result = SectorDailyStateBuilder(minimum_valid_minutes=3).build(panel)
    row = 20
    next_rows = slice((row + 1) * 3, (row + 2) * 3)
    expected = np.mean(np.expm1(np.sum(panel.raw_velocity[next_rows], axis=0)))

    assert result.labels.target_time[row] == result.states.sessions[row + 1]
    assert result.labels.next_return[row, 0] == pytest.approx(expected)


def test_sector_state_is_causal_and_does_not_depend_on_future_minutes() -> None:
    panel = _panel()
    builder = SectorDailyStateBuilder(minimum_valid_minutes=3)
    original = builder.build(panel)
    changed = panel.raw_velocity.copy()
    changed[-3:] = 100.0

    repeated = builder.build(replace(panel, raw_velocity=changed))

    np.testing.assert_allclose(original.states.values[-2], repeated.states.values[-2])


def test_sector_reverse_flow_uses_all_constituents_in_one_field() -> None:
    panel = _panel()
    raw = panel.raw_velocity.copy()
    raw[-3:] = np.asarray(
        (
            (0.04, 0.02, -0.02, 0.0),
            (0.01, 0.01, 0.0, 0.0),
            (0.01, 0.01, 0.0, 0.0),
        )
    )

    result = SectorDailyStateBuilder(minimum_valid_minutes=3).build(
        replace(panel, raw_velocity=raw)
    )

    assert result.convection_valid[-1, 0, 0]
    assert result.convection_total_flow[-1, 0, 0] == pytest.approx(0.02)
    assert result.convection_observations[-1, 0, 0] == 12
