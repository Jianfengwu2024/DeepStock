from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bnhmm.boundary_fk.sector_state import SectorDailyStateBuilder
from bnhmm.boundary_fk.types import NormalizedBoundaryPanel


def _panel(days: int = 62) -> NormalizedBoundaryPanel:
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
    close = np.zeros_like(raw)
    for day in range(days):
        value = (day + 1) * 1e-4
        raw[day * 3 : day * 3 + 3] = (2 * value, value, -0.5 * value, -0.5 * value)
        prior = np.full(len(symbols), 10.0) if day == 0 else close[day * 3 - 1]
        close[day * 3] = prior
        close[day * 3 + 1] = prior * np.exp(raw[day * 3 + 1])
        close[day * 3 + 2] = close[day * 3 + 1] * np.exp(raw[day * 3 + 2])
    valid = np.ones_like(raw, dtype=np.bool_)
    return NormalizedBoundaryPanel(
        symbols=symbols,
        timestamps=timestamps,
        close=close,
        price_valid=valid,
        raw_velocity=raw,
        standardized_velocity=raw / 1e-4,
        relative_volume=np.ones_like(raw),
        velocity_scale=np.full_like(raw, 1e-4),
        minute_volume_baseline=np.ones_like(raw),
        raw_velocity_valid=valid,
        velocity_valid=valid,
        density_valid=valid,
    )


def _continuous_panel(days: int = 62) -> NormalizedBoundaryPanel:
    symbols = ("A", "B", "C", "D")
    sessions = np.busday_offset(np.datetime64("2024-01-02"), np.arange(days))
    timestamps = np.asarray(
        [
            day + np.timedelta64(minute, "m")
            for day in sessions
            for minute in (571, 572)
        ],
        dtype="datetime64[ns]",
    )
    base = np.asarray((10.0, 20.0, 30.0, 40.0), dtype=np.float64)
    close = np.vstack(
        [
            base * np.exp(day * 0.01 + minute_offset)
            for day in range(days)
            for minute_offset in (0.0, 0.002)
        ]
    )
    raw = np.full_like(close, np.nan)
    raw[1::2] = 0.002
    valid = np.ones_like(close, dtype=np.bool_)
    raw_valid = np.zeros_like(valid)
    raw_valid[1::2] = True
    return NormalizedBoundaryPanel(
        symbols=symbols,
        timestamps=timestamps,
        close=close,
        price_valid=valid,
        raw_velocity=raw,
        standardized_velocity=np.where(raw_valid, raw / 1e-4, np.nan),
        relative_volume=np.ones_like(close),
        velocity_scale=np.full_like(close, 1e-4),
        minute_volume_baseline=np.ones_like(close),
        raw_velocity_valid=raw_valid,
        velocity_valid=raw_valid,
        density_valid=valid,
    )


def test_sector_builder_uses_one_whole_sector_state_without_memberships() -> None:
    result = SectorDailyStateBuilder(minimum_valid_minutes=3).build(_panel())

    assert result.states.values.shape == (62, 1, 4)
    assert result.labels.next_return.shape == (62, 1)
    assert result.memberships == ()
    assert result.states.valid[-1, 0]
    assert np.isfinite(result.states.feature("rolling_log_position_60d")[-1, 0])
    assert np.isfinite(result.states.feature("rolling_path_efficiency_60d")[-1, 0])


def test_sector_label_is_next_day_equal_weight_constituent_return() -> None:
    panel = _panel()
    result = SectorDailyStateBuilder(minimum_valid_minutes=3).build(panel)
    row = 60
    current_close = panel.close[row * 3 + 2]
    next_close = panel.close[(row + 1) * 3 + 2]
    expected = np.mean(next_close / current_close - 1.0)

    assert result.labels.target_time[row] == result.states.sessions[row + 1]
    assert result.labels.next_return[row, 0] == pytest.approx(expected)


def test_sector_state_is_causal_and_does_not_depend_on_future_minutes() -> None:
    panel = _panel()
    builder = SectorDailyStateBuilder(minimum_valid_minutes=3)
    original = builder.build(panel)
    changed = panel.close.copy()
    changed[-3:] *= 3.0

    repeated = builder.build(replace(panel, close=changed))

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


def test_sector_state_uses_one_previous_close_anchor_across_60_sessions() -> None:
    panel = _continuous_panel()

    result = SectorDailyStateBuilder(
        minimum_valid_minutes=2,
        history_sessions=60,
    ).build(panel)

    signal_row = 60
    assert result.states.valid[signal_row, 0]
    assert result.states.feature("rolling_log_position_60d")[signal_row, 0] == pytest.approx(
        0.6
    )
    assert result.states.feature("rolling_path_efficiency_60d")[signal_row, 0] == pytest.approx(
        1.0
    )
    assert result.labels.next_return[signal_row, 0] == pytest.approx(np.expm1(0.01))


def test_sector_rolling_state_does_not_depend_on_the_next_session() -> None:
    panel = _continuous_panel()
    builder = SectorDailyStateBuilder(minimum_valid_minutes=2, history_sessions=60)
    original = builder.build(panel)
    changed_close = panel.close.copy()
    changed_close[-2:] *= 3.0

    repeated = builder.build(replace(panel, close=changed_close))

    np.testing.assert_allclose(
        original.states.values[60],
        repeated.states.values[60],
        equal_nan=True,
    )
