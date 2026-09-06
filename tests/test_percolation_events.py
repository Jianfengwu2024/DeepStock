from __future__ import annotations

import numpy as np

from bnhmm.percolation_branching.events import ActivationEventBuilder
from bnhmm.percolation_branching.types import ActivationConfig
from bnhmm.sector_flow.types import SectorMinutePanel


def _panel(*, sessions: int = 12, symbols: int = 3) -> SectorMinutePanel:
    timestamps = np.asarray(
        [
            np.datetime64("2025-01-01")
            + np.timedelta64(day, "D")
            + np.timedelta64(9 * 60 + 31 + minute, "m")
            for day in range(sessions)
            for minute in range(6)
        ],
        dtype="datetime64[ns]",
    )
    shape = (len(timestamps), symbols)
    close = np.ones(shape, dtype=np.float64) * 100.0
    for day in range(sessions):
        rows = slice(day * 6, (day + 1) * 6)
        path = 100.0 * np.exp(np.arange(6, dtype=np.float64)[:, None] * 0.002)
        close[rows] = path * (1.0 + np.arange(symbols) * 0.001)
    return SectorMinutePanel(
        symbols=tuple(f"S{index}" for index in range(symbols)),
        timestamps=timestamps,
        open=close.copy(),
        close=close,
        volume=np.full(shape, 100.0, dtype=np.float64),
        valid=np.ones(shape, dtype=np.bool_),
    )


def test_builder_uses_only_completed_history_for_current_session() -> None:
    panel = _panel()
    config = ActivationConfig(
        velocity_lookback_sessions=3,
        volume_lookback_sessions=3,
        minimum_history_sessions=3,
        minimum_minutes_per_session=4,
        activation_threshold=0.5,
    )
    baseline = ActivationEventBuilder(config).build(panel)
    changed_close = panel.close.copy()
    changed_volume = panel.volume.copy()
    changed_close[-6:] *= np.exp(np.arange(6)[:, None] * 0.20)
    changed_volume[-6:] *= 50.0
    changed = SectorMinutePanel(
        symbols=panel.symbols,
        timestamps=panel.timestamps,
        open=panel.open,
        close=changed_close,
        volume=changed_volume,
        valid=panel.valid,
    )
    rebuilt = ActivationEventBuilder(config).build(changed)

    np.testing.assert_allclose(
        baseline.velocity_scale[-6:], rebuilt.velocity_scale[-6:], equal_nan=True
    )
    np.testing.assert_allclose(
        baseline.volume_baseline[-6:], rebuilt.volume_baseline[-6:], equal_nan=True
    )


def test_price_sets_direction_and_volume_only_sets_mark() -> None:
    panel = _panel()
    volume = panel.volume.copy()
    volume[-5, 0] = 900.0
    enriched = SectorMinutePanel(
        symbols=panel.symbols,
        timestamps=panel.timestamps,
        open=panel.open,
        close=panel.close,
        volume=volume,
        valid=panel.valid,
    )
    events = ActivationEventBuilder(
        ActivationConfig(
            velocity_lookback_sessions=3,
            volume_lookback_sessions=3,
            minimum_history_sessions=3,
            minimum_minutes_per_session=4,
            activation_threshold=0.5,
        )
    ).build(enriched)

    assert events.bull[-5, 0]
    assert not events.bear[-5, 0]
    assert events.marks[-5, 0] > 1.0
    assert np.array_equal(events.direction("bull"), events.bull)


def test_invalid_configuration_is_rejected() -> None:
    try:
        ActivationConfig(activation_threshold=0.0)
    except ValueError as error:
        assert "threshold" in str(error)
    else:  # pragma: no cover
        raise AssertionError("zero activation threshold must fail")
