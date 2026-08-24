from __future__ import annotations

import numpy as np
import pytest

from bnhmm.boundary_fk.normalization import (
    BoundaryNormalizationConfig,
    CausalBoundaryNormalizer,
)
from bnhmm.sector_flow.types import SectorMinutePanel


def _history_panel(
    *,
    current_return: float = 0.02,
    current_volume: float = 300.0,
) -> SectorMinutePanel:
    timestamps: list[np.datetime64] = []
    closes: list[list[float]] = []
    volumes: list[list[float]] = []
    prices = np.asarray([10.0, 20.0, 30.0])
    for session_index, day in enumerate(
        ("2025-01-02", "2025-01-03", "2025-01-06")
    ):
        timestamps.extend(
            (
                np.datetime64(f"{day}T09:31"),
                np.datetime64(f"{day}T09:32"),
                np.datetime64(f"{day}T09:33"),
            )
        )
        first_return = current_return if session_index == 2 else 0.01
        day_prices = np.vstack(
            (
                prices,
                prices * np.exp(first_return),
                prices * np.exp(first_return + 0.02),
            )
        )
        closes.extend(day_prices.tolist())
        volume = current_volume if session_index == 2 else 240.0
        volumes.extend(np.full((3, 3), volume).tolist())
        prices = day_prices[-1]
    close = np.asarray(closes, dtype=np.float64)
    return SectorMinutePanel(
        symbols=("A", "B", "C"),
        timestamps=np.asarray(timestamps, dtype="datetime64[ns]"),
        open=close.copy(),
        close=close,
        volume=np.asarray(volumes, dtype=np.float64),
        valid=np.ones(close.shape, dtype=np.bool_),
    )


def _normalizer() -> CausalBoundaryNormalizer:
    return CausalBoundaryNormalizer(
        BoundaryNormalizationConfig(
            lookback_sessions=2,
            minimum_history_sessions=2,
            minutes_per_session=240,
            minimum_session_observations=2,
            minimum_velocity_scale=1e-8,
        )
    )


def test_current_session_does_not_change_its_own_baselines() -> None:
    base = _normalizer().transform(_history_panel())
    shocked = _normalizer().transform(
        _history_panel(current_return=0.20, current_volume=30_000.0)
    )
    current = base.timestamps.astype("datetime64[D]") == np.datetime64("2025-01-06")

    np.testing.assert_allclose(
        base.velocity_scale[current], shocked.velocity_scale[current], equal_nan=True
    )
    np.testing.assert_allclose(
        base.minute_volume_baseline[current],
        shocked.minute_volume_baseline[current],
        equal_nan=True,
    )


def test_velocity_scale_is_prior_session_all_minute_rms() -> None:
    normalized = _normalizer().transform(_history_panel())
    row = int(
        np.flatnonzero(
            normalized.timestamps.astype("datetime64[m]")
            == np.datetime64("2025-01-06T09:32")
        )[0]
    )
    expected_scale = np.sqrt((0.01**2 + 0.02**2 + 0.01**2 + 0.02**2) / 4.0)

    assert normalized.velocity_scale[row, 0] == pytest.approx(expected_scale)
    assert normalized.standardized_velocity[row, 0] == pytest.approx(
        0.02 / expected_scale
    )


def test_relative_volume_uses_prior_mean_daily_volume_divided_by_240() -> None:
    normalized = _normalizer().transform(_history_panel(current_volume=300.0))
    row = int(
        np.flatnonzero(
            normalized.timestamps.astype("datetime64[m]")
            == np.datetime64("2025-01-06T09:31")
        )[0]
    )
    # Each compact synthetic history day has total volume 720. The model's
    # explicit 240-minute clock therefore yields a baseline of 3 per minute.
    assert normalized.minute_volume_baseline[row, 0] == pytest.approx(3.0)
    assert normalized.relative_volume[row, 0] == pytest.approx(100.0)


def test_returns_do_not_bridge_sessions_or_lunch_breaks() -> None:
    timestamps = np.asarray(
        [
            "2025-01-02T11:30",
            "2025-01-02T13:01",
            "2025-01-03T09:31",
        ],
        dtype="datetime64[ns]",
    )
    close = np.asarray([[10.0, 20.0, 30.0], [11.0, 22.0, 33.0], [12.0, 24.0, 36.0]])
    panel = SectorMinutePanel(
        symbols=("A", "B", "C"),
        timestamps=timestamps,
        open=close.copy(),
        close=close,
        volume=np.ones(close.shape),
        valid=np.ones(close.shape, dtype=np.bool_),
    )

    normalized = CausalBoundaryNormalizer(
        BoundaryNormalizationConfig(
            lookback_sessions=2,
            minimum_history_sessions=1,
            minimum_session_observations=1,
        )
    ).transform(panel)

    assert not normalized.raw_velocity_valid.any()
    assert np.isnan(normalized.raw_velocity).all()
