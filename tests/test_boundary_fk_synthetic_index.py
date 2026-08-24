from __future__ import annotations

import numpy as np
import pytest

from bnhmm.boundary_fk.synthetic_index import SyntheticSectorIndexBuilder
from bnhmm.data.source import MarketCapRecord
from bnhmm.sector_flow.types import SectorMinutePanel


def _panel(
    *,
    timestamps: list[str],
    closes: list[list[float]],
    valid: list[list[bool]] | None = None,
    symbols: tuple[str, ...] = ("A", "B", "C", "NO_MINUTE"),
) -> SectorMinutePanel:
    close = np.asarray(closes, dtype=np.float64)
    mask = np.asarray(valid, dtype=np.bool_) if valid is not None else np.isfinite(close)
    open_ = np.where(mask, close, np.nan)
    volume = np.where(mask, 1.0, np.nan)
    return SectorMinutePanel(
        symbols=symbols,
        timestamps=np.asarray(timestamps, dtype="datetime64[ns]"),
        open=open_,
        close=close,
        volume=volume,
        valid=mask,
    )


def _caps(*rows: tuple[str, str, float]) -> tuple[MarketCapRecord, ...]:
    return tuple(
        MarketCapRecord(symbol=symbol, session=np.datetime64(session), circ_mv=value)
        for symbol, session, value in rows
    )


def test_removes_constituents_without_any_minute_bars() -> None:
    panel = _panel(
        timestamps=[
            "2023-12-29T09:31",
            "2023-12-29T09:32",
            "2024-01-02T09:31",
            "2024-01-02T09:32",
        ],
        closes=[
            [10.0, 20.0, 30.0, np.nan],
            [10.0, 20.0, 30.0, np.nan],
            [10.0, 20.0, 30.0, np.nan],
            [10.1, 20.2, 30.3, np.nan],
        ],
    )
    caps = _caps(
        ("A", "2023-12-29", 10.0),
        ("B", "2023-12-29", 30.0),
        ("C", "2023-12-29", 60.0),
        ("NO_MINUTE", "2023-12-29", 900.0),
    )

    index = SyntheticSectorIndexBuilder().build(panel, caps)

    assert index.excluded_symbols == ("NO_MINUTE",)
    assert index.weights_for("2024-01").symbols == ("A", "B", "C")


def test_uses_prior_month_end_caps_and_freezes_weights_for_the_month() -> None:
    panel = _panel(
        timestamps=[
            "2023-12-29T09:31",
            "2023-12-29T09:32",
            "2024-01-02T09:31",
            "2024-01-02T09:32",
            "2024-01-31T09:31",
            "2024-01-31T09:32",
            "2024-02-01T09:31",
            "2024-02-01T09:32",
        ],
        closes=[
            [10.0, 20.0, 30.0, np.nan],
            [10.0, 20.0, 30.0, np.nan],
            [10.0, 20.0, 30.0, np.nan],
            [10.1, 20.2, 30.3, np.nan],
            [11.0, 21.0, 31.0, np.nan],
            [11.1, 21.1, 31.1, np.nan],
            [12.0, 22.0, 32.0, np.nan],
            [12.1, 22.1, 32.1, np.nan],
        ],
    )
    caps = _caps(
        ("A", "2023-12-20", 5.0),
        ("A", "2023-12-29", 10.0),
        ("B", "2023-12-29", 30.0),
        ("C", "2023-12-29", 60.0),
        ("A", "2024-01-31", 20.0),
        ("B", "2024-01-31", 30.0),
        ("C", "2024-01-31", 50.0),
    )

    index = SyntheticSectorIndexBuilder().build(panel, caps)
    january = index.weights_for("2024-01")
    february = index.weights_for("2024-02")

    assert january.weight_date == np.datetime64("2023-12-29")
    np.testing.assert_allclose(january.weights, [0.1, 0.3, 0.6])
    np.testing.assert_allclose(index.weights_at("2024-01-02"), january.weights)
    np.testing.assert_allclose(index.weights_at("2024-01-31"), january.weights)
    np.testing.assert_allclose(february.weights, [0.2, 0.3, 0.5])


def test_future_market_caps_cannot_change_earlier_index_values() -> None:
    panel = _panel(
        timestamps=[
            "2023-12-29T09:31",
            "2023-12-29T09:32",
            "2024-01-02T09:31",
            "2024-01-02T09:32",
        ],
        closes=[
            [10.0, 20.0, 30.0, np.nan],
            [10.0, 20.0, 30.0, np.nan],
            [10.0, 20.0, 30.0, np.nan],
            [11.0, 18.0, 33.0, np.nan],
        ],
    )
    historical = _caps(
        ("A", "2023-12-29", 10.0),
        ("B", "2023-12-29", 30.0),
        ("C", "2023-12-29", 60.0),
    )
    future_shock = historical + _caps(
        ("A", "2024-12-31", 1_000_000.0),
        ("B", "2024-12-31", 1.0),
        ("C", "2024-12-31", 1.0),
    )

    base = SyntheticSectorIndexBuilder().build(panel, historical)
    changed = SyntheticSectorIndexBuilder().build(panel, future_shock)

    np.testing.assert_allclose(base.velocity, changed.velocity, equal_nan=True)
    assert np.array_equal(base.valid, changed.valid)


def test_missing_minute_returns_are_renormalized_instead_of_filled_with_zero() -> None:
    panel = _panel(
        timestamps=[
            "2023-12-29T09:31",
            "2023-12-29T09:32",
            "2024-01-02T09:31",
            "2024-01-02T09:32",
        ],
        closes=[
            [10.0, 20.0, 30.0, np.nan],
            [10.0, 20.0, 30.0, np.nan],
            [10.0, 20.0, 30.0, np.nan],
            [11.0, np.nan, np.nan, np.nan],
        ],
    )
    caps = _caps(
        ("A", "2023-12-29", 20.0),
        ("B", "2023-12-29", 30.0),
        ("C", "2023-12-29", 50.0),
    )

    index = SyntheticSectorIndexBuilder(
        min_active_weight=0.1,
        min_active_fraction=0.3,
    ).build(panel, caps)

    assert index.valid[-1]
    assert index.active_weight[-1] == pytest.approx(0.2)
    assert index.active_fraction[-1] == pytest.approx(1.0 / 3.0)
    assert index.velocity[-1] == pytest.approx(np.log(11.0 / 10.0))


def test_minute_is_invalid_when_active_coverage_is_too_low() -> None:
    panel = _panel(
        timestamps=[
            "2023-12-29T09:31",
            "2023-12-29T09:32",
            "2024-01-02T09:31",
            "2024-01-02T09:32",
        ],
        closes=[
            [10.0, 20.0, 30.0, np.nan],
            [10.0, 20.0, 30.0, np.nan],
            [10.0, 20.0, 30.0, np.nan],
            [11.0, np.nan, np.nan, np.nan],
        ],
    )
    caps = _caps(
        ("A", "2023-12-29", 20.0),
        ("B", "2023-12-29", 30.0),
        ("C", "2023-12-29", 50.0),
    )

    index = SyntheticSectorIndexBuilder(
        min_active_weight=0.5,
        min_active_fraction=0.5,
    ).build(panel, caps)

    assert not index.valid[-1]
    assert np.isnan(index.velocity[-1])


def test_daily_open_to_close_return_uses_the_same_frozen_weights() -> None:
    panel = _panel(
        timestamps=[
            "2023-12-29T09:31",
            "2023-12-29T15:00",
            "2024-01-02T09:31",
            "2024-01-02T15:00",
        ],
        closes=[
            [10.0, 20.0, 30.0, np.nan],
            [10.0, 20.0, 30.0, np.nan],
            [10.0, 20.0, 30.0, np.nan],
            [11.0, 18.0, 33.0, np.nan],
        ],
    )
    caps = _caps(
        ("A", "2023-12-29", 10.0),
        ("B", "2023-12-29", 30.0),
        ("C", "2023-12-29", 60.0),
    )

    index = SyntheticSectorIndexBuilder().build(panel, caps)
    row = int(np.flatnonzero(index.daily_sessions == np.datetime64("2024-01-02"))[0])
    expected = (
        0.1 * np.log(11.0 / 10.0)
        + 0.3 * np.log(18.0 / 20.0)
        + 0.6 * np.log(33.0 / 30.0)
    )

    assert index.daily_valid[row]
    assert index.daily_open_to_close_return[row] == pytest.approx(expected)


def test_month_without_prior_positive_market_caps_fails_without_equal_weighting() -> None:
    panel = _panel(
        timestamps=[
            "2023-12-29T09:31",
            "2023-12-29T09:32",
            "2024-01-02T09:31",
            "2024-01-02T09:32",
        ],
        closes=[
            [10.0, 20.0, 30.0, np.nan],
            [10.0, 20.0, 30.0, np.nan],
            [10.0, 20.0, 30.0, np.nan],
            [11.0, 21.0, 31.0, np.nan],
        ],
    )

    with pytest.raises(ValueError, match="上月末.*流通市值"):
        SyntheticSectorIndexBuilder().build(panel, ())
