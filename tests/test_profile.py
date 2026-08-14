from __future__ import annotations

import unittest
from datetime import date, timedelta

import numpy as np

from bnhmm import BarBatch, PriceVolumeProfileBuilder, ProfileConfig


def synthetic_sessions(session_count: int = 2) -> BarBatch:
    timestamps: list[np.datetime64] = []
    prices: list[float] = []
    volumes: list[float] = []
    current = date(2026, 1, 5)
    for session_index in range(session_count):
        day = current + timedelta(days=session_index)
        for start_hour, start_minute in ((9, 30), (13, 0)):
            start = np.datetime64(f"{day.isoformat()}T{start_hour:02d}:{start_minute:02d}")
            for minute in range(1, 121):
                timestamps.append(start + np.timedelta64(minute, "m"))
                prices.append(10.0 + 0.001 * (session_index * 240 + len(prices) % 240))
                volumes.append(1000.0 + 10.0 * (minute % 15))
    close = np.asarray(prices, dtype=np.float64)
    return BarBatch(
        symbol="000001",
        timestamp=np.asarray(timestamps, dtype="datetime64[ns]"),
        open=close - 0.005,
        high=close + 0.01,
        low=close - 0.01,
        close=close,
        volume=np.asarray(volumes, dtype=np.float64),
    )


class ProfileBuilderTests(unittest.TestCase):
    def test_default_builds_one_full_minute_profile_per_session(self) -> None:
        states = PriceVolumeProfileBuilder().build(synthetic_sessions(12))

        self.assertEqual(len(states), 2)
        self.assertTrue(all(item.timeframe == "1d" for item in states))
        self.assertTrue(all(item.bar_count == 240 for item in states))
        self.assertTrue(
            all(
                str(item.period_end).endswith("15:00:00.000000000")
                for item in states
            )
        )

    def test_builds_four_hour_states_and_one_day_state_per_session(self) -> None:
        config = ProfileConfig(
            state_timeframes=("60m", "1d"),
            relative_volume_min_observations=1,
        )
        states = PriceVolumeProfileBuilder(config).build(synthetic_sessions(3))
        self.assertEqual(sum(item.timeframe == "60m" for item in states), 8)
        self.assertEqual(sum(item.timeframe == "1d" for item in states), 2)
        for state in states:
            bins = len(state.distribution)
            self.assertGreaterEqual(bins, 3)
            self.assertEqual(state.price_distribution.shape, (bins,))
            self.assertEqual(state.relative_volume_intensity.shape, (bins,))
            self.assertEqual(state.normalized_relative_volume_level.shape, (bins,))
            self.assertEqual(state.price_velocity_distribution.shape, (bins, 5))
            self.assertEqual(state.volume_velocity_distribution.shape, (bins, 5))
            self.assertEqual(state.bin_coordinates.shape, (bins,))
            self.assertEqual(state.normalized_bin_coordinates.shape, (bins,))
            self.assertEqual(state.adjacent_pairs.shape, (bins - 1, 2))
            self.assertEqual(state.adjacent_triples.shape, (bins - 2, 3))
            self.assertAlmostEqual(float(state.distribution.sum()), 1.0)
            self.assertAlmostEqual(float(state.price_distribution.sum()), 1.0)
            np.testing.assert_allclose(np.diff(state.bin_coordinates), 1.0)
            self.assertAlmostEqual(float(state.price_velocity_distribution.sum()), 1.0)
            self.assertAlmostEqual(float(state.volume_velocity_distribution.sum()), 1.0)
            self.assertLess(float(state.price_velocity_distribution.sum(axis=1).max()), 1.0)
            self.assertLess(float(state.volume_velocity_distribution.sum(axis=1).max()), 1.0)
            self.assertGreater(state.first_bar_time, state.period_start)
            self.assertGreater(state.start_price, 0)
            self.assertGreater(state.end_price, 0)

    def test_histogram_deposit_conserves_weight(self) -> None:
        histogram = np.zeros(8, dtype=np.float64)
        edges = np.linspace(-2, 2, 9)
        PriceVolumeProfileBuilder._deposit(histogram, edges, -0.75, 0.75, 7.5)
        self.assertAlmostEqual(float(histogram.sum()), 7.5)

    def test_auction_boundary_row_is_excluded_from_daily_state(self) -> None:
        bars = synthetic_sessions(12)
        extra_timestamp = np.asarray(["2026-01-05T09:30:00"], dtype="datetime64[ns]")
        timestamps = np.concatenate([extra_timestamp, bars.timestamp])
        extra = np.asarray([10.0])
        extended = BarBatch(
            symbol=bars.symbol,
            timestamp=timestamps,
            open=np.concatenate([extra, bars.open]),
            high=np.concatenate([extra, bars.high]),
            low=np.concatenate([extra, bars.low]),
            close=np.concatenate([extra, bars.close]),
            volume=np.concatenate([np.asarray([999999.0]), bars.volume]),
        )
        states = PriceVolumeProfileBuilder(ProfileConfig()).build(extended)
        daily = next(state for state in states if state.timeframe == "1d")
        self.assertEqual(daily.bar_count, 240)
        self.assertNotEqual(daily.total_raw_volume, float(extended.volume.sum()))

    def test_small_minute_gap_does_not_discard_the_daily_state(self) -> None:
        bars = synthetic_sessions(12)
        incomplete = bars.take(np.arange(len(bars) - 1))

        states = PriceVolumeProfileBuilder().build(incomplete)

        self.assertEqual(len(states), 2)
        self.assertEqual(states[-1].bar_count, 239)
        self.assertEqual(
            states[-1].period_end,
            np.datetime64("2026-01-16T15:00:00", "ns"),
        )

    def test_prices_below_previous_close_use_negative_coordinates(self) -> None:
        bars = synthetic_sessions(11)
        sessions = bars.timestamp.astype("datetime64[D]")
        adjustment = np.where(sessions == np.unique(sessions)[-1], -1.0, 0.0)
        shifted = BarBatch(
            symbol=bars.symbol,
            timestamp=bars.timestamp,
            open=bars.open + adjustment,
            high=bars.high + adjustment,
            low=bars.low + adjustment,
            close=bars.close + adjustment,
            volume=bars.volume,
        )

        states = PriceVolumeProfileBuilder().build(shifted)

        self.assertEqual(len(states), 1)
        self.assertLess(float(states[0].bin_coordinates.max()), 0.0)
        self.assertAlmostEqual(states[0].log_bin_width, np.log1p(0.001))

    def test_current_session_cannot_change_its_rolling_normalization(self) -> None:
        bars = synthetic_sessions(11)
        sessions = bars.timestamp.astype("datetime64[D]")
        last_session = np.unique(sessions)[-1]
        changed_volume = bars.volume.copy()
        changed_volume[sessions == last_session] *= 100.0
        changed = BarBatch(
            symbol=bars.symbol,
            timestamp=bars.timestamp,
            open=bars.open,
            high=bars.high,
            low=bars.low,
            close=bars.close,
            volume=changed_volume,
        )

        original_state = PriceVolumeProfileBuilder().build(bars)[0]
        changed_state = PriceVolumeProfileBuilder().build(changed)[0]

        self.assertEqual(original_state.normalization_sessions, 10)
        self.assertAlmostEqual(
            original_state.average_minute_volume,
            changed_state.average_minute_volume,
        )
        self.assertAlmostEqual(original_state.location_scale, changed_state.location_scale)
        self.assertGreater(
            changed_state.total_relative_volume,
            original_state.total_relative_volume,
        )


if __name__ == "__main__":
    unittest.main()
