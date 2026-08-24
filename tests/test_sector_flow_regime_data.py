from __future__ import annotations

import unittest

import numpy as np

from bnhmm import BarBatch
from bnhmm.sector_flow.regime_data import (
    SectorRegimeDataConfig,
    SectorRegimeDatasetBuilder,
    SectorRegimeState,
    build_sector_state_image,
)
from bnhmm.sector_flow.types import DailyRiverSlice


def river_slice(day: str, offset: float) -> DailyRiverSlice:
    timestamps = np.asarray(
        [f"{day}T09:31", f"{day}T09:32", f"{day}T09:33", f"{day}T09:34"],
        dtype="datetime64[m]",
    )
    velocity = np.asarray(
        [[offset + 1.0, 2.0, np.nan, 4.0], [-1.0, offset, 1.0, 2.0]],
        dtype=np.float64,
    )
    valid = np.isfinite(velocity)
    return DailyRiverSlice(
        session=np.datetime64(day, "D"),
        topology_month=np.datetime64(day[:7], "M"),
        symbols=("B", "A"),
        timestamps=timestamps,
        velocity=velocity,
        residual_velocity=np.where(valid, velocity * 0.5, np.nan),
        activity=np.where(valid, 1.0 + abs(offset), np.nan),
        valid=valid,
    )


def index_bars(target_close: float = 103.0) -> BarBatch:
    dates = np.asarray(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
        dtype="datetime64[D]",
    )
    opens = np.full(4, 100.0)
    closes = np.asarray([101.0, 99.0, target_close, 100.5])
    return BarBatch(
        symbol="881121",
        timestamp=dates.astype("datetime64[ns]"),
        open=opens,
        high=np.maximum(opens, closes) + 1.0,
        low=np.minimum(opens, closes) - 1.0,
        close=closes,
        volume=np.ones(4),
    )


class SectorRegimeDataTests(unittest.TestCase):
    def test_image_has_fixed_channels_and_preserves_missing_mask(self) -> None:
        image = build_sector_state_image(
            river_slice("2025-01-02", 0.0), height=3, minutes=4
        )

        self.assertEqual(image.values.shape, (4, 3, 4))
        self.assertEqual(image.symbols, ("B", "A"))
        self.assertEqual(float(image.values[3, 0, 2]), 0.0)
        self.assertTrue(np.all(image.values[:, 2] == 0.0))
        self.assertTrue(np.all(np.isfinite(image.values)))

    def test_builder_uses_short_context_and_next_session_image(self) -> None:
        slices = tuple(
            river_slice(day, float(index))
            for index, day in enumerate(
                ("2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07")
            )
        )
        config = SectorRegimeDataConfig(
            context_sessions=2,
            image_height=3,
            image_minutes=4,
            volatility_lookback_sessions=3,
            volatility_minimum_observations=2,
            neutral_sigma=0.5,
            minimum_return_scale=0.001,
        )

        samples = SectorRegimeDatasetBuilder(config).build(slices, index_bars())

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].images.shape, (2, 4, 3, 4))
        self.assertEqual(samples[0].next_image.shape, (4, 3, 4))
        self.assertEqual(samples[0].signal_time, slices[1].timestamps[-1])
        self.assertEqual(samples[0].available_at, slices[2].timestamps[0])
        self.assertEqual(samples[0].target_state, SectorRegimeState.BULL)
        self.assertEqual(samples[0].previous_state, SectorRegimeState.BEAR)

    def test_target_day_return_does_not_change_its_causal_scale(self) -> None:
        slices = tuple(
            river_slice(day, float(index))
            for index, day in enumerate(
                ("2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07")
            )
        )
        config = SectorRegimeDataConfig(
            context_sessions=2,
            image_height=3,
            image_minutes=4,
            volatility_lookback_sessions=3,
            volatility_minimum_observations=2,
            neutral_sigma=0.5,
            minimum_return_scale=0.001,
        )
        builder = SectorRegimeDatasetBuilder(config)

        ordinary = builder.build(slices, index_bars(target_close=101.0))[0]
        extreme = builder.build(slices, index_bars(target_close=150.0))[0]

        self.assertAlmostEqual(ordinary.return_scale, extreme.return_scale)
        self.assertNotEqual(ordinary.target_return, extreme.target_return)


if __name__ == "__main__":
    unittest.main()
