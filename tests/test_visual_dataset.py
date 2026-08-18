from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from bnhmm.config import HOLOGRAPHIC_FIELD_COUNT
from bnhmm.types import BarBatch, HolographicState
from bnhmm.visual_dataset import (
    VisualDatasetConfig,
    VisualDatasetExporter,
    rasterize_state,
    render_state_png,
    vector_length,
)


def sample_state() -> HolographicState:
    price_coordinates = np.arange(-3, 3, dtype=np.float64) + 0.5
    volume_coordinates = np.linspace(-0.8, 0.8, 5)
    fields = np.zeros((6, 5, HOLOGRAPHIC_FIELD_COUNT), dtype=np.float64)
    fields[1, 2, 0] = 0.25
    fields[2, 2, 0] = 0.50
    fields[3, 3, 0] = 0.25
    fields[..., 1] = fields[..., 0] * 1.5
    fields[..., 2] = fields[..., 0] * -0.5
    return HolographicState(
        symbol="688001",
        timeframe="1d",
        period_start=np.datetime64("2025-06-03T09:30"),
        period_end=np.datetime64("2025-06-03T15:00"),
        first_bar_time=np.datetime64("2025-06-03T09:31"),
        start_price=10.1,
        end_price=10.2,
        high_price=10.3,
        low_price=9.9,
        anchor_price=10.0,
        log_bin_width=float(np.log1p(10 / 10_000)),
        bin_coordinates=price_coordinates,
        normalized_bin_coordinates=price_coordinates / 10,
        volume_coordinates=volume_coordinates,
        fields=fields,
        normalization_sessions=10,
        average_minute_volume=1000.0,
        location_scale=0.02,
        price_velocity_mean=0.0,
        price_velocity_scale=0.001,
        relative_volume_mean=0.5,
        relative_volume_scale=0.2,
        volume_velocity_mean=0.0,
        volume_velocity_scale=0.1,
        total_raw_volume=240000.0,
        total_relative_volume=240.0,
        bar_count=240,
        profile_clipped=False,
        flow_samples=22,
    )


class VisualDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = VisualDatasetConfig(
            price_bins_each_side=10,
            volume_bins=5,
            image_width=640,
            image_height=360,
        )

    def test_raster_has_fixed_shape_and_exact_zero_padding(self) -> None:
        raster = rasterize_state(sample_state(), self.config)

        self.assertEqual(raster.density.shape, (20, 5))
        self.assertEqual(raster.price_velocity.shape, (20, 5))
        self.assertEqual(raster.copied_price_bins, 6)
        self.assertAlmostEqual(float(raster.density.sum()), 1.0)
        self.assertTrue(np.all(raster.density[:7] == 0.0))
        self.assertTrue(np.all(raster.density[13:] == 0.0))
        self.assertAlmostEqual(float(raster.price_velocity[8, 2]), 1.5)
        self.assertAlmostEqual(float(raster.volume_velocity[8, 2]), -0.5)

    def test_previous_close_is_fixed_center_boundary(self) -> None:
        state = sample_state()
        zero_source_boundary = int(
            round(-self.config.minimum_price_edge)
        )
        self.assertEqual(zero_source_boundary, self.config.price_bins // 2)
        self.assertEqual(state.anchor_price, 10.0)

    def test_vector_length_is_monotonic_and_bounded(self) -> None:
        lengths = [vector_length(value, self.config) for value in (0, 0.5, 1, 2, 8)]
        self.assertEqual(lengths, sorted(lengths))
        self.assertEqual(lengths[0], 0.0)
        self.assertEqual(lengths[-1], self.config.vector_max_pixels)

    def test_renderer_writes_identical_canvas_dimensions(self) -> None:
        state = sample_state()
        raster = rasterize_state(state, self.config)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory, "first.png")
            second = Path(directory, "second.png")
            render_state_png(state, raster, first, self.config)
            render_state_png(state, raster, second, self.config)
            with Image.open(first) as image:
                self.assertEqual(image.size, (640, 360))
            with Image.open(second) as image:
                self.assertEqual(image.size, (640, 360))

    def test_full_session_ohlc_does_not_inherit_continuous_flow_filter(self) -> None:
        state = sample_state()
        bars = BarBatch(
            symbol=state.symbol,
            timestamp=np.asarray(
                ["2025-06-03T09:30", "2025-06-03T09:31", "2025-06-03T15:00"],
                dtype="datetime64[ns]",
            ),
            open=np.asarray([9.8, 10.1, 10.2]),
            high=np.asarray([10.4, 10.2, 10.3]),
            low=np.asarray([9.8, 10.0, 10.1]),
            close=np.asarray([9.8, 10.2, 10.25]),
            volume=np.asarray([100.0, 200.0, 300.0]),
        )

        attached = VisualDatasetExporter._attach_full_session_ohlc([state], bars)[0]

        self.assertEqual(attached.start_price, 9.8)
        self.assertEqual(attached.high_price, 10.4)
        self.assertEqual(attached.low_price, 9.8)
        self.assertEqual(attached.end_price, 10.25)
        self.assertEqual(attached.anchor_price, state.anchor_price)


if __name__ == "__main__":
    unittest.main()
