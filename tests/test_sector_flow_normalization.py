from __future__ import annotations

import unittest

import numpy as np

from bnhmm import BarBatch
from bnhmm.sector_flow import (
    CausalSectorNormalizer,
    SectorMinutePanelBuilder,
    SectorNormalizationConfig,
)


def history_panel(*, current_increment: float = 0.01, current_open_volume: float = 1000.0):
    timestamps: list[np.datetime64] = []
    closes: list[float] = []
    volumes: list[float] = []
    price = 10.0
    for session in range(5):
        day = np.datetime64("2025-01-02") + np.timedelta64(session, "D")
        timestamps.extend((day + np.timedelta64(9 * 60 + 31, "m"), day + np.timedelta64(9 * 60 + 32, "m")))
        increment = current_increment if session == 4 else 0.01
        closes.extend((price, price * np.exp(increment)))
        volumes.extend(((current_open_volume if session == 4 else 1000.0), 100.0))
        price *= 1.01
    close = np.asarray(closes, dtype=np.float64)
    batch = BarBatch(
        symbol="000001",
        timestamp=np.asarray(timestamps, dtype="datetime64[ns]"),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=np.asarray(volumes, dtype=np.float64),
    )
    return SectorMinutePanelBuilder().build({batch.symbol: batch})


class SectorFlowNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SectorNormalizationConfig(
            lookback_sessions=4,
            minimum_observations=4,
            minimum_velocity_scale=1e-6,
            activity_clip=20.0,
        )

    def test_current_session_does_not_change_its_historical_baselines(self) -> None:
        original = CausalSectorNormalizer(self.config).transform(history_panel())
        changed = CausalSectorNormalizer(self.config).transform(
            history_panel(current_increment=0.05, current_open_volume=5000.0)
        )
        current = original.timestamps.astype("datetime64[D]") == np.datetime64("2025-01-06")

        np.testing.assert_allclose(
            original.velocity_scale[current], changed.velocity_scale[current], equal_nan=True
        )
        np.testing.assert_allclose(
            original.volume_baseline[current], changed.volume_baseline[current], equal_nan=True
        )

    def test_velocity_uses_zero_centered_rms_without_mean_subtraction(self) -> None:
        normalized = CausalSectorNormalizer(self.config).transform(history_panel())
        row = np.flatnonzero(
            normalized.timestamps.astype("datetime64[m]")
            == np.datetime64("2025-01-06T09:32")
        )[0]

        self.assertTrue(normalized.valid[row, 0])
        self.assertGreater(normalized.velocity[row, 0], 0.9)
        self.assertLess(normalized.velocity[row, 0], 1.1)

    def test_volume_baseline_is_specific_to_the_same_minute_slot(self) -> None:
        normalized = CausalSectorNormalizer(self.config).transform(history_panel())
        opening = np.flatnonzero(
            normalized.timestamps.astype("datetime64[m]")
            == np.datetime64("2025-01-06T09:31")
        )[0]
        regular = opening + 1

        self.assertAlmostEqual(normalized.volume_baseline[opening, 0], 1000.0)
        self.assertAlmostEqual(normalized.volume_baseline[regular, 0], 100.0)
        self.assertAlmostEqual(normalized.activity[opening, 0], 1.0)


if __name__ == "__main__":
    unittest.main()
