from __future__ import annotations

import unittest

import numpy as np

from bnhmm import BarBatch, ProfileConfig, RelativeVolumeBaseline


def full_session(day: int, volume: float) -> BarBatch:
    start = np.datetime64(f"2026-01-{day:02d}T09:31:00", "ns")
    timestamp = start + np.arange(240).astype("timedelta64[m]")
    price = np.full(240, 10.0, dtype=np.float64)
    return BarBatch(
        "000001",
        timestamp,
        price,
        price,
        price,
        price,
        np.full(240, volume),
    )


class RelativeVolumeTests(unittest.TestCase):
    def test_current_session_is_not_in_its_own_baseline(self) -> None:
        config = ProfileConfig(relative_volume_min_observations=3)
        baseline = RelativeVolumeBaseline(config)
        for day in range(1, 4):
            np.testing.assert_allclose(
                baseline.transform(full_session(day, 10.0)), 1.0
            )

        ratio = baseline.transform(full_session(4, 20.0))
        np.testing.assert_allclose(ratio, 2.0)
        self.assertEqual(baseline.observations_for_symbol("000001"), 4)


if __name__ == "__main__":
    unittest.main()

