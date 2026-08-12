from __future__ import annotations

import unittest

import numpy as np

from bnhmm import BarBatch, ProfileConfig, RelativeVolumeBaseline


def one_bar(day: int, volume: float) -> BarBatch:
    timestamp = np.asarray([f"2026-01-{day:02d}T09:31:00"], dtype="datetime64[ns]")
    price = np.asarray([10.0], dtype=np.float64)
    return BarBatch("000001", timestamp, price, price, price, price, np.asarray([volume]))


class RelativeVolumeTests(unittest.TestCase):
    def test_current_session_is_not_in_its_own_baseline(self) -> None:
        config = ProfileConfig(relative_volume_min_observations=3)
        baseline = RelativeVolumeBaseline(config)
        for day in range(1, 4):
            self.assertEqual(float(baseline.transform(one_bar(day, 10.0))[0]), 1.0)

        ratio = float(baseline.transform(one_bar(4, 20.0))[0])
        self.assertAlmostEqual(ratio, 2.0)
        self.assertEqual(baseline.observations_for_slot(9, 31), 4)


if __name__ == "__main__":
    unittest.main()

