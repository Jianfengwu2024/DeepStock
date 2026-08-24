from __future__ import annotations

import unittest

import numpy as np

from bnhmm import BarBatch
from bnhmm.sector_flow import SectorMinutePanelBuilder


def bars(symbol: str, times: list[str], closes: list[float], volumes: list[float]) -> BarBatch:
    close = np.asarray(closes, dtype=np.float64)
    return BarBatch(
        symbol=symbol,
        timestamp=np.asarray(times, dtype="datetime64[ns]"),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=np.asarray(volumes, dtype=np.float64),
    )


class SectorMinutePanelTests(unittest.TestCase):
    def test_preserves_bar_open_for_next_session_execution(self) -> None:
        batch = BarBatch(
            symbol="000001",
            timestamp=np.asarray(["2025-01-02T09:31"], dtype="datetime64[ns]"),
            open=np.asarray([9.8]),
            high=np.asarray([10.1]),
            low=np.asarray([9.7]),
            close=np.asarray([10.0]),
            volume=np.asarray([100.0]),
        )

        panel = SectorMinutePanelBuilder().build({batch.symbol: batch})

        self.assertEqual(panel.open[0, 0], 9.8)
        self.assertEqual(panel.close[0, 0], 10.0)

    def test_aligns_symbols_on_canonical_minutes_and_masks_missing_bars(self) -> None:
        first = bars(
            "000001",
            ["2025-01-02T09:31", "2025-01-02T09:32", "2025-01-02T13:01"],
            [10.0, 10.1, 10.2],
            [100.0, 110.0, 120.0],
        )
        second = bars(
            "000002",
            ["2025-01-02T09:31", "2025-01-02T13:01"],
            [20.0, 20.2],
            [200.0, 220.0],
        )

        panel = SectorMinutePanelBuilder().build(
            {"000002": second, "000001": first}
        )

        self.assertEqual(panel.symbols, ("000001", "000002"))
        self.assertEqual(panel.close.shape, (240, 2))
        self.assertEqual(str(panel.timestamps[0].astype("datetime64[m]")), "2025-01-02T09:31")
        self.assertEqual(str(panel.timestamps[119].astype("datetime64[m]")), "2025-01-02T11:30")
        self.assertEqual(str(panel.timestamps[120].astype("datetime64[m]")), "2025-01-02T13:01")
        self.assertTrue(panel.valid[1, 0])
        self.assertFalse(panel.valid[1, 1])
        self.assertTrue(np.isnan(panel.close[1, 1]))

    def test_returns_preserve_flat_prices_and_reject_discontinuous_edges(self) -> None:
        batch = bars(
            "000001",
            [
                "2025-01-02T09:31",
                "2025-01-02T09:32",
                "2025-01-02T09:34",
                "2025-01-02T13:01",
                "2025-01-03T09:31",
                "2025-01-03T09:32",
            ],
            [10.0, 10.0, 10.2, 10.3, 10.5, 10.6],
            [100.0] * 6,
        )

        panel = SectorMinutePanelBuilder().build({batch.symbol: batch})
        returns, valid = panel.log_returns()
        index = {
            str(value.astype("datetime64[m]")): position
            for position, value in enumerate(panel.timestamps)
        }

        flat = index["2025-01-02T09:32"]
        gap = index["2025-01-02T09:34"]
        afternoon = index["2025-01-02T13:01"]
        next_day = index["2025-01-03T09:31"]
        self.assertTrue(valid[flat, 0])
        self.assertEqual(returns[flat, 0], 0.0)
        self.assertFalse(valid[gap, 0])
        self.assertFalse(valid[afternoon, 0])
        self.assertFalse(valid[next_day, 0])


if __name__ == "__main__":
    unittest.main()
