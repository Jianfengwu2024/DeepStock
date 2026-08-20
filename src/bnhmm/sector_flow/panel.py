"""Build canonical one-minute panels without temporal aggregation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..types import BarBatch
from .types import SectorMinutePanel


_CONTINUOUS_MINUTE_SLOTS = np.concatenate(
    (
        np.arange(9 * 60 + 31, 11 * 60 + 31, dtype=np.int64),
        np.arange(13 * 60 + 1, 15 * 60 + 1, dtype=np.int64),
    )
)


class SectorMinutePanelBuilder:
    def build(self, bars_by_symbol: Mapping[str, BarBatch]) -> SectorMinutePanel:
        if not bars_by_symbol:
            raise ValueError("at least one sector symbol is required")
        symbols = tuple(sorted(bars_by_symbol))
        if any(bars_by_symbol[symbol].symbol != symbol for symbol in symbols):
            raise ValueError("bar-batch symbol does not match its panel key")

        sessions = sorted(
            {
                session
                for batch in bars_by_symbol.values()
                for session in batch.timestamp.astype("datetime64[D]")
            }
        )
        timestamps = np.asarray(
            [
                session + np.timedelta64(int(slot), "m")
                for session in sessions
                for slot in _CONTINUOUS_MINUTE_SLOTS
            ],
            dtype="datetime64[ns]",
        )
        shape = (len(timestamps), len(symbols))
        close = np.full(shape, np.nan, dtype=np.float64)
        volume = np.full(shape, np.nan, dtype=np.float64)
        valid = np.zeros(shape, dtype=np.bool_)
        positions = {
            int(timestamp.astype("datetime64[m]").astype(np.int64)): index
            for index, timestamp in enumerate(timestamps)
        }

        for column, symbol in enumerate(symbols):
            batch = bars_by_symbol[symbol]
            seen: set[int] = set()
            for index, timestamp in enumerate(batch.timestamp):
                key = int(timestamp.astype("datetime64[m]").astype(np.int64))
                destination = positions.get(key)
                if destination is None:
                    continue
                if key in seen:
                    raise ValueError(f"duplicate minute bar for {symbol}: {timestamp}")
                seen.add(key)
                close[destination, column] = float(batch.close[index])
                volume[destination, column] = float(batch.volume[index])
                valid[destination, column] = True

        return SectorMinutePanel(
            symbols=symbols,
            timestamps=timestamps,
            close=close,
            volume=volume,
            valid=valid,
        )


