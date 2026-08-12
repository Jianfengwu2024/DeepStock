"""Array-oriented domain types used across data, model, and evaluation layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
TimeArray = NDArray[np.datetime64]


@dataclass(frozen=True, slots=True)
class BarBatch:
    symbol: str
    timestamp: TimeArray
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    volume: FloatArray

    def __post_init__(self) -> None:
        arrays = (self.timestamp, self.open, self.high, self.low, self.close, self.volume)
        lengths = {len(item) for item in arrays}
        if len(lengths) != 1:
            raise ValueError("all bar arrays must have equal length")
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if len(self) and np.any(self.timestamp[1:] < self.timestamp[:-1]):
            raise ValueError("timestamps must be sorted ascending")
        if len(self) and (
            np.any(~np.isfinite(self.open))
            or np.any(~np.isfinite(self.high))
            or np.any(~np.isfinite(self.low))
            or np.any(~np.isfinite(self.close))
            or np.any(~np.isfinite(self.volume))
        ):
            raise ValueError("bar values must be finite")
        if len(self) and (
            np.any(self.low > self.high)
            or np.any(self.volume < 0)
            or np.any(self.open <= 0)
            or np.any(self.high <= 0)
            or np.any(self.low <= 0)
            or np.any(self.close <= 0)
        ):
            raise ValueError("bar price/volume invariants are violated")

    def __len__(self) -> int:
        return len(self.timestamp)

    def take(self, indices: NDArray[np.integer] | Sequence[int]) -> "BarBatch":
        selected = np.asarray(indices)
        return BarBatch(
            symbol=self.symbol,
            timestamp=self.timestamp[selected],
            open=self.open[selected],
            high=self.high[selected],
            low=self.low[selected],
            close=self.close[selected],
            volume=self.volume[selected],
        )

    @classmethod
    def empty(cls, symbol: str) -> "BarBatch":
        return cls(
            symbol=symbol,
            timestamp=np.asarray([], dtype="datetime64[ns]"),
            open=np.asarray([], dtype=np.float64),
            high=np.asarray([], dtype=np.float64),
            low=np.asarray([], dtype=np.float64),
            close=np.asarray([], dtype=np.float64),
            volume=np.asarray([], dtype=np.float64),
        )

    @classmethod
    def from_records(cls, symbol: str, records: Iterable[Mapping[str, object]]) -> "BarBatch":
        rows = list(records)
        if not rows:
            return cls.empty(symbol)
        rows.sort(key=lambda row: str(row["timestamp"]))
        return cls(
            symbol=symbol,
            timestamp=np.asarray([row["timestamp"] for row in rows], dtype="datetime64[ns]"),
            open=np.asarray([row["open"] for row in rows], dtype=np.float64),
            high=np.asarray([row["high"] for row in rows], dtype=np.float64),
            low=np.asarray([row["low"] for row in rows], dtype=np.float64),
            close=np.asarray([row["close"] for row in rows], dtype=np.float64),
            volume=np.asarray([row["volume"] for row in rows], dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class ProfileState:
    symbol: str
    timeframe: str
    period_start: np.datetime64
    period_end: np.datetime64
    anchor_price: float
    log_price_scale: float
    price_edges: FloatArray
    distribution: FloatArray
    adjacent_pairs: FloatArray
    adjacent_triples: FloatArray
    total_raw_volume: float
    total_relative_volume: float
    bar_count: int

    def __post_init__(self) -> None:
        if self.timeframe not in {"60m", "1d"}:
            raise ValueError("profile timeframe must be 60m or 1d")
        if len(self.price_edges) != len(self.distribution) + 1:
            raise ValueError("price_edges must have one more element than distribution")
        if self.adjacent_pairs.shape != (len(self.distribution) - 1, 2):
            raise ValueError("adjacent_pairs has an invalid shape")
        if self.adjacent_triples.shape != (len(self.distribution) - 2, 3):
            raise ValueError("adjacent_triples has an invalid shape")
        if not np.isclose(float(self.distribution.sum()), 1.0, atol=1e-9):
            raise ValueError("profile distribution must sum to one")
        if self.period_end <= self.period_start:
            raise ValueError("profile period must have positive duration")

