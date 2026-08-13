"""Price/relative-volume histogram state construction from one-minute bars."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from ..config import ProfileConfig
from ..types import BarBatch, ProfileState
from .relative_volume import RelativeVolumeBaseline


_HOUR_BUCKETS = (
    (9 * 60 + 30, 10 * 60 + 30),
    (10 * 60 + 30, 11 * 60 + 30),
    (13 * 60, 14 * 60),
    (14 * 60, 15 * 60),
)


@dataclass(slots=True)
class _NormalizerState:
    previous_close: float | None
    daily_ranges: deque[float]


class PriceVolumeProfileBuilder:
    """Build configured minute-derived states without using future sessions."""

    def __init__(self, config: ProfileConfig | None = None) -> None:
        self.config = config or ProfileConfig()
        self.relative_volume = RelativeVolumeBaseline(self.config)
        self._normalizers: dict[str, _NormalizerState] = defaultdict(
            lambda: _NormalizerState(
                previous_close=None,
                daily_ranges=deque(maxlen=self.config.scale_lookback_sessions),
            )
        )

    def build(self, bars: BarBatch) -> list[ProfileState]:
        if not len(bars):
            return []

        bars = self._continuous_session_bars(bars)
        if not len(bars):
            return []
        relative = self.relative_volume.transform(bars)
        sessions = bars.timestamp.astype("datetime64[D]")
        states: list[ProfileState] = []

        for session in np.unique(sessions):
            day_indices = np.flatnonzero(sessions == session)
            day = bars.take(day_indices)
            day_relative = relative[day_indices]
            normalizer = self._normalizers[bars.symbol]
            anchor = normalizer.previous_close or float(day.open[0])
            scale = self._historical_scale(normalizer)

            if "60m" in self.config.state_timeframes:
                states.extend(self._build_hour_states(day, day_relative, anchor, scale))
            if "1d" in self.config.state_timeframes and len(day) >= self.config.minimum_day_bars:
                session_date = day.timestamp[0].astype("datetime64[D]")
                states.append(
                    self._build_state(
                        day,
                        day_relative,
                        "1d",
                        session_date + np.timedelta64(9 * 60 + 30, "m"),
                        session_date + np.timedelta64(15 * 60, "m"),
                        anchor,
                        scale,
                    )
                )

            daily_range = float(np.log(float(day.high.max()) / float(day.low.min())))
            if np.isfinite(daily_range) and daily_range > 0:
                normalizer.daily_ranges.append(daily_range)
            normalizer.previous_close = float(day.close[-1])

        return states

    @staticmethod
    def _continuous_session_bars(bars: BarBatch) -> BarBatch:
        """Normalize mixed source conventions to the canonical 240 A-share minutes."""
        minute_slots = bars.timestamp.astype("datetime64[m]").astype(np.int64) % (24 * 60)
        morning = (minute_slots > 9 * 60 + 30) & (minute_slots <= 11 * 60 + 30)
        afternoon = (minute_slots > 13 * 60) & (minute_slots <= 15 * 60)
        return bars.take(np.flatnonzero(morning | afternoon))

    def _historical_scale(self, state: _NormalizerState) -> float:
        if not state.daily_ranges:
            return self.config.default_log_scale
        scale = float(np.median(np.fromiter(state.daily_ranges, dtype=np.float64)))
        return max(scale, self.config.minimum_log_scale)

    def _build_hour_states(
        self,
        day: BarBatch,
        relative: NDArray[np.float64],
        anchor: float,
        scale: float,
    ) -> list[ProfileState]:
        minute_slots = day.timestamp.astype("datetime64[m]").astype(np.int64) % (24 * 60)
        session_date = day.timestamp[0].astype("datetime64[D]")
        result: list[ProfileState] = []

        for start_slot, end_slot in _HOUR_BUCKETS:
            # Source minute bars are end-stamped: (09:30, 10:30], for example.
            indices = np.flatnonzero((minute_slots > start_slot) & (minute_slots <= end_slot))
            if len(indices) < self.config.minimum_hour_bars:
                continue
            start = session_date + np.timedelta64(start_slot, "m")
            end = session_date + np.timedelta64(end_slot, "m")
            result.append(
                self._build_state(
                    day.take(indices),
                    relative[indices],
                    "60m",
                    start,
                    end,
                    anchor,
                    scale,
                )
            )
        return result

    def _build_state(
        self,
        bars: BarBatch,
        relative: NDArray[np.float64],
        timeframe: str,
        period_start: np.datetime64,
        period_end: np.datetime64,
        anchor: float,
        scale: float,
    ) -> ProfileState:
        edges = np.linspace(
            -self.config.price_span,
            self.config.price_span,
            self.config.bins + 1,
            dtype=np.float64,
        )
        histogram = np.zeros(self.config.bins, dtype=np.float64)

        normalized_low = np.log(bars.low / anchor) / scale
        normalized_high = np.log(bars.high / anchor) / scale
        weights = np.maximum(relative, 0.0)

        for low, high, weight in zip(normalized_low, normalized_high, weights, strict=True):
            if weight <= 0:
                continue
            self._deposit(histogram, edges, float(low), float(high), float(weight))

        total = float(histogram.sum())
        if total <= 0:
            histogram.fill(1.0 / self.config.bins)
        else:
            histogram /= total

        return ProfileState(
            symbol=bars.symbol,
            timeframe=timeframe,
            period_start=period_start.astype("datetime64[ns]"),
            period_end=period_end.astype("datetime64[ns]"),
            first_bar_time=bars.timestamp[0].astype("datetime64[ns]"),
            start_price=float(bars.open[0]),
            end_price=float(bars.close[-1]),
            anchor_price=anchor,
            log_price_scale=scale,
            price_edges=edges,
            distribution=histogram,
            adjacent_pairs=sliding_window_view(histogram, 2).copy(),
            adjacent_triples=sliding_window_view(histogram, 3).copy(),
            total_raw_volume=float(bars.volume.sum()),
            total_relative_volume=float(weights.sum()),
            bar_count=len(bars),
        )

    @staticmethod
    def _deposit(
        histogram: NDArray[np.float64],
        edges: NDArray[np.float64],
        low: float,
        high: float,
        weight: float,
    ) -> None:
        if high < low:
            low, high = high, low
        if high <= edges[0]:
            histogram[0] += weight
            return
        if low >= edges[-1]:
            histogram[-1] += weight
            return

        clipped_low = max(low, float(edges[0]))
        clipped_high = min(high, float(edges[-1]))
        if np.isclose(clipped_low, clipped_high):
            index = int(np.clip(np.searchsorted(edges, clipped_low, side="right") - 1, 0, len(histogram) - 1))
            histogram[index] += weight
            return

        overlaps = np.maximum(
            0.0,
            np.minimum(edges[1:], clipped_high) - np.maximum(edges[:-1], clipped_low),
        )
        overlap_total = float(overlaps.sum())
        if overlap_total <= 0:
            center = (clipped_low + clipped_high) / 2
            index = int(np.clip(np.searchsorted(edges, center, side="right") - 1, 0, len(histogram) - 1))
            histogram[index] += weight
        else:
            histogram += weight * overlaps / overlap_total
