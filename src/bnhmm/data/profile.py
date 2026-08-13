"""Price/relative-volume histogram state construction from one-minute bars."""

from __future__ import annotations

from collections import defaultdict
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


class PriceVolumeProfileBuilder:
    """Build configured minute-derived states without using future sessions."""

    def __init__(self, config: ProfileConfig | None = None) -> None:
        self.config = config or ProfileConfig()
        self.relative_volume = RelativeVolumeBaseline(self.config)
        self._normalizers: dict[str, _NormalizerState] = defaultdict(
            lambda: _NormalizerState(previous_close=None)
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
            if normalizer.previous_close is None:
                normalizer.previous_close = float(day.close[-1])
                continue
            anchor = normalizer.previous_close

            if "60m" in self.config.state_timeframes:
                states.extend(self._build_hour_states(day, day_relative, anchor))
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
                    )
                )

            normalizer.previous_close = float(day.close[-1])

        return states

    @staticmethod
    def _continuous_session_bars(bars: BarBatch) -> BarBatch:
        """Normalize mixed source conventions to the canonical 240 A-share minutes."""
        minute_slots = bars.timestamp.astype("datetime64[m]").astype(np.int64) % (24 * 60)
        morning = (minute_slots > 9 * 60 + 30) & (minute_slots <= 11 * 60 + 30)
        afternoon = (minute_slots > 13 * 60) & (minute_slots <= 15 * 60)
        return bars.take(np.flatnonzero(morning | afternoon))

    def _build_hour_states(
        self,
        day: BarBatch,
        relative: NDArray[np.float64],
        anchor: float,
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
    ) -> ProfileState:
        log_bin_width = float(np.log1p(self.config.bin_width_bps / 10_000.0))
        normalized_low = np.log(bars.low / anchor)
        normalized_high = np.log(bars.high / anchor)
        minimum_edge = int(np.floor(float(normalized_low.min()) / log_bin_width))
        maximum_edge = int(np.ceil(float(normalized_high.max()) / log_bin_width))
        minimum_edge -= self.config.profile_margin_bins
        maximum_edge += self.config.profile_margin_bins
        if maximum_edge - minimum_edge < 3:
            maximum_edge = minimum_edge + 3

        profile_clipped = maximum_edge - minimum_edge > self.config.max_profile_bins
        if profile_clipped:
            center = (minimum_edge + maximum_edge) // 2
            minimum_edge = center - self.config.max_profile_bins // 2
            maximum_edge = minimum_edge + self.config.max_profile_bins

        edge_indices = np.arange(minimum_edge, maximum_edge + 1, dtype=np.float64)
        edges = edge_indices * log_bin_width
        bin_coordinates = edge_indices[:-1] + 0.5
        profile_bins = len(bin_coordinates)
        price_histogram = np.zeros(profile_bins, dtype=np.float64)
        relative_volume_histogram = np.zeros(profile_bins, dtype=np.float64)
        weights = np.maximum(relative, 0.0)

        for low, high, weight in zip(normalized_low, normalized_high, weights, strict=True):
            self._deposit(price_histogram, edges, float(low), float(high), 1.0)
            if weight <= 0:
                continue
            self._deposit(
                relative_volume_histogram,
                edges,
                float(low),
                float(high),
                float(weight),
            )

        relative_volume_intensity = relative_volume_histogram / max(len(bars), 1)
        self._normalize(price_histogram)
        self._normalize(relative_volume_histogram)

        return ProfileState(
            symbol=bars.symbol,
            timeframe=timeframe,
            period_start=period_start.astype("datetime64[ns]"),
            period_end=period_end.astype("datetime64[ns]"),
            first_bar_time=bars.timestamp[0].astype("datetime64[ns]"),
            start_price=float(bars.open[0]),
            end_price=float(bars.close[-1]),
            high_price=float(bars.high.max()),
            low_price=float(bars.low.min()),
            anchor_price=anchor,
            log_bin_width=log_bin_width,
            price_edges=edges,
            bin_coordinates=bin_coordinates,
            price_distribution=price_histogram,
            distribution=relative_volume_histogram,
            relative_volume_intensity=relative_volume_intensity,
            adjacent_pairs=sliding_window_view(relative_volume_histogram, 2).copy(),
            adjacent_triples=sliding_window_view(relative_volume_histogram, 3).copy(),
            total_raw_volume=float(bars.volume.sum()),
            total_relative_volume=float(weights.sum()),
            bar_count=len(bars),
            profile_clipped=profile_clipped,
        )

    def _normalize(self, histogram: NDArray[np.float64]) -> None:
        total = float(histogram.sum())
        if total <= 0:
            histogram.fill(1.0 / len(histogram))
        else:
            histogram /= total

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
