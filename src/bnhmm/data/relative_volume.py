"""Leakage-safe relative-volume normalization by exchange minute slot."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from numpy.typing import NDArray

from ..config import ProfileConfig
from ..types import BarBatch


class RelativeVolumeBaseline:
    """Compute volume ratios using completed prior sessions only."""

    def __init__(self, config: ProfileConfig) -> None:
        self._lookback = config.relative_volume_lookback_sessions
        self._minimum = config.relative_volume_min_observations
        self._clip = config.relative_volume_clip
        self._history: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._lookback)
        )

    def transform(self, bars: BarBatch) -> NDArray[np.float64]:
        if not len(bars):
            return np.asarray([], dtype=np.float64)

        sessions = bars.timestamp.astype("datetime64[D]")
        minute_slots = bars.timestamp.astype("datetime64[m]").astype(np.int64) % (24 * 60)
        relative = np.ones(len(bars), dtype=np.float64)

        for session in np.unique(sessions):
            indices = np.flatnonzero(sessions == session)
            pending: list[tuple[int, float]] = []
            for index in indices:
                slot = int(minute_slots[index])
                volume = float(bars.volume[index])
                history = self._history[slot]
                if len(history) >= self._minimum:
                    baseline = float(np.median(np.fromiter(history, dtype=np.float64)))
                    if baseline > 0:
                        relative[index] = np.clip(volume / baseline, 0.0, self._clip)
                pending.append((slot, volume))

            # The current session cannot influence any baseline in that session.
            for slot, volume in pending:
                self._history[slot].append(volume)

        return relative

    def observations_for_slot(self, hour: int, minute: int) -> int:
        return len(self._history[hour * 60 + minute])

