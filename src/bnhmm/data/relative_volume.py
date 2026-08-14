"""Leakage-safe relative-volume normalization by prior daily volume."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from numpy.typing import NDArray

from ..config import ProfileConfig
from ..types import BarBatch


class RelativeVolumeBaseline:
    """Divide minute volume by prior average daily volume divided by 240."""

    def __init__(self, config: ProfileConfig) -> None:
        self._lookback = config.relative_volume_lookback_sessions
        self._minimum = config.relative_volume_min_observations
        self._clip = config.relative_volume_clip
        self._minutes = config.trading_minutes_per_session
        self._history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._lookback)
        )

    def transform(self, bars: BarBatch) -> NDArray[np.float64]:
        if not len(bars):
            return np.asarray([], dtype=np.float64)

        sessions = bars.timestamp.astype("datetime64[D]")
        relative = np.ones(len(bars), dtype=np.float64)

        for session in np.unique(sessions):
            indices = np.flatnonzero(sessions == session)
            history = self._history[bars.symbol]
            if len(history) >= self._minimum:
                baseline = float(np.mean(history)) / self._minutes
                if baseline > 0:
                    relative[indices] = np.clip(
                        bars.volume[indices] / baseline, 0.0, self._clip
                    )
            # The current session enters the baseline only after all its bars transform.
            history.append(float(bars.volume[indices].sum()))

        return relative

    def observations_for_symbol(self, symbol: str) -> int:
        return len(self._history[symbol])

