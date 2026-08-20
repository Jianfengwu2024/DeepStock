"""Causal minute-slot normalization for sector panels."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from .types import NormalizedSectorPanel, SectorMinutePanel


@dataclass(frozen=True, slots=True)
class SectorNormalizationConfig:
    lookback_sessions: int = 20
    minimum_observations: int = 10
    minimum_velocity_scale: float = 1e-5
    velocity_clip: float = 8.0
    activity_clip: float = 10.0

    def __post_init__(self) -> None:
        if self.lookback_sessions < 2:
            raise ValueError("normalization lookback must be at least two sessions")
        if not 1 <= self.minimum_observations <= self.lookback_sessions:
            raise ValueError("minimum observations must fit the lookback")
        if self.minimum_velocity_scale <= 0.0:
            raise ValueError("minimum velocity scale must be positive")
        if self.velocity_clip <= 0.0 or self.activity_clip <= 1.0:
            raise ValueError("normalization clipping limits are invalid")


class CausalSectorNormalizer:
    def __init__(self, config: SectorNormalizationConfig | None = None) -> None:
        self.config = config or SectorNormalizationConfig()

    def transform(self, panel: SectorMinutePanel) -> NormalizedSectorPanel:
        raw_velocity, raw_valid = panel.log_returns()
        shape = panel.close.shape
        velocity = np.full(shape, np.nan, dtype=np.float64)
        activity = np.full(shape, np.nan, dtype=np.float64)
        velocity_scale = np.full(shape, np.nan, dtype=np.float64)
        volume_baseline = np.full(shape, np.nan, dtype=np.float64)
        valid = np.zeros(shape, dtype=np.bool_)
        activity_valid = np.zeros(shape, dtype=np.bool_)
        velocity_history = defaultdict(
            lambda: deque(maxlen=self.config.lookback_sessions)
        )
        volume_history = defaultdict(
            lambda: deque(maxlen=self.config.lookback_sessions)
        )
        slots = (
            panel.timestamps.astype("datetime64[m]").astype(np.int64) % (24 * 60)
        )
        sessions = panel.timestamps.astype("datetime64[D]")

        for session in np.unique(sessions):
            rows = np.flatnonzero(sessions == session)
            for row in rows:
                slot = int(slots[row])
                for column in range(len(panel.symbols)):
                    velocity_samples = velocity_history[(column, slot)]
                    if len(velocity_samples) >= self.config.minimum_observations:
                        scale = max(
                            float(np.sqrt(np.mean(np.square(velocity_samples)))),
                            self.config.minimum_velocity_scale,
                        )
                        velocity_scale[row, column] = scale
                        if raw_valid[row, column]:
                            velocity[row, column] = np.clip(
                                raw_velocity[row, column] / scale,
                                -self.config.velocity_clip,
                                self.config.velocity_clip,
                            )
                            valid[row, column] = True

                    volume_samples = volume_history[(column, slot)]
                    if len(volume_samples) >= self.config.minimum_observations:
                        baseline = max(float(np.median(volume_samples)), 1e-12)
                        volume_baseline[row, column] = baseline
                        if panel.valid[row, column]:
                            activity[row, column] = np.clip(
                                panel.volume[row, column] / baseline,
                                0.0,
                                self.config.activity_clip,
                            )
                            activity_valid[row, column] = True

            for row in rows:
                slot = int(slots[row])
                for column in range(len(panel.symbols)):
                    if raw_valid[row, column]:
                        velocity_history[(column, slot)].append(
                            float(raw_velocity[row, column])
                        )
                    if panel.valid[row, column]:
                        volume_history[(column, slot)].append(
                            float(panel.volume[row, column])
                        )

        return NormalizedSectorPanel(
            symbols=panel.symbols,
            timestamps=panel.timestamps,
            raw_velocity=raw_velocity,
            velocity=velocity,
            activity=activity,
            velocity_scale=velocity_scale,
            volume_baseline=volume_baseline,
            valid=valid,
            activity_valid=activity_valid,
        )
