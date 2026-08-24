"""Causal price-velocity and participation-density normalization."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..sector_flow.types import SectorMinutePanel
from .types import NormalizedBoundaryPanel


@dataclass(frozen=True, slots=True)
class BoundaryNormalizationConfig:
    lookback_sessions: int = 10
    minimum_history_sessions: int = 10
    minutes_per_session: int = 240
    minimum_session_observations: int = 200
    minimum_velocity_scale: float = 1e-6

    def __post_init__(self) -> None:
        if self.lookback_sessions < 1:
            raise ValueError("normalization lookback must be positive")
        if not 1 <= self.minimum_history_sessions <= self.lookback_sessions:
            raise ValueError("minimum history must fit the normalization lookback")
        if self.minutes_per_session < 1:
            raise ValueError("minutes_per_session must be positive")
        if not 1 <= self.minimum_session_observations <= self.minutes_per_session:
            raise ValueError("minimum session observations must fit the session")
        if self.minimum_velocity_scale <= 0.0:
            raise ValueError("minimum velocity scale must be positive")


class CausalBoundaryNormalizer:
    def __init__(self, config: BoundaryNormalizationConfig | None = None) -> None:
        self.config = config or BoundaryNormalizationConfig()

    def transform(self, panel: SectorMinutePanel) -> NormalizedBoundaryPanel:
        raw_velocity, raw_valid = panel.log_returns()
        shape = panel.close.shape
        standardized = np.full(shape, np.nan, dtype=np.float64)
        relative_volume = np.full(shape, np.nan, dtype=np.float64)
        velocity_scale = np.full(shape, np.nan, dtype=np.float64)
        minute_volume_baseline = np.full(shape, np.nan, dtype=np.float64)
        velocity_valid = np.zeros(shape, dtype=np.bool_)
        density_valid = np.zeros(shape, dtype=np.bool_)
        velocity_history = [
            deque(maxlen=self.config.lookback_sessions) for _ in panel.symbols
        ]
        volume_history = [
            deque(maxlen=self.config.lookback_sessions) for _ in panel.symbols
        ]
        sessions = panel.timestamps.astype("datetime64[D]")

        for session in np.unique(sessions):
            rows = np.flatnonzero(sessions == session)
            for column in range(len(panel.symbols)):
                prior_velocity = velocity_history[column]
                if len(prior_velocity) >= self.config.minimum_history_sessions:
                    samples = np.concatenate(tuple(prior_velocity))
                    scale = max(
                        float(np.sqrt(np.mean(np.square(samples)))),
                        self.config.minimum_velocity_scale,
                    )
                    velocity_scale[rows, column] = scale
                    current_valid = raw_valid[rows, column]
                    current_rows = rows[current_valid]
                    standardized[current_rows, column] = (
                        raw_velocity[current_rows, column] / scale
                    )
                    velocity_valid[current_rows, column] = True

                prior_volume = volume_history[column]
                if len(prior_volume) >= self.config.minimum_history_sessions:
                    baseline = max(
                        float(np.mean(prior_volume)) / self.config.minutes_per_session,
                        1e-12,
                    )
                    minute_volume_baseline[rows, column] = baseline
                    observed = panel.valid[rows, column]
                    observed_rows = rows[observed]
                    relative_volume[observed_rows, column] = (
                        panel.volume[observed_rows, column] / baseline
                    )
                    density_valid[observed_rows, column] = True

            # Current-session observations become available only after every
            # current row has been transformed with the frozen prior baseline.
            for column in range(len(panel.symbols)):
                session_velocity = raw_velocity[rows, column][raw_valid[rows, column]]
                observed_volume = panel.volume[rows, column][panel.valid[rows, column]]
                if len(session_velocity) >= self.config.minimum_session_observations:
                    velocity_history[column].append(session_velocity.copy())
                if len(observed_volume) >= self.config.minimum_session_observations:
                    volume_history[column].append(float(observed_volume.sum()))

        return NormalizedBoundaryPanel(
            symbols=panel.symbols,
            timestamps=panel.timestamps.copy(),
            raw_velocity=raw_velocity,
            standardized_velocity=standardized,
            relative_volume=relative_volume,
            velocity_scale=velocity_scale,
            minute_volume_baseline=minute_volume_baseline,
            raw_velocity_valid=raw_valid,
            velocity_valid=velocity_valid,
            density_valid=density_valid,
        )
