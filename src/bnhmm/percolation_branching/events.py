"""Causal price activations with relative-volume event marks."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..sector_flow.types import SectorMinutePanel
from .types import ActivationConfig, ActivationPanel


class ActivationEventBuilder:
    def __init__(self, config: ActivationConfig | None = None) -> None:
        self.config = config or ActivationConfig()

    def build(self, panel: SectorMinutePanel) -> ActivationPanel:
        raw_velocity, raw_valid = panel.log_returns()
        shape = panel.close.shape
        standardized = np.full(shape, np.nan, dtype=np.float64)
        velocity_scale = np.full(shape, np.nan, dtype=np.float64)
        relative_volume = np.full(shape, np.nan, dtype=np.float64)
        volume_baseline = np.full(shape, np.nan, dtype=np.float64)
        marks = np.ones(shape, dtype=np.float64)
        event_valid = np.zeros(shape, dtype=np.bool_)
        velocity_history = [
            deque(maxlen=self.config.velocity_lookback_sessions)
            for _ in panel.symbols
        ]
        volume_history = [
            deque(maxlen=self.config.volume_lookback_sessions)
            for _ in panel.symbols
        ]
        row_sessions = panel.timestamps.astype("datetime64[D]")
        minute_slots = panel.timestamps.astype("datetime64[m]").astype(np.int64) % (24 * 60)

        for session in np.unique(row_sessions):
            rows = np.flatnonzero(row_sessions == session)
            slots = minute_slots[rows]
            for column in range(len(panel.symbols)):
                prior_velocity = velocity_history[column]
                if len(prior_velocity) >= self.config.minimum_history_sessions:
                    samples = np.concatenate(tuple(prior_velocity))
                    scale = max(
                        float(np.sqrt(np.mean(np.square(samples)))),
                        self.config.minimum_velocity_scale,
                    )
                    current_valid = raw_valid[rows, column]
                    current_rows = rows[current_valid]
                    velocity_scale[current_rows, column] = scale
                    standardized[current_rows, column] = raw_velocity[current_rows, column] / scale
                    event_valid[current_rows, column] = True

                prior_volume = volume_history[column]
                if len(prior_volume) >= self.config.minimum_history_sessions:
                    for local, row in enumerate(rows):
                        values = [
                            history.get(int(slots[local]))
                            for history in prior_volume
                        ]
                        observed = [float(value) for value in values if value is not None]
                        if not observed or not panel.valid[row, column]:
                            continue
                        baseline = max(float(np.mean(observed)), 1e-12)
                        ratio = max(float(panel.volume[row, column]) / baseline, 0.0)
                        volume_baseline[row, column] = baseline
                        relative_volume[row, column] = ratio
                        marks[row, column] = np.sqrt(min(ratio, self.config.volume_mark_cap))

            for column in range(len(panel.symbols)):
                session_velocity = raw_velocity[rows, column][raw_valid[rows, column]]
                if len(session_velocity) >= self.config.minimum_minutes_per_session:
                    velocity_history[column].append(session_velocity.copy())
                observed = panel.valid[rows, column]
                if int(observed.sum()) >= self.config.minimum_minutes_per_session:
                    volume_history[column].append(
                        {
                            int(slot): float(value)
                            for slot, value in zip(
                                slots[observed], panel.volume[rows, column][observed], strict=True
                            )
                        }
                    )

        bull = event_valid & (standardized >= self.config.activation_threshold)
        bear = event_valid & (standardized <= -self.config.activation_threshold)
        return ActivationPanel(
            symbols=panel.symbols,
            timestamps=panel.timestamps.copy(),
            standardized_velocity=standardized,
            velocity_scale=velocity_scale,
            relative_volume=relative_volume,
            volume_baseline=volume_baseline,
            marks=marks,
            bull=bull,
            bear=bear,
            valid=event_valid,
        )
