"""Leakage-safe rolling normalization for daily phase-space states."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import ProfileConfig
from ..types import BarBatch


@dataclass(frozen=True, slots=True)
class PhaseSpaceNormalization:
    sessions: int
    average_minute_volume: float
    location_scale: float
    price_velocity_mean: float
    price_velocity_scale: float
    relative_volume_mean: float
    relative_volume_scale: float
    volume_velocity_mean: float
    volume_velocity_scale: float


@dataclass(frozen=True, slots=True)
class NormalizedSession:
    relative_volume: NDArray[np.float64]
    relative_volume_level: NDArray[np.float64]
    price_velocity: NDArray[np.float64]
    volume_velocity: NDArray[np.float64]
    velocity_anchor_log_price: NDArray[np.float64]
    velocity_edge_valid: NDArray[np.bool_]

    def take(self, indices: NDArray[np.integer]) -> "NormalizedSession":
        return NormalizedSession(
            relative_volume=self.relative_volume[indices],
            relative_volume_level=self.relative_volume_level[indices],
            price_velocity=self.price_velocity[indices],
            volume_velocity=self.volume_velocity[indices],
            velocity_anchor_log_price=self.velocity_anchor_log_price[indices],
            velocity_edge_valid=self.velocity_edge_valid[indices],
        )


@dataclass(frozen=True, slots=True)
class _CompletedSession:
    timestamp: NDArray[np.datetime64]
    log_close: NDArray[np.float64]
    high: float
    low: float
    volume: NDArray[np.float64]


class RollingPhaseSpaceNormalizer:
    """Fit each session only from the symbol's preceding completed sessions."""

    def __init__(self, config: ProfileConfig) -> None:
        self.config = config
        self._history: dict[str, deque[_CompletedSession]] = defaultdict(
            lambda: deque(maxlen=config.relative_volume_lookback_sessions)
        )

    def snapshot(self, symbol: str) -> PhaseSpaceNormalization | None:
        history = self._history[symbol]
        if len(history) < self.config.relative_volume_min_observations:
            return None

        average_minute_volume = float(
            np.mean([item.volume.sum() for item in history])
            / self.config.trading_minutes_per_session
        )
        average_minute_volume = max(average_minute_volume, 1e-12)
        log_bin_width = float(np.log1p(self.config.bin_width_bps / 10_000.0))
        location_scale = max(
            float(np.mean([np.log(item.high / item.low) for item in history])),
            log_bin_width,
        )
        valid_edges = [
            self._continuous_minute_edges(item.timestamp) for item in history
        ]
        price_velocity = self._session_differences(
            [item.log_close for item in history], valid_edges
        )
        relative_levels = [
            np.log1p(
                np.clip(
                    item.volume / average_minute_volume,
                    0.0,
                    self.config.relative_volume_clip,
                )
            )
            for item in history
        ]
        relative_level = np.concatenate(relative_levels)
        volume_velocity = self._session_differences(
            relative_levels, valid_edges
        )
        price_mean, price_scale = self._location_scale(
            price_velocity, minimum=log_bin_width / 10.0
        )
        relative_mean, relative_scale = self._location_scale(
            relative_level, minimum=0.05
        )
        volume_mean, volume_scale = self._location_scale(
            volume_velocity, minimum=0.05
        )
        return PhaseSpaceNormalization(
            sessions=len(history),
            average_minute_volume=average_minute_volume,
            location_scale=location_scale,
            price_velocity_mean=price_mean,
            price_velocity_scale=price_scale,
            relative_volume_mean=relative_mean,
            relative_volume_scale=relative_scale,
            volume_velocity_mean=volume_mean,
            volume_velocity_scale=volume_scale,
        )

    def transform(
        self, bars: BarBatch, normalization: PhaseSpaceNormalization
    ) -> NormalizedSession:
        relative = np.clip(
            bars.volume / normalization.average_minute_volume,
            0.0,
            self.config.relative_volume_clip,
        )
        relative_log = np.log1p(relative)
        price_log = np.log(bars.close)
        valid_edges = self._continuous_minute_edges(bars.timestamp)
        price_velocity = np.zeros(len(bars), dtype=np.float64)
        volume_velocity = np.zeros(len(bars), dtype=np.float64)
        velocity_anchor = price_log.copy()
        if len(bars) > 1:
            price_velocity[1:] = np.diff(price_log)
            volume_velocity[1:] = np.diff(relative_log)
            velocity_anchor[1:] = (price_log[:-1] + price_log[1:]) / 2.0
        return NormalizedSession(
            relative_volume=relative,
            relative_volume_level=self._standardize(
                relative_log,
                normalization.relative_volume_mean,
                normalization.relative_volume_scale,
            ),
            price_velocity=self._standardize(
                price_velocity,
                normalization.price_velocity_mean,
                normalization.price_velocity_scale,
            ),
            volume_velocity=self._standardize(
                volume_velocity,
                normalization.volume_velocity_mean,
                normalization.volume_velocity_scale,
            ),
            velocity_anchor_log_price=velocity_anchor,
            velocity_edge_valid=valid_edges,
        )

    def update(self, bars: BarBatch) -> None:
        if not len(bars):
            return
        self._history[bars.symbol].append(
            _CompletedSession(
                timestamp=bars.timestamp.copy(),
                log_close=np.log(bars.close.copy()),
                high=float(bars.high.max()),
                low=float(bars.low.min()),
                volume=bars.volume.copy(),
            )
        )

    def observations(self, symbol: str) -> int:
        return len(self._history[symbol])

    def _standardize(
        self, values: NDArray[np.float64], mean: float, scale: float
    ) -> NDArray[np.float64]:
        return np.clip(
            (values - mean) / scale,
            -self.config.feature_clip,
            self.config.feature_clip,
        )

    @staticmethod
    def _session_differences(
        sessions: list[NDArray[np.float64]],
        valid_edges: list[NDArray[np.bool_]],
    ) -> NDArray[np.float64]:
        differences = [
            np.diff(item)[valid[1:]]
            for item, valid in zip(sessions, valid_edges, strict=True)
            if len(item) > 1 and np.any(valid[1:])
        ]
        return np.concatenate(differences) if differences else np.zeros(1)

    @staticmethod
    def _continuous_minute_edges(
        timestamp: NDArray[np.datetime64],
    ) -> NDArray[np.bool_]:
        valid = np.zeros(len(timestamp), dtype=np.bool_)
        if len(timestamp) > 1:
            elapsed = np.diff(timestamp.astype("datetime64[m]").astype(np.int64))
            valid[1:] = elapsed == 1
        return valid

    @staticmethod
    def _location_scale(
        values: NDArray[np.float64], *, minimum: float
    ) -> tuple[float, float]:
        mean = float(values.mean())
        scale = max(float(values.std()), minimum)
        return mean, scale
