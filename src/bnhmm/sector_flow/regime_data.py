"""Causal sector-river images and next-session regime labels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..types import BarBatch
from .types import DailyRiverSlice


class SectorRegimeState(IntEnum):
    BEAR = 0
    FLAT = 1
    BULL = 2


@dataclass(frozen=True, slots=True)
class SectorRegimeDataConfig:
    context_sessions: int = 5
    image_height: int = 128
    image_minutes: int = 240
    volatility_lookback_sessions: int = 20
    volatility_minimum_observations: int = 10
    neutral_sigma: float = 0.35
    minimum_return_scale: float = 0.002
    feature_clip: float = 6.0

    def __post_init__(self) -> None:
        if self.context_sessions < 2:
            raise ValueError("sector context must contain at least two sessions")
        if self.image_height < 2 or self.image_minutes < 2:
            raise ValueError("sector image dimensions must be at least two")
        if self.volatility_lookback_sessions < 2:
            raise ValueError("volatility lookback must contain at least two sessions")
        if not 2 <= self.volatility_minimum_observations <= self.volatility_lookback_sessions:
            raise ValueError("volatility observations must fit the lookback")
        if self.neutral_sigma <= 0.0 or self.minimum_return_scale <= 0.0:
            raise ValueError("regime label scales must be positive")
        if self.feature_clip <= 0.0:
            raise ValueError("sector image feature clip must be positive")


@dataclass(frozen=True, slots=True)
class SectorStateImage:
    session: np.datetime64
    symbols: tuple[str, ...]
    values: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.values.shape[0] != 4:
            raise ValueError("sector state image must have four channels")
        if np.any(~np.isfinite(self.values)):
            raise ValueError("sector state image must be finite")


@dataclass(frozen=True, slots=True)
class SectorRegimeSample:
    images: NDArray[np.float32]
    next_image: NDArray[np.float32]
    target_state: SectorRegimeState
    target_return: float
    return_scale: float
    signal_time: np.datetime64
    available_at: np.datetime64
    target_time: np.datetime64

    def __post_init__(self) -> None:
        if self.images.ndim != 4 or self.images.shape[1:] != self.next_image.shape:
            raise ValueError("sector regime context and target image shapes do not align")
        if not self.signal_time < self.available_at <= self.target_time:
            raise ValueError("sector regime sample timing must be causal")
        if not np.isfinite(self.target_return) or self.return_scale <= 0.0:
            raise ValueError("sector regime return target is invalid")


def build_sector_state_image(
    river_slice: DailyRiverSlice,
    *,
    height: int,
    minutes: int,
    feature_clip: float = 6.0,
) -> SectorStateImage:
    if height < len(river_slice.symbols) or minutes < len(river_slice.timestamps):
        raise ValueError("sector image dimensions cannot crop a river slice")
    values = np.zeros((4, height, minutes), dtype=np.float32)
    rows = len(river_slice.symbols)
    columns = len(river_slice.timestamps)
    valid = river_slice.valid
    velocity = np.where(valid, river_slice.velocity, 0.0)
    residual = np.where(valid, river_slice.residual_velocity, 0.0)
    activity = np.where(valid, np.maximum(river_slice.activity, 0.0), 0.0)
    values[0, :rows, :columns] = np.clip(velocity, -feature_clip, feature_clip)
    values[1, :rows, :columns] = np.clip(residual, -feature_clip, feature_clip)
    values[2, :rows, :columns] = np.clip(
        np.log1p(activity), 0.0, feature_clip
    )
    values[3, :rows, :columns] = valid.astype(np.float32)
    return SectorStateImage(
        session=np.datetime64(river_slice.session, "D"),
        symbols=river_slice.symbols,
        values=values,
    )


class SectorRegimeDatasetBuilder:
    def __init__(self, config: SectorRegimeDataConfig | None = None) -> None:
        self.config = config or SectorRegimeDataConfig()

    def build(
        self,
        river_slices: Sequence[DailyRiverSlice],
        index_bars: BarBatch,
    ) -> tuple[SectorRegimeSample, ...]:
        ordered = tuple(sorted(river_slices, key=lambda item: item.session))
        if len(ordered) <= self.config.context_sessions:
            return ()
        images = tuple(
            build_sector_state_image(
                item,
                height=self.config.image_height,
                minutes=self.config.image_minutes,
                feature_clip=self.config.feature_clip,
            )
            for item in ordered
        )
        index_days = index_bars.timestamp.astype("datetime64[D]")
        index_positions = {day: position for position, day in enumerate(index_days)}
        intraday_returns = np.log(index_bars.close / index_bars.open)
        samples: list[SectorRegimeSample] = []
        for current in range(self.config.context_sessions - 1, len(ordered) - 1):
            target = current + 1
            target_day = np.datetime64(ordered[target].session, "D")
            target_position = index_positions.get(target_day)
            if target_position is None:
                continue
            history_start = max(
                0, target_position - self.config.volatility_lookback_sessions
            )
            history = intraday_returns[history_start:target_position]
            history = history[np.isfinite(history)]
            if len(history) < self.config.volatility_minimum_observations:
                continue
            scale = max(float(np.std(history)), self.config.minimum_return_scale)
            target_return = float(intraday_returns[target_position])
            threshold = self.config.neutral_sigma * scale
            state = (
                SectorRegimeState.BULL
                if target_return > threshold
                else SectorRegimeState.BEAR
                if target_return < -threshold
                else SectorRegimeState.FLAT
            )
            context_start = current - self.config.context_sessions + 1
            signal_slice = ordered[current]
            target_slice = ordered[target]
            samples.append(
                SectorRegimeSample(
                    images=np.stack(
                        [item.values for item in images[context_start : current + 1]]
                    ),
                    next_image=images[target].values,
                    target_state=state,
                    target_return=target_return,
                    return_scale=scale,
                    signal_time=np.datetime64(signal_slice.timestamps[-1], "ns"),
                    available_at=np.datetime64(target_slice.timestamps[0], "ns"),
                    target_time=np.datetime64(target_slice.timestamps[-1], "ns"),
                )
            )
        return tuple(samples)

