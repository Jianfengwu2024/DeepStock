"""Typed contracts shared by the percolation branching pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


Direction = Literal["bull", "bear"]
PhaseName = Literal["bear", "none", "conflict", "bull"]
PHASE_NAMES: tuple[PhaseName, ...] = ("bear", "none", "conflict", "bull")


@dataclass(frozen=True, slots=True)
class ActivationConfig:
    velocity_lookback_sessions: int = 10
    volume_lookback_sessions: int = 20
    minimum_history_sessions: int = 10
    minimum_minutes_per_session: int = 200
    activation_threshold: float = 1.5
    volume_mark_cap: float = 3.0
    minimum_velocity_scale: float = 1e-6

    def __post_init__(self) -> None:
        if self.velocity_lookback_sessions < self.minimum_history_sessions:
            raise ValueError("velocity lookback cannot be shorter than minimum history")
        if self.volume_lookback_sessions < self.minimum_history_sessions:
            raise ValueError("volume lookback cannot be shorter than minimum history")
        if self.minimum_history_sessions < 2:
            raise ValueError("minimum event history must contain at least two sessions")
        if self.minimum_minutes_per_session < 2:
            raise ValueError("minimum session observations must exceed one")
        if self.activation_threshold <= 0.0:
            raise ValueError("activation threshold must be positive")
        if self.volume_mark_cap <= 1.0:
            raise ValueError("volume mark cap must exceed one")
        if self.minimum_velocity_scale <= 0.0:
            raise ValueError("minimum velocity scale must be positive")


@dataclass(frozen=True, slots=True)
class ActivationPanel:
    symbols: tuple[str, ...]
    timestamps: NDArray[np.datetime64]
    standardized_velocity: NDArray[np.float64]
    velocity_scale: NDArray[np.float64]
    relative_volume: NDArray[np.float64]
    volume_baseline: NDArray[np.float64]
    marks: NDArray[np.float64]
    bull: NDArray[np.bool_]
    bear: NDArray[np.bool_]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        shape = (len(self.timestamps), len(self.symbols))
        arrays = (
            self.standardized_velocity,
            self.velocity_scale,
            self.relative_volume,
            self.volume_baseline,
            self.marks,
            self.bull,
            self.bear,
            self.valid,
        )
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("activation symbols must be unique and non-empty")
        if any(item.shape != shape for item in arrays):
            raise ValueError("activation arrays must align with timestamps and symbols")
        if np.any(self.bull & self.bear):
            raise ValueError("one minute cannot be both bull and bear activated")
        if np.any((self.bull | self.bear) & ~self.valid):
            raise ValueError("activation events must be valid")
        if np.any(self.valid & ~np.isfinite(self.standardized_velocity)):
            raise ValueError("valid activation velocities must be finite")
        if np.any((self.bull | self.bear) & (~np.isfinite(self.marks) | (self.marks <= 0))):
            raise ValueError("activation marks must be finite and positive")

    @property
    def sessions(self) -> NDArray[np.datetime64]:
        return np.unique(self.timestamps.astype("datetime64[D]"))

    def direction(self, name: Direction) -> NDArray[np.bool_]:
        if name == "bull":
            return self.bull
        if name == "bear":
            return self.bear
        raise ValueError(f"unknown percolation direction: {name}")
