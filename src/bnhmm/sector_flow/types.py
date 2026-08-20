"""Typed contracts for synchronized sector-flow data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SectorMinutePanel:
    symbols: tuple[str, ...]
    timestamps: NDArray[np.datetime64]
    close: NDArray[np.float64]
    volume: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        shape = (len(self.timestamps), len(self.symbols))
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("sector symbols must be unique and non-empty")
        if self.close.shape != shape or self.volume.shape != shape or self.valid.shape != shape:
            raise ValueError("sector panel arrays must align with timestamps and symbols")
        if len(self.timestamps) > 1 and np.any(self.timestamps[1:] <= self.timestamps[:-1]):
            raise ValueError("sector panel timestamps must be strictly increasing")
        if np.any(self.valid & (~np.isfinite(self.close) | ~np.isfinite(self.volume))):
            raise ValueError("valid sector panel cells must contain finite values")
        if np.any(self.valid & ((self.close <= 0.0) | (self.volume < 0.0))):
            raise ValueError("valid sector panel prices and volumes are invalid")

    def log_returns(self) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        returns = np.full(self.close.shape, np.nan, dtype=np.float64)
        valid = np.zeros(self.valid.shape, dtype=np.bool_)
        if len(self.timestamps) < 2:
            return returns, valid
        minute = self.timestamps.astype("datetime64[m]").astype(np.int64)
        continuous = np.diff(minute) == 1
        edges = self.valid[1:] & self.valid[:-1] & continuous[:, None]
        rows, columns = np.nonzero(edges)
        destination = rows + 1
        returns[destination, columns] = np.log(
            self.close[destination, columns] / self.close[rows, columns]
        )
        valid[destination, columns] = True
        return returns, valid


@dataclass(frozen=True, slots=True)
class NormalizedSectorPanel:
    symbols: tuple[str, ...]
    timestamps: NDArray[np.datetime64]
    raw_velocity: NDArray[np.float64]
    velocity: NDArray[np.float64]
    activity: NDArray[np.float64]
    velocity_scale: NDArray[np.float64]
    volume_baseline: NDArray[np.float64]
    valid: NDArray[np.bool_]
    activity_valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        shape = (len(self.timestamps), len(self.symbols))
        arrays = (
            self.raw_velocity,
            self.velocity,
            self.activity,
            self.velocity_scale,
            self.volume_baseline,
            self.valid,
            self.activity_valid,
        )
        if any(item.shape != shape for item in arrays):
            raise ValueError("normalized sector arrays must share panel shape")
        if np.any(self.valid & ~np.isfinite(self.velocity)):
            raise ValueError("valid normalized velocities must be finite")
        if np.any(self.activity_valid & ~np.isfinite(self.activity)):
            raise ValueError("valid normalized activities must be finite")

