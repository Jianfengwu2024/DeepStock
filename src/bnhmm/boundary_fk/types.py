"""Immutable contracts for the BoundaryFlowFK pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class NormalizedBoundaryPanel:
    symbols: tuple[str, ...]
    timestamps: NDArray[np.datetime64]
    raw_velocity: NDArray[np.float64]
    standardized_velocity: NDArray[np.float64]
    relative_volume: NDArray[np.float64]
    velocity_scale: NDArray[np.float64]
    minute_volume_baseline: NDArray[np.float64]
    raw_velocity_valid: NDArray[np.bool_]
    velocity_valid: NDArray[np.bool_]
    density_valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        shape = (len(self.timestamps), len(self.symbols))
        arrays = (
            self.raw_velocity,
            self.standardized_velocity,
            self.relative_volume,
            self.velocity_scale,
            self.minute_volume_baseline,
            self.raw_velocity_valid,
            self.velocity_valid,
            self.density_valid,
        )
        if any(item.shape != shape for item in arrays):
            raise ValueError("boundary normalization arrays must share panel shape")
        if np.any(self.raw_velocity_valid & ~np.isfinite(self.raw_velocity)):
            raise ValueError("valid raw velocities must be finite")
        if np.any(self.velocity_valid & ~np.isfinite(self.standardized_velocity)):
            raise ValueError("valid standardized velocities must be finite")
        if np.any(self.density_valid & ~np.isfinite(self.relative_volume)):
            raise ValueError("valid relative volume must be finite")
        if np.any(self.density_valid & (self.relative_volume < 0.0)):
            raise ValueError("relative volume cannot be negative")


@dataclass(frozen=True, slots=True)
class MonthlyIndexWeights:
    month: np.datetime64
    weight_date: np.datetime64
    symbols: tuple[str, ...]
    weights: NDArray[np.float64]
    training_cutoff: np.datetime64
    excluded_symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.month != self.month.astype("datetime64[M]"):
            raise ValueError("index weight month must use monthly precision")
        if len(self.symbols) < 1 or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("index weight symbols must be unique and non-empty")
        if self.weights.shape != (len(self.symbols),):
            raise ValueError("index weights must align with symbols")
        if np.any(~np.isfinite(self.weights)) or np.any(self.weights <= 0.0):
            raise ValueError("index weights must be finite and positive")
        if not np.isclose(float(self.weights.sum()), 1.0, atol=1e-12):
            raise ValueError("index weights must sum to one")
        month_start = self.month.astype("datetime64[D]")
        if not self.weight_date < month_start or not self.training_cutoff < month_start:
            raise ValueError("index weights must be fixed before their effective month")


@dataclass(frozen=True, slots=True)
class SyntheticSectorIndex:
    timestamps: NDArray[np.datetime64]
    velocity: NDArray[np.float64]
    valid: NDArray[np.bool_]
    active_weight: NDArray[np.float64]
    active_fraction: NDArray[np.float64]
    monthly_weights: tuple[MonthlyIndexWeights, ...]
    daily_sessions: NDArray[np.datetime64]
    daily_open_to_close_return: NDArray[np.float64]
    daily_valid: NDArray[np.bool_]
    daily_active_weight: NDArray[np.float64]
    daily_active_fraction: NDArray[np.float64]
    excluded_symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        minute_shape = (len(self.timestamps),)
        minute_arrays = (
            self.velocity,
            self.valid,
            self.active_weight,
            self.active_fraction,
        )
        if any(item.shape != minute_shape for item in minute_arrays):
            raise ValueError("synthetic index minute arrays must align")
        daily_shape = (len(self.daily_sessions),)
        daily_arrays = (
            self.daily_open_to_close_return,
            self.daily_valid,
            self.daily_active_weight,
            self.daily_active_fraction,
        )
        if any(item.shape != daily_shape for item in daily_arrays):
            raise ValueError("synthetic index daily arrays must align")
        if np.any(self.valid & ~np.isfinite(self.velocity)):
            raise ValueError("valid synthetic minute returns must be finite")
        if np.any(self.daily_valid & ~np.isfinite(self.daily_open_to_close_return)):
            raise ValueError("valid synthetic daily returns must be finite")
        if np.any((self.active_weight < 0.0) | (self.active_weight > 1.0 + 1e-12)):
            raise ValueError("active minute weight must lie in [0, 1]")
        if np.any((self.active_fraction < 0.0) | (self.active_fraction > 1.0)):
            raise ValueError("active minute fraction must lie in [0, 1]")

    def weights_for(self, month: str | np.datetime64) -> MonthlyIndexWeights:
        requested = np.datetime64(month, "M")
        for item in self.monthly_weights:
            if item.month == requested:
                return item
        raise KeyError(f"synthetic index weights unavailable for {requested}")

    def weights_at(self, timestamp: str | np.datetime64) -> NDArray[np.float64]:
        return self.weights_for(np.datetime64(timestamp, "M")).weights.copy()
