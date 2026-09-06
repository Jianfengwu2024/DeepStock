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


@dataclass(frozen=True, slots=True)
class ReachabilityResult:
    giant_fraction: float
    source_membership: NDArray[np.bool_]
    event_density: float
    edge_count: int

    def __post_init__(self) -> None:
        if self.source_membership.ndim != 2 or (
            self.source_membership.shape[0] != self.source_membership.shape[1]
        ):
            raise ValueError("source membership must be a square matrix")
        if not 0.0 <= self.giant_fraction <= 1.0:
            raise ValueError("giant fraction must be a probability")
        if not 0.0 <= self.event_density <= 1.0:
            raise ValueError("event density must be a probability")
        if self.edge_count < 0:
            raise ValueError("edge count cannot be negative")


@dataclass(frozen=True, slots=True)
class NullThresholdTable:
    month: np.datetime64
    training_cutoff: np.datetime64
    direction: Direction
    permutations: int
    density_edges: NDArray[np.float64]
    thresholds: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.density_edges.ndim != 1 or self.thresholds.ndim != 1:
            raise ValueError("null threshold arrays must be one-dimensional")
        if len(self.density_edges) != len(self.thresholds) + 1:
            raise ValueError("density edges must bound every threshold bin")
        if np.any(np.diff(self.density_edges) < 0.0):
            raise ValueError("density edges must be ordered")
        if np.any((self.thresholds < 0.0) | (self.thresholds > 1.0)):
            raise ValueError("null thresholds must be probabilities")
        if self.permutations < 1:
            raise ValueError("at least one null permutation is required")
        if self.training_cutoff >= self.month.astype("datetime64[M]").astype("datetime64[D]"):
            raise ValueError("null calibration must end before its target month")

    def threshold_for(self, event_density: float) -> float:
        if not np.isfinite(event_density) or event_density < 0.0:
            raise ValueError("event density must be finite and nonnegative")
        index = int(
            np.searchsorted(self.density_edges[1:-1], event_density, side="right")
        )
        return float(self.thresholds[min(index, len(self.thresholds) - 1)])
