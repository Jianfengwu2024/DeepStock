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
class BoundaryMinuteMoments:
    timestamps: NDArray[np.datetime64]
    water: NDArray[np.float64]
    momentum: NDArray[np.float64]
    velocity: NDArray[np.float64]
    mean_standardized_velocity: NDArray[np.float64]
    coherence: NDArray[np.float64]
    turbulence: NDArray[np.float64]
    breadth: NDArray[np.float64]
    concentration: NDArray[np.float64]
    effective_constituents: NDArray[np.float64]
    decoupling: NDArray[np.float64]
    thickness: NDArray[np.float64]
    thickness_slope: NDArray[np.float64]
    viscosity: NDArray[np.float64]
    active_weight: NDArray[np.float64]
    active_fraction: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        shape = (len(self.timestamps),)
        arrays = (
            self.water,
            self.momentum,
            self.velocity,
            self.mean_standardized_velocity,
            self.coherence,
            self.turbulence,
            self.breadth,
            self.concentration,
            self.effective_constituents,
            self.decoupling,
            self.thickness,
            self.thickness_slope,
            self.viscosity,
            self.active_weight,
            self.active_fraction,
            self.valid,
        )
        if any(item.shape != shape for item in arrays):
            raise ValueError("boundary moment vectors must align with timestamps")
        economic = arrays[:-3]
        if any(np.any(self.valid & ~np.isfinite(item)) for item in economic):
            raise ValueError("valid boundary moments must be finite")
        if np.any(self.valid & ((self.coherence < 0.0) | (self.coherence > 1.0))):
            raise ValueError("coherence must lie in [0, 1]")
        if np.any(self.valid & ((self.concentration <= 0.0) | (self.concentration > 1.0))):
            raise ValueError("concentration must lie in (0, 1]")
        if np.any(self.valid & ((self.viscosity < 0.0) | (self.viscosity > 2.0))):
            raise ValueError("viscosity must lie in [0, 2]")


@dataclass(frozen=True, slots=True)
class BoundaryDailyStates:
    sessions: NDArray[np.datetime64]
    topology_months: NDArray[np.datetime64]
    feature_names: tuple[str, ...]
    units: tuple[str, ...]
    values: NDArray[np.float64]
    valid: NDArray[np.bool_]
    branch_degraded: NDArray[np.bool_]

    def __post_init__(self) -> None:
        rows = len(self.sessions)
        columns = len(self.feature_names)
        if not self.feature_names or len(set(self.feature_names)) != columns:
            raise ValueError("daily boundary feature names must be unique and non-empty")
        if columns > 36 or len(self.units) != columns or any(not unit for unit in self.units):
            raise ValueError("daily boundary schema must contain at most 36 named units")
        if self.values.shape != (rows, columns):
            raise ValueError("daily boundary values do not match their schema")
        if any(item.shape != (rows,) for item in (self.topology_months, self.valid, self.branch_degraded)):
            raise ValueError("daily boundary metadata must align with sessions")
        if np.any(self.valid & ~np.all(np.isfinite(self.values), axis=1)):
            raise ValueError("valid daily boundary states must be finite")

    def feature(self, name: str) -> NDArray[np.float64]:
        try:
            column = self.feature_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return self.values[:, column]


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


@dataclass(frozen=True, slots=True)
class MonthlyBranchMembership:
    month: np.datetime64
    training_cutoff: np.datetime64
    branches: tuple[tuple[str, ...], tuple[str, ...]]

    def __post_init__(self) -> None:
        if self.month != self.month.astype("datetime64[M]"):
            raise ValueError("branch membership month must use monthly precision")
        if not self.training_cutoff < self.month.astype("datetime64[ns]"):
            raise ValueError("branch membership must be fixed before its month")
        flattened = self.branches[0] + self.branches[1]
        if not self.branches[0] or not self.branches[1]:
            raise ValueError("both frozen branches must contain symbols")
        if len(flattened) != len(set(flattened)):
            raise ValueError("a symbol cannot belong to both branches")


@dataclass(frozen=True, slots=True)
class TwoBranchDailyStates:
    sessions: NDArray[np.datetime64]
    topology_months: NDArray[np.datetime64]
    feature_names: tuple[str, ...]
    values: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        expected = (len(self.sessions), 2, len(self.feature_names))
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("two-branch feature names must be unique and non-empty")
        if self.values.shape != expected or self.valid.shape != expected[:2]:
            raise ValueError("two-branch states must use session, branch, feature axes")
        if self.topology_months.shape != (len(self.sessions),):
            raise ValueError("two-branch topology months must align with sessions")
        if np.any(self.valid & ~np.all(np.isfinite(self.values), axis=2)):
            raise ValueError("valid two-branch states must be finite")

    def feature(self, name: str) -> NDArray[np.float64]:
        try:
            column = self.feature_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return self.values[:, :, column]


@dataclass(frozen=True, slots=True)
class TwoBranchLabels:
    signal_sessions: NDArray[np.datetime64]
    target_time: NDArray[np.datetime64]
    next_return: NDArray[np.float64]
    historical_volatility: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        rows = len(self.signal_sessions)
        if self.target_time.shape != (rows,):
            raise ValueError("two-branch target times must align with sessions")
        expected = (rows, 2)
        arrays = (self.next_return, self.historical_volatility, self.valid)
        if any(item.shape != expected for item in arrays):
            raise ValueError("two-branch labels must align with both branches")
        if np.any(self.valid & ~np.isfinite(self.next_return)):
            raise ValueError("valid branch returns must be finite")
        if np.any(
            self.valid
            & (~np.isfinite(self.historical_volatility) | (self.historical_volatility <= 0.0))
        ):
            raise ValueError("valid branch volatility must be finite and positive")


@dataclass(frozen=True, slots=True)
class TwoBranchDataset:
    states: TwoBranchDailyStates
    labels: TwoBranchLabels
    memberships: tuple[MonthlyBranchMembership, ...]
    symbols: tuple[str, ...]
    member_next_returns: NDArray[np.float64]
    member_next_valid: NDArray[np.bool_]
    evaluation_start: np.datetime64 | None = None
    evaluation_end: np.datetime64 | None = None
    convection_strength: NDArray[np.float64] | None = None
    convection_valid: NDArray[np.bool_] | None = None
    convection_total_flow: NDArray[np.float64] | None = None
    convection_observations: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        rows = len(self.states.sessions)
        expected = (rows, len(self.symbols))
        branch_shape = (rows, 2)
        if self.convection_strength is None:
            object.__setattr__(
                self,
                "convection_strength",
                np.full(branch_shape, np.nan, dtype=np.float64),
            )
        if self.convection_valid is None:
            object.__setattr__(
                self,
                "convection_valid",
                np.zeros(branch_shape, dtype=np.bool_),
            )
        if self.convection_total_flow is None:
            object.__setattr__(
                self,
                "convection_total_flow",
                np.zeros(branch_shape, dtype=np.float64),
            )
        if self.convection_observations is None:
            object.__setattr__(
                self,
                "convection_observations",
                np.zeros(branch_shape, dtype=np.int64),
            )
        if self.member_next_returns.shape != expected or self.member_next_valid.shape != expected:
            raise ValueError("member next returns must align with sessions and symbols")
        if not np.array_equal(self.states.sessions, self.labels.signal_sessions):
            raise ValueError("two-branch states and labels must align")
        if np.any(self.member_next_valid & ~np.isfinite(self.member_next_returns)):
            raise ValueError("valid member next returns must be finite")
        convection_arrays = (
            self.convection_strength,
            self.convection_valid,
            self.convection_total_flow,
            self.convection_observations,
        )
        if any(item is None or item.shape != branch_shape for item in convection_arrays):
            raise ValueError("convection diagnostics must align with sessions and branches")
        if np.any(self.convection_valid & ~np.isfinite(self.convection_strength)):
            raise ValueError("valid convection strength must be finite")
        if np.any(
            self.convection_valid
            & ((self.convection_strength < 0.0) | (self.convection_strength > 1.0))
        ):
            raise ValueError("convection strength must lie in [0, 1]")
        if np.any(self.convection_total_flow < 0.0) or np.any(
            self.convection_observations < 0
        ):
            raise ValueError("convection diagnostics cannot be negative")
        if self.evaluation_start is not None and self.evaluation_end is not None:
            if np.datetime64(self.evaluation_end, "D") < np.datetime64(
                self.evaluation_start, "D"
            ):
                raise ValueError("evaluation end cannot precede evaluation start")
