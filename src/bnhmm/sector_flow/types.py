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


@dataclass(frozen=True, slots=True)
class SectorFactorParameters:
    loadings: NDArray[np.float64]
    residual_variance: NDArray[np.float64]
    persistence: float
    process_variance: float

    def __post_init__(self) -> None:
        if self.loadings.ndim != 1 or self.residual_variance.shape != self.loadings.shape:
            raise ValueError("factor parameters must contain one value per symbol")
        if np.any(self.residual_variance <= 0.0) or self.process_variance <= 0.0:
            raise ValueError("factor variances must be positive")
        if not -0.999 <= self.persistence <= 0.999:
            raise ValueError("factor persistence is outside the stable range")


@dataclass(frozen=True, slots=True)
class SectorFlowPosterior:
    symbols: tuple[str, ...]
    timestamps: NDArray[np.datetime64]
    factor_mean: NDArray[np.float64]
    factor_variance: NDArray[np.float64]
    leave_one_out_mean: NDArray[np.float64]
    leave_one_out_variance: NDArray[np.float64]
    activity: NDArray[np.float64]
    breadth: NDArray[np.float64]
    coherence: NDArray[np.float64]
    dispersion: NDArray[np.float64]
    flow_strength: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        rows = len(self.timestamps)
        if self.leave_one_out_mean.shape != (rows, len(self.symbols)):
            raise ValueError("leave-one-out means do not match the sector panel")
        if self.leave_one_out_variance.shape != self.leave_one_out_mean.shape:
            raise ValueError("leave-one-out variances do not match their means")
        vectors = (
            self.factor_mean,
            self.factor_variance,
            self.activity,
            self.breadth,
            self.coherence,
            self.dispersion,
            self.flow_strength,
            self.valid,
        )
        if any(item.shape != (rows,) for item in vectors):
            raise ValueError("sector-flow posterior vectors must match timestamps")


@dataclass(frozen=True, slots=True)
class FlowAttributionParameters:
    symbols: tuple[str, ...]
    lags: NDArray[np.int64]
    leader_coefficient_mean: NDArray[np.float64]
    leader_coefficient_covariance: NDArray[np.float64]
    response_coefficient_mean: NDArray[np.float64]
    response_coefficient_covariance: NDArray[np.float64]
    leader_residual_scale: NDArray[np.float64]
    response_residual_scale: NDArray[np.float64]
    leader_prior_mean: NDArray[np.float64]
    response_prior_mean: NDArray[np.float64]

    def __post_init__(self) -> None:
        symbols = len(self.symbols)
        coefficients = len(self.lags)
        matrix_shape = (symbols, coefficients)
        covariance_shape = (symbols, coefficients, coefficients)
        if self.leader_coefficient_mean.shape != matrix_shape:
            raise ValueError("leader coefficients do not match symbols and lags")
        if self.response_coefficient_mean.shape != matrix_shape:
            raise ValueError("response coefficients do not match symbols and lags")
        if self.leader_coefficient_covariance.shape != covariance_shape:
            raise ValueError("leader covariance does not match coefficients")
        if self.response_coefficient_covariance.shape != covariance_shape:
            raise ValueError("response covariance does not match coefficients")
        if self.leader_prior_mean.shape != (coefficients,):
            raise ValueError("leader prior does not match lags")
        if self.response_prior_mean.shape != (coefficients,):
            raise ValueError("response prior does not match lags")
        if self.leader_residual_scale.shape != (symbols,):
            raise ValueError("leader residual scale does not match symbols")
        if self.response_residual_scale.shape != (symbols,):
            raise ValueError("response residual scale does not match symbols")


@dataclass(frozen=True, slots=True)
class StockFlowAttribution:
    symbol: str
    leader_probability: float
    leader_horizon_minutes: int
    leader_coefficient_mean: float
    leader_coefficient_std: float
    response_probability: float
    response_delay_minutes: int
    response_coefficient_mean: float
    response_coefficient_std: float
    coupling_mean: float
    coupling_std: float
    residual_scale: float


@dataclass(frozen=True, slots=True)
class FlowAttributionPosterior:
    symbols: tuple[str, ...]
    timestamps: NDArray[np.datetime64]
    leader_mean: NDArray[np.float64]
    leader_variance: NDArray[np.float64]
    response_mean: NDArray[np.float64]
    response_variance: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        shape = (len(self.timestamps), len(self.symbols))
        arrays = (
            self.leader_mean,
            self.leader_variance,
            self.response_mean,
            self.response_variance,
            self.valid,
        )
        if any(item.shape != shape for item in arrays):
            raise ValueError("attribution posterior arrays must share panel shape")
