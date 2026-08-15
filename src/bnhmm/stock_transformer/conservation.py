"""Leakage-safe discovery of a stochastic volume-density conservation law."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ConservationConfig:
    grid_points: int = 64
    ridge: float = 1e-3
    minimum_density: float = 1e-10

    def __post_init__(self) -> None:
        if self.grid_points < 16:
            raise ValueError("conservation grid requires at least sixteen points")
        if self.ridge < 0 or self.minimum_density <= 0:
            raise ValueError("conservation regularization must be non-negative")


@dataclass(frozen=True, slots=True)
class DensityFrame:
    symbol: str
    signal_time: np.datetime64
    density: FloatArray

    def __post_init__(self) -> None:
        if self.density.ndim != 1 or len(self.density) < 3:
            raise ValueError("density frames must be one-dimensional")
        if np.any(~np.isfinite(self.density)) or np.any(self.density < 0):
            raise ValueError("density must be finite and non-negative")
        if not np.isclose(float(self.density.sum()), 1.0, atol=1e-8):
            raise ValueError("density must have unit mass")


@dataclass(frozen=True, slots=True)
class ConservationLaw:
    diffusion: float
    drift: float
    residual_mean: float
    noise_scale: float
    continuity_rmse: float
    transition_count: int
    grid_points: int

    def __post_init__(self) -> None:
        if self.diffusion < 0 or self.noise_scale <= 0:
            raise ValueError("discovered diffusion and noise scale are invalid")
        if self.transition_count < 1 or self.grid_points < 3:
            raise ValueError("conservation law requires fitted transitions")

    def as_json(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConservationTransition:
    symbol: str
    signal_time: np.datetime64
    previous_density: FloatArray
    density: FloatArray
    density_change: FloatArray
    density_gradient: FloatArray
    current: FloatArray
    expected_current: FloatArray
    stochastic_current: FloatArray
    continuity_residual: FloatArray

    @property
    def mass_error(self) -> float:
        return float(abs(self.density_change.sum()))


class DensityProjector:
    """Project variable price bins onto a fixed normalized price coordinate."""

    def __init__(self, config: ConservationConfig | None = None) -> None:
        self.config = config or ConservationConfig()
        self.grid = np.linspace(-1.0, 1.0, self.config.grid_points)

    def project(
        self,
        symbol: str,
        signal_time: np.datetime64,
        coordinates: FloatArray,
        density: FloatArray,
    ) -> DensityFrame:
        coordinates = np.asarray(coordinates, dtype=np.float64)
        density = np.asarray(density, dtype=np.float64)
        if coordinates.ndim != 1 or density.shape != coordinates.shape:
            raise ValueError("coordinates and density must be aligned vectors")
        if len(coordinates) < 2 or np.any(np.diff(coordinates) < 0):
            raise ValueError("coordinates must be sorted and non-empty")
        projected = np.interp(self.grid, coordinates, density, left=0.0, right=0.0)
        projected = np.maximum(projected, 0.0)
        total = float(projected.sum())
        if total <= self.config.minimum_density:
            projected.fill(1.0 / len(projected))
        else:
            projected /= total
        return DensityFrame(
            symbol=symbol,
            signal_time=np.datetime64(signal_time, "ns"),
            density=projected,
        )


class ConservationLawDiscoverer:
    """Fit J = -D grad(rho) + v rho + xi with conditional mean-zero xi."""

    def __init__(self, config: ConservationConfig | None = None) -> None:
        self.config = config or ConservationConfig()
        self.grid = np.linspace(-1.0, 1.0, self.config.grid_points)
        self.dx = float(self.grid[1] - self.grid[0])

    def fit(self, series: Iterable[Sequence[DensityFrame]]) -> ConservationLaw:
        pairs = list(self._pairs(series))
        if not pairs:
            raise ValueError("conservation discovery requires adjacent density frames")
        normal = np.eye(2, dtype=np.float64) * self.config.ridge
        target = np.zeros(2, dtype=np.float64)
        continuity_square = 0.0
        continuity_count = 0
        prepared: list[tuple[FloatArray, FloatArray]] = []
        for previous, current in pairs:
            fields = self._raw_fields(previous.density, current.density)
            design = np.column_stack((-fields["gradient"], fields["mid_density"]))
            normal += design.T @ design
            target += design.T @ fields["current"]
            continuity_square += float(np.square(fields["continuity"]).sum())
            continuity_count += len(fields["continuity"])
            prepared.append((design, fields["current"]))
        coefficients = np.linalg.solve(normal, target)
        diffusion = max(float(coefficients[0]), 0.0)
        drift = float(coefficients[1])
        residuals = np.concatenate(
            [current - design @ np.asarray((diffusion, drift)) for design, current in prepared]
        )
        residual_mean = float(residuals.mean())
        centered = residuals - residual_mean
        noise_scale = max(float(centered.std(ddof=1)), 1e-6)
        return ConservationLaw(
            diffusion=diffusion,
            drift=drift,
            residual_mean=residual_mean,
            noise_scale=noise_scale,
            continuity_rmse=float(
                np.sqrt(continuity_square / max(continuity_count, 1))
            ),
            transition_count=len(pairs),
            grid_points=self.config.grid_points,
        )

    def transform(
        self,
        previous: DensityFrame,
        current: DensityFrame,
        law: ConservationLaw,
    ) -> ConservationTransition:
        self._validate_pair(previous, current)
        if law.grid_points != self.config.grid_points:
            raise ValueError("law and density grid do not match")
        fields = self._raw_fields(previous.density, current.density)
        expected = (
            -law.diffusion * fields["gradient"]
            + law.drift * fields["mid_density"]
            + law.residual_mean
        )
        return ConservationTransition(
            symbol=current.symbol,
            signal_time=current.signal_time,
            previous_density=previous.density.copy(),
            density=current.density.copy(),
            density_change=fields["change"],
            density_gradient=fields["gradient"],
            current=fields["current"],
            expected_current=expected,
            stochastic_current=fields["current"] - expected,
            continuity_residual=fields["continuity"],
        )

    def transform_series(
        self, frames: Sequence[DensityFrame], law: ConservationLaw
    ) -> tuple[ConservationTransition, ...]:
        return tuple(
            self.transform(previous, current, law)
            for previous, current in zip(frames, frames[1:], strict=False)
        )

    def _raw_fields(self, previous: FloatArray, current: FloatArray) -> dict[str, FloatArray]:
        if previous.shape != (self.config.grid_points,) or current.shape != previous.shape:
            raise ValueError("density vectors do not match the conservation grid")
        change = current - previous
        edge_current = np.zeros(len(change) + 1, dtype=np.float64)
        edge_current[1:] = -np.cumsum(change)
        edge_current[-1] = 0.0
        cell_current = 0.5 * (edge_current[:-1] + edge_current[1:])
        mid_density = 0.5 * (previous + current)
        gradient = np.gradient(mid_density, self.dx)
        continuity = change + np.diff(edge_current)
        return {
            "change": change,
            "current": cell_current,
            "mid_density": mid_density,
            "gradient": gradient,
            "continuity": continuity,
        }

    def _pairs(
        self, series: Iterable[Sequence[DensityFrame]]
    ) -> Iterable[tuple[DensityFrame, DensityFrame]]:
        for frames in series:
            for previous, current in zip(frames, frames[1:], strict=False):
                self._validate_pair(previous, current)
                yield previous, current

    def _validate_pair(self, previous: DensityFrame, current: DensityFrame) -> None:
        if previous.symbol != current.symbol:
            raise ValueError("conservation transitions cannot cross symbols")
        if current.signal_time <= previous.signal_time:
            raise ValueError("density frames must be ordered in time")
        expected = (self.config.grid_points,)
        if previous.density.shape != expected or current.density.shape != expected:
            raise ValueError("density frame does not match configured grid")
