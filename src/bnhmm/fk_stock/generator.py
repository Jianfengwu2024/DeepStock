"""Leakage-safe Bayesian estimation of a local return generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..data.dataset import ProfileObservations
from ..stock_transformer.conservation import (
    ConservationConfig,
    ConservationLaw,
    ConservationLawDiscoverer,
    DensityFrame,
    DensityProjector,
)


FloatArray = NDArray[np.float64]
FEATURE_NAMES = (
    "rolling_drift",
    "latest_matured_return",
    "rolling_diffusion",
    "signed_conservation_flow",
    "current_coherence",
    "density_center_change",
    "density_spread_change",
    "stochastic_current_fraction",
    "density_gradient_energy",
)


@dataclass(frozen=True, slots=True)
class FKGeneratorConfig:
    drift_lookback_sessions: int = 20
    ridge: float = 1.0
    minimum_return_history: int = 5
    minimum_fit_rows: int = 20
    minimum_noise_variance: float = 1e-4

    def __post_init__(self) -> None:
        if self.drift_lookback_sessions < 2:
            raise ValueError("drift lookback must contain at least two sessions")
        if not 2 <= self.minimum_return_history <= self.drift_lookback_sessions:
            raise ValueError("minimum return history must fit the drift lookback")
        if self.ridge <= 0 or self.minimum_fit_rows < 2:
            raise ValueError("generator regularization and sample count must be positive")
        if self.minimum_noise_variance <= 0:
            raise ValueError("minimum generator variance must be positive")


@dataclass(frozen=True, slots=True)
class FKFeatureRow:
    symbol: str
    signal_time: np.datetime64
    execution_time: np.datetime64
    target_time: np.datetime64
    target: int
    forward_return: float
    standardized_return: float
    return_location: float
    return_scale: float
    features: FloatArray
    stochastic_current_fraction: float
    current_coherence: float
    continuity_error: float
    matured_return_count: int

    def __post_init__(self) -> None:
        if self.features.shape != (len(FEATURE_NAMES),):
            raise ValueError("FK feature vector has an unexpected shape")
        if np.any(~np.isfinite(self.features)):
            raise ValueError("FK features must be finite")
        if self.target_time <= self.signal_time:
            raise ValueError("FK target must occur after its signal")


@dataclass(frozen=True, slots=True)
class BayesianGeneratorPosterior:
    coefficient_mean: FloatArray
    coefficient_covariance: FloatArray
    feature_location: FloatArray
    feature_scale: FloatArray
    noise_variance: float
    class_prior: tuple[float, ...]
    fit_rows: int
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        dimension = len(self.feature_names) + 1
        if self.coefficient_mean.shape != (dimension,):
            raise ValueError("generator coefficient vector has an unexpected shape")
        if self.coefficient_covariance.shape != (dimension, dimension):
            raise ValueError("generator covariance has an unexpected shape")
        if self.feature_location.shape != (dimension - 1,):
            raise ValueError("generator feature normalization is invalid")
        if self.feature_scale.shape != self.feature_location.shape:
            raise ValueError("generator feature scales are invalid")
        if self.noise_variance <= 0 or self.fit_rows < 2:
            raise ValueError("generator posterior requires fitted observations")

    def design_vector(self, row: FKFeatureRow) -> FloatArray:
        normalized = (row.features - self.feature_location) / self.feature_scale
        return np.concatenate((np.ones(1, dtype=np.float64), normalized))

    def as_json(self) -> dict[str, object]:
        payload = asdict(self)
        for key in (
            "coefficient_mean",
            "coefficient_covariance",
            "feature_location",
            "feature_scale",
        ):
            payload[key] = payload[key].tolist()
        return payload


class FKGenerator:
    """Estimate a Bayesian ridge generator from fully matured training labels."""

    def __init__(self, config: FKGeneratorConfig | None = None) -> None:
        self.config = config or FKGeneratorConfig()

    def fit(self, rows: Sequence[FKFeatureRow]) -> BayesianGeneratorPosterior:
        if len(rows) < self.config.minimum_fit_rows:
            raise ValueError(
                f"FK generator needs at least {self.config.minimum_fit_rows} rows"
            )
        features = np.asarray([row.features for row in rows], dtype=np.float64)
        targets = np.asarray([row.standardized_return for row in rows], dtype=np.float64)
        location = features.mean(axis=0)
        scale = features.std(axis=0, ddof=1)
        scale = np.where(scale > 1e-8, scale, 1.0)
        design = np.column_stack((np.ones(len(rows)), (features - location) / scale))

        penalty = np.eye(design.shape[1], dtype=np.float64) * self.config.ridge
        penalty[0, 0] = self.config.ridge * 1e-6
        initial = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
        residual = targets - design @ initial
        noise_variance = max(
            float(np.square(residual).sum() / max(len(rows) - design.shape[1], 1)),
            self.config.minimum_noise_variance,
        )
        precision = design.T @ design / noise_variance + penalty
        covariance = np.linalg.inv(precision)
        coefficient_mean = covariance @ (design.T @ targets / noise_variance)
        counts = np.bincount(
            np.asarray([row.target for row in rows], dtype=np.int64), minlength=4
        ).astype(np.float64)
        class_prior = tuple(float(value) for value in (counts + 1.0) / (counts.sum() + 4.0))
        return BayesianGeneratorPosterior(
            coefficient_mean=coefficient_mean,
            coefficient_covariance=covariance,
            feature_location=location,
            feature_scale=scale,
            noise_variance=noise_variance,
            class_prior=class_prior,
            fit_rows=len(rows),
        )

    def build_rows(
        self,
        observations: Sequence[ProfileObservations],
        law: ConservationLaw,
        conservation_config: ConservationConfig,
    ) -> list[FKFeatureRow]:
        projector = DensityProjector(conservation_config)
        discoverer = ConservationLawDiscoverer(conservation_config)
        rows: list[FKFeatureRow] = []
        for item in observations:
            frames = _frames(item, projector)
            for index in range(1, len(frames)):
                signal_time = item.signal_times[index]
                matured = np.flatnonzero(item.target_times < signal_time)
                matured = matured[-self.config.drift_lookback_sessions :]
                if len(matured) < self.config.minimum_return_history:
                    continue
                transition = discoverer.transform(frames[index - 1], frames[index], law)
                known_returns = item.standardized_returns[matured]
                current_l1 = max(float(np.abs(transition.current).sum()), 1e-10)
                signed_flow = float(transition.current.sum() / current_l1)
                coherence = float(abs(transition.current.sum()) / current_l1)
                stochastic_fraction = float(
                    np.linalg.norm(transition.stochastic_current)
                    / max(np.linalg.norm(transition.current), 1e-10)
                )
                grid = discoverer.grid
                previous_center = float(grid @ transition.previous_density)
                current_center = float(grid @ transition.density)
                previous_spread = float(
                    np.sqrt(np.maximum(((grid - previous_center) ** 2) @ transition.previous_density, 0.0))
                )
                current_spread = float(
                    np.sqrt(np.maximum(((grid - current_center) ** 2) @ transition.density, 0.0))
                )
                continuity_error = float(np.sqrt(np.mean(np.square(transition.continuity_residual))))
                features = np.asarray(
                    (
                        known_returns.mean(),
                        known_returns[-1],
                        known_returns.std(ddof=1),
                        signed_flow,
                        coherence,
                        current_center - previous_center,
                        current_spread - previous_spread,
                        stochastic_fraction,
                        np.sqrt(np.mean(np.square(transition.density_gradient))),
                    ),
                    dtype=np.float64,
                )
                rows.append(
                    FKFeatureRow(
                        symbol=item.symbol,
                        signal_time=signal_time,
                        execution_time=item.execution_times[index],
                        target_time=item.target_times[index],
                        target=int(item.targets[index]),
                        forward_return=float(item.forward_returns[index]),
                        standardized_return=float(item.standardized_returns[index]),
                        return_location=float(item.return_locations[index]),
                        return_scale=float(item.return_scales[index]),
                        features=features,
                        stochastic_current_fraction=stochastic_fraction,
                        current_coherence=coherence,
                        continuity_error=continuity_error,
                        matured_return_count=len(matured),
                    )
                )
        return rows


def _frames(
    observations: ProfileObservations, projector: DensityProjector
) -> list[DensityFrame]:
    return [
        projector.project(
            observations.symbol,
            observations.signal_times[index],
            observations.profiles[index][:, 3].astype(np.float64),
            observations.profiles[index][:, 1].astype(np.float64),
        )
        for index in range(len(observations))
    ]
