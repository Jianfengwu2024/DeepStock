"""Low-dimensional Bayesian local-trend filter for daily reproduction numbers."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, log, sqrt
from typing import Iterable

import numpy as np
from scipy.special import digamma, polygamma

from .types import (
    CriticalFilterState,
    CriticalStateForecast,
    DailyBranchingObservation,
)


@dataclass(frozen=True, slots=True)
class CriticalStateFilter:
    slope_persistence: float = 0.75
    level_process_variance: float = 0.04
    slope_process_variance: float = 0.01
    gamma_prior_shape: float = 0.5
    gamma_prior_rate: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.slope_persistence < 1.0:
            raise ValueError("slope persistence must be stable")
        if self.level_process_variance <= 0.0 or self.slope_process_variance <= 0.0:
            raise ValueError("filter process variances must be positive")

    @property
    def transition(self) -> np.ndarray:
        return np.asarray([[1.0, 1.0], [0.0, self.slope_persistence]])

    @property
    def process_covariance(self) -> np.ndarray:
        return np.diag(
            [self.level_process_variance, self.slope_process_variance]
        )

    def initialize(
        self, observations: Iterable[DailyBranchingObservation]
    ) -> CriticalFilterState:
        state = CriticalFilterState(
            mean=np.asarray([log(0.8), 0.0], dtype=np.float64),
            covariance=np.diag([0.5, 0.1]).astype(np.float64),
            external_shape=self.gamma_prior_shape,
            external_rate=self.gamma_prior_rate,
        )
        for observation in observations:
            state = self.update(state, observation)
        return state

    def update(
        self,
        state: CriticalFilterState,
        observation: DailyBranchingObservation,
    ) -> CriticalFilterState:
        if (
            not np.isnat(state.last_session)
            and not np.isnat(observation.session)
            and observation.session <= state.last_session
        ):
            raise ValueError("critical observations must be chronological")
        transition = self.transition
        predicted_mean = transition @ state.mean
        predicted_covariance = (
            transition @ state.covariance @ transition.T
            + self.process_covariance
        )
        posterior_shape = self.gamma_prior_shape + observation.offspring
        posterior_rate = self.gamma_prior_rate + observation.parent_exposure
        measurement = float(digamma(posterior_shape) - np.log(posterior_rate))
        measurement_variance = max(float(polygamma(1, posterior_shape)), 1e-6)
        innovation_variance = predicted_covariance[0, 0] + measurement_variance
        gain = predicted_covariance[:, 0] / innovation_variance
        updated_mean = predicted_mean + gain * (measurement - predicted_mean[0])
        updated_covariance = predicted_covariance - np.outer(
            gain, predicted_covariance[0]
        )
        updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
        return CriticalFilterState(
            mean=updated_mean,
            covariance=updated_covariance,
            external_shape=state.external_shape + observation.external,
            external_rate=state.external_rate + 1.0,
            last_session=observation.session,
        )

    def predict(self, state: CriticalFilterState) -> CriticalStateForecast:
        mean = self.transition @ state.mean
        covariance = (
            self.transition @ state.covariance @ self.transition.T
            + self.process_covariance
        )
        variance = max(float(covariance[0, 0]), 1e-12)
        z = float(mean[0] / sqrt(variance))
        probability = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        external_mean = state.external_shape / state.external_rate
        external_variance = state.external_shape / state.external_rate**2
        return CriticalStateForecast(
            log_reproduction_mean=float(mean[0]),
            log_reproduction_variance=variance,
            slope_mean=float(mean[1]),
            slope_variance=max(float(covariance[1, 1]), 0.0),
            probability_supercritical=float(np.clip(probability, 0.0, 1.0)),
            external_mean=float(external_mean),
            external_variance=float(external_variance),
        )
