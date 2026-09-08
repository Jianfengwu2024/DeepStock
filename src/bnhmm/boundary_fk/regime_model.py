"""Two-state Bayesian continuation/reversal regime for path efficiency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import TwoBranchDailyStates, TwoBranchLabels


@dataclass(frozen=True, slots=True)
class BayesianPathRegimeModelConfig:
    minimum_rows: int = 30
    neutral_sigma: float = 0.35
    grid_points: int = 25
    stay_prior_alpha: float = 8.0
    stay_prior_beta: float = 2.0
    emission_prior_alpha: float = 8.0
    emission_prior_beta: float = 2.0

    def __post_init__(self) -> None:
        if self.minimum_rows < 10 or self.neutral_sigma <= 0.0:
            raise ValueError("path regime fitting settings are invalid")
        if not 11 <= self.grid_points <= 101:
            raise ValueError("path regime grid points must lie in [11, 101]")
        priors = (
            self.stay_prior_alpha,
            self.stay_prior_beta,
            self.emission_prior_alpha,
            self.emission_prior_beta,
        )
        if any(value <= 0.0 for value in priors):
            raise ValueError("path regime beta priors must be positive")


@dataclass(frozen=True, slots=True)
class BayesianPathRegimePosterior:
    branch: int
    training_cutoff: np.datetime64
    fit_rows: int
    continuation_probability: float
    stay_probability: float
    emission_reliability: float
    next_agreement_probability: float
    next_agreement_probability_std: float

    def __post_init__(self) -> None:
        probabilities = (
            self.continuation_probability,
            self.stay_probability,
            self.emission_reliability,
            self.next_agreement_probability,
        )
        if self.branch < 0 or self.fit_rows < 1:
            raise ValueError("path regime posterior row counts are invalid")
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("path regime posterior probabilities are invalid")
        if self.next_agreement_probability_std < 0.0:
            raise ValueError("path regime posterior uncertainty cannot be negative")

    def bullish_probability(self, states: TwoBranchDailyStates, row: int) -> float:
        if row < 0 or row >= len(states.sessions):
            raise IndexError("path regime forecast row is out of range")
        if not states.valid[row, self.branch]:
            raise ValueError("cannot forecast an invalid path regime state")
        path = float(states.feature("signed_path_efficiency")[row, self.branch])
        if not np.isfinite(path):
            raise ValueError("path efficiency must be finite")
        if path > 0.0:
            return self.next_agreement_probability
        if path < 0.0:
            return 1.0 - self.next_agreement_probability
        return 0.5


class BayesianPathRegimeModel:
    def __init__(self, config: BayesianPathRegimeModelConfig | None = None) -> None:
        self.config = config or BayesianPathRegimeModelConfig()

    def fit(
        self,
        states: TwoBranchDailyStates,
        labels: TwoBranchLabels,
        *,
        branch: int,
        training_cutoff: np.datetime64,
    ) -> BayesianPathRegimePosterior:
        if branch < 0 or branch >= states.values.shape[1]:
            raise ValueError("path regime branch index is outside the state tensor")
        if not np.array_equal(states.sessions, labels.signal_sessions):
            raise ValueError("path regime states and labels must align")
        cutoff = np.datetime64(training_cutoff, "ns")
        path = states.feature("signed_path_efficiency")[:, branch]
        selected = (
            states.valid[:, branch]
            & labels.valid[:, branch]
            & (labels.target_time <= cutoff)
            & (np.abs(path) > 1e-12)
            & (
                np.abs(labels.next_return[:, branch])
                > self.config.neutral_sigma
                * labels.historical_volatility[:, branch]
            )
        )
        if int(selected.sum()) < self.config.minimum_rows:
            raise ValueError("insufficient significant returns for path regime fit")
        agreement = np.sign(labels.next_return[selected, branch]) == np.sign(
            path[selected]
        )

        grid = np.linspace(0.51, 0.99, self.config.grid_points)
        log_weights: list[float] = []
        continuation: list[float] = []
        next_agreement: list[float] = []
        stay_values: list[float] = []
        emission_values: list[float] = []
        for stay in grid:
            stay_log_prior = _beta_log_kernel(
                stay,
                self.config.stay_prior_alpha,
                self.config.stay_prior_beta,
            )
            for reliability in grid:
                probability = 0.5
                log_likelihood = 0.0
                for agrees in agreement:
                    prior_probability = (
                        probability * stay
                        + (1.0 - probability) * (1.0 - stay)
                    )
                    continuation_likelihood = (
                        reliability if agrees else 1.0 - reliability
                    )
                    reversal_likelihood = (
                        1.0 - reliability if agrees else reliability
                    )
                    evidence = (
                        prior_probability * continuation_likelihood
                        + (1.0 - prior_probability) * reversal_likelihood
                    )
                    log_likelihood += float(np.log(max(evidence, 1e-300)))
                    probability = (
                        prior_probability * continuation_likelihood / evidence
                    )
                predicted_continuation = (
                    probability * stay + (1.0 - probability) * (1.0 - stay)
                )
                predicted_agreement = (
                    predicted_continuation * reliability
                    + (1.0 - predicted_continuation) * (1.0 - reliability)
                )
                log_weights.append(
                    log_likelihood
                    + stay_log_prior
                    + _beta_log_kernel(
                        reliability,
                        self.config.emission_prior_alpha,
                        self.config.emission_prior_beta,
                    )
                )
                continuation.append(predicted_continuation)
                next_agreement.append(predicted_agreement)
                stay_values.append(stay)
                emission_values.append(reliability)

        log_weight = np.asarray(log_weights, dtype=np.float64)
        log_weight -= float(np.max(log_weight))
        weight = np.exp(log_weight)
        weight /= float(np.sum(weight))
        agreement_values = np.asarray(next_agreement, dtype=np.float64)
        agreement_mean = float(weight @ agreement_values)
        agreement_variance = float(
            weight @ np.square(agreement_values - agreement_mean)
        )
        return BayesianPathRegimePosterior(
            branch=branch,
            training_cutoff=cutoff,
            fit_rows=int(selected.sum()),
            continuation_probability=float(
                weight @ np.asarray(continuation, dtype=np.float64)
            ),
            stay_probability=float(weight @ np.asarray(stay_values, dtype=np.float64)),
            emission_reliability=float(
                weight @ np.asarray(emission_values, dtype=np.float64)
            ),
            next_agreement_probability=agreement_mean,
            next_agreement_probability_std=float(np.sqrt(agreement_variance)),
        )


def _beta_log_kernel(value: float, alpha: float, beta: float) -> float:
    return (alpha - 1.0) * float(np.log(value)) + (beta - 1.0) * float(
        np.log1p(-value)
    )
