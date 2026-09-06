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


@dataclass(frozen=True, slots=True)
class MonthlyBackboneSupport:
    month: np.datetime64
    training_cutoff: np.datetime64
    symbols: tuple[str, ...]
    adjacency: NDArray[np.bool_]
    similarity: NDArray[np.float64]
    session_count: int
    completeness: NDArray[np.float64]

    def __post_init__(self) -> None:
        count = len(self.symbols)
        if self.adjacency.shape != (count, count):
            raise ValueError("backbone adjacency must align with symbols")
        if self.similarity.shape != self.adjacency.shape:
            raise ValueError("backbone similarity must align with adjacency")
        if self.completeness.shape != (count,):
            raise ValueError("backbone completeness must align with symbols")
        if np.any(np.diag(self.adjacency)) or not np.array_equal(
            self.adjacency, self.adjacency.T
        ):
            raise ValueError("backbone support must be symmetric without self edges")
        if self.session_count < 1:
            raise ValueError("backbone requires completed sessions")
        if self.training_cutoff >= self.month.astype("datetime64[M]").astype("datetime64[D]"):
            raise ValueError("backbone training must precede its target month")


@dataclass(frozen=True, slots=True)
class DailyBranchingObservation:
    parent_exposure: float
    offspring: float
    external: float
    self_events: float = 0.0
    session: np.datetime64 = np.datetime64("NaT")

    def __post_init__(self) -> None:
        values = (self.parent_exposure, self.offspring, self.external, self.self_events)
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("branching observations must be finite and nonnegative")

    @property
    def reproduction_estimate(self) -> float:
        return float(self.offspring / max(self.parent_exposure, 1e-12))


@dataclass(frozen=True, slots=True)
class BranchingPosterior:
    direction: Direction
    symbols: tuple[str, ...]
    adjacency: NDArray[np.bool_]
    mean: NDArray[np.float64]
    variance: NDArray[np.float64]
    normalized_mean: NDArray[np.float64]
    lag_weights: NDArray[np.float64]
    external_rate: NDArray[np.float64]
    self_rate: NDArray[np.float64]
    training_cutoff: np.datetime64

    def __post_init__(self) -> None:
        count = len(self.symbols)
        matrices = (self.adjacency, self.mean, self.variance, self.normalized_mean)
        if any(matrix.shape != (count, count) for matrix in matrices):
            raise ValueError("branching matrices must align with symbols")
        if self.external_rate.shape != (count,) or self.self_rate.shape != (count,):
            raise ValueError("branching rate vectors must align with symbols")
        if self.lag_weights.ndim != 1 or len(self.lag_weights) == 0:
            raise ValueError("branching lag weights are required")
        if not np.isclose(float(self.lag_weights.sum()), 1.0):
            raise ValueError("branching lag weights must sum to one")
        if np.any(self.mean[~self.adjacency] != 0.0):
            raise ValueError("branching mass must stay on candidate edges")

    def observe(
        self, events: ActivationPanel, session: np.datetime64
    ) -> DailyBranchingObservation:
        event_matrix = events.direction(self.direction)
        rows = np.flatnonzero(
            events.timestamps.astype("datetime64[D]")
            == np.datetime64(session, "D")
        )
        if len(rows) == 0:
            raise ValueError("observation session is unavailable")
        local = event_matrix[rows]
        exposure = 0.0
        offspring = 0.0
        self_events = 0.0
        external = 0.0
        for minute in range(len(rows)):
            common = int(local[minute].sum()) >= max(2, int(np.ceil(len(self.symbols) * 0.3)))
            for child in np.flatnonzero(local[minute]):
                if common:
                    external += 1.0
                    continue
                cross = 0.0
                own = 0.0
                for lag, lag_weight in enumerate(self.lag_weights, start=1):
                    if minute < lag:
                        continue
                    parent_events = local[minute - lag]
                    cross += float(
                        np.sum(self.mean[parent_events, child]) * lag_weight
                    )
                    own += float(parent_events[child]) * self.self_rate[child] * lag_weight
                    exposure += float(
                        np.sum(self.normalized_mean[parent_events, child]) * lag_weight
                    )
                total = self.external_rate[child] + own + cross
                if total <= 0.0:
                    external += 1.0
                else:
                    offspring += cross / total
                    self_events += own / total
                    external += self.external_rate[child] / total
        return DailyBranchingObservation(
            parent_exposure=exposure,
            offspring=offspring,
            external=external,
            self_events=self_events,
            session=np.datetime64(session, "D"),
        )


@dataclass(frozen=True, slots=True)
class CriticalFilterState:
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    external_shape: float
    external_rate: float
    last_session: np.datetime64 = np.datetime64("NaT")

    def __post_init__(self) -> None:
        if self.mean.shape != (2,) or self.covariance.shape != (2, 2):
            raise ValueError("critical filter state must contain log R and slope")
        if np.any(~np.isfinite(self.mean)) or np.any(~np.isfinite(self.covariance)):
            raise ValueError("critical filter state must be finite")
        if self.external_shape <= 0.0 or self.external_rate <= 0.0:
            raise ValueError("external Gamma state must be positive")


@dataclass(frozen=True, slots=True)
class CriticalStateForecast:
    log_reproduction_mean: float
    log_reproduction_variance: float
    slope_mean: float
    slope_variance: float
    probability_supercritical: float
    external_mean: float
    external_variance: float

    def __post_init__(self) -> None:
        values = (
            self.log_reproduction_mean,
            self.log_reproduction_variance,
            self.slope_mean,
            self.slope_variance,
            self.probability_supercritical,
            self.external_mean,
            self.external_variance,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("critical forecast values must be finite")
        if self.log_reproduction_variance < 0.0 or self.slope_variance < 0.0:
            raise ValueError("critical forecast variances cannot be negative")
        if not 0.0 <= self.probability_supercritical <= 1.0:
            raise ValueError("supercritical probability must be valid")
        if self.external_mean < 0.0 or self.external_variance < 0.0:
            raise ValueError("external forecast moments cannot be negative")


@dataclass(frozen=True, slots=True)
class PercolationPrediction:
    phase_probabilities: tuple[float, float, float, float]
    bull_reproduction_mean: float
    bear_reproduction_mean: float
    bull_supercritical_probability: float
    bear_supercritical_probability: float
    bull_giant_fraction_mean: float
    bear_giant_fraction_mean: float
    bull_membership_probability: NDArray[np.float64]
    bear_membership_probability: NDArray[np.float64]
    signal_time: np.datetime64
    available_at: np.datetime64
    target_session: np.datetime64
    training_cutoff: np.datetime64
    model_version: str = "percolation-branching-v1"

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.phase_probabilities, dtype=np.float64)
        if probabilities.shape != (4,) or np.any(probabilities < 0.0):
            raise ValueError("four phase probabilities are required")
        if not np.isclose(float(probabilities.sum()), 1.0, atol=1e-6):
            raise ValueError("phase probabilities must sum to one")
        if self.bull_membership_probability.shape != self.bear_membership_probability.shape:
            raise ValueError("directional membership probabilities must align")
        if np.any(
            (self.bull_membership_probability < 0.0)
            | (self.bull_membership_probability > 1.0)
            | (self.bear_membership_probability < 0.0)
            | (self.bear_membership_probability > 1.0)
        ):
            raise ValueError("membership probabilities must be valid")
        target_day = self.target_session.astype("datetime64[D]")
        if not (
            self.training_cutoff <= self.signal_time
            and self.signal_time < self.available_at
            and self.signal_time.astype("datetime64[D]") < target_day
            and self.available_at.astype("datetime64[D]") <= target_day
        ):
            raise ValueError("prediction timestamps must be causal")
        if not self.model_version:
            raise ValueError("prediction model version is required")

    @property
    def predicted_phase(self) -> PhaseName:
        return PHASE_NAMES[int(np.argmax(self.phase_probabilities))]


@dataclass(frozen=True, slots=True)
class PercolationDecision:
    open_position: bool
    symbols: tuple[str, ...]
    selected_symbols: tuple[str, ...]
    target_weights: tuple[float, ...]
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.target_weights) != len(self.symbols) or len(self.scores) != len(self.symbols):
            raise ValueError("decision vectors must align with symbols")
        if any(weight < 0.0 or not np.isfinite(weight) for weight in self.target_weights):
            raise ValueError("target weights must be finite and nonnegative")
        total = float(sum(self.target_weights))
        if total > 1.0 + 1e-9 or self.open_position != (total > 0.0):
            raise ValueError("decision open state must match target exposure")


@dataclass(frozen=True, slots=True)
class PercolationExecution:
    actual_weights: NDArray[np.float64]
    cash_weight: float
    gross_return: float
    net_return: float
    turnover: float
    failed_buy_orders: int
    failed_sell_orders: int

    def __post_init__(self) -> None:
        values = (self.cash_weight, self.gross_return, self.net_return, self.turnover)
        if any(not np.isfinite(value) for value in values):
            raise ValueError("execution values must be finite")
        if np.any(~np.isfinite(self.actual_weights)) or np.any(self.actual_weights < 0.0):
            raise ValueError("actual weights must be finite and nonnegative")
        if self.cash_weight < -1e-9 or self.turnover < 0.0:
            raise ValueError("execution cash and turnover cannot be negative")
        if self.failed_buy_orders < 0 or self.failed_sell_orders < 0:
            raise ValueError("failed order counts cannot be negative")


@dataclass(frozen=True, slots=True)
class PercolationEvaluationObservation:
    prediction: PercolationPrediction
    actual_phase: PhaseName
    decision: PercolationDecision
    execution: PercolationExecution


@dataclass(frozen=True, slots=True)
class PercolationMetrics:
    observations: int
    confusion_matrix: NDArray[np.int64]
    balanced_accuracy: float
    macro_f1: float
    precision: tuple[float, float, float, float]
    recall: tuple[float, float, float, float]
    brier_score: float
    log_loss: float
    ece: float
    gross_total_return: float
    net_total_return: float
    maximum_drawdown: float
    average_exposure: float
    average_turnover: float
    open_rate: float
    failed_buy_orders: int
    failed_sell_orders: int


@dataclass(frozen=True, slots=True)
class PhaseBaselines:
    majority: PhaseName
    persistence: PhaseName
    transition: PhaseName
    simple_reproduction: PhaseName
