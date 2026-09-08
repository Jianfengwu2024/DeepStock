"""Long-only state machine for the whole-sector Bayesian FK strategy."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .feynman_kac import BoundaryFeynmanKacForecast


@dataclass(frozen=True, slots=True)
class BoundaryPolicyConfig:
    minimum_direction_probability: float = 0.51
    minimum_significant_move_probability: float = 0.50
    minimum_signal_to_noise: float = 0.10
    minimum_survival_probability: float = 0.35
    require_feature_evidence: bool = False

    def __post_init__(self) -> None:
        if not 0.5 <= self.minimum_direction_probability <= 1.0:
            raise ValueError("direction probability must lie in [0.5, 1]")
        if not 0.0 <= self.minimum_significant_move_probability <= 1.0:
            raise ValueError("significant move probability must lie in [0, 1]")
        if self.minimum_signal_to_noise < 0.0:
            raise ValueError("signal-to-noise gate must be non-negative")
        if not 0.0 <= self.minimum_survival_probability <= 1.0:
            raise ValueError("survival probability must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class BranchPositionState:
    exposure: float = 0.0
    consecutive_flat_signals: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.exposure <= 1.0:
            raise ValueError("branch exposure must lie in [0, 1]")
        if self.consecutive_flat_signals not in range(4):
            raise ValueError("flat signal count must be between 0 and 3")
        if self.exposure == 0.0 and self.consecutive_flat_signals:
            raise ValueError("an empty branch cannot retain a flat signal count")


@dataclass(frozen=True, slots=True)
class BoundaryPolicyDecision:
    open_position: bool
    gate_reason: str
    survival_conditional_bullish_probability: float
    selected_symbols: tuple[str, ...] = ()
    target_weights: tuple[float, ...] = ()
    branch_directions: tuple[str, ...] = ()
    held_branches: tuple[bool, ...] = ()
    branch_target_exposures: tuple[float, ...] = ()
    branch_position_states: tuple[BranchPositionState, ...] = ()


class BoundaryPolicy:
    def __init__(self, config: BoundaryPolicyConfig | None = None) -> None:
        self.config = config or BoundaryPolicyConfig()

    def decide(
        self,
        forecasts: tuple[BoundaryFeynmanKacForecast, ...],
        *,
        branches: tuple[tuple[str, ...], ...],
        neutral_thresholds: tuple[float, ...],
        previous_branch_states: tuple[BranchPositionState, ...] | None = None,
    ) -> BoundaryPolicyDecision:
        branch_count = len(branches)
        if not branch_count or len(forecasts) != branch_count or len(neutral_thresholds) != branch_count:
            raise ValueError("forecasts, branches, and thresholds must share a non-zero length")
        previous = previous_branch_states or (BranchPositionState(),) * branch_count
        if len(previous) != branch_count:
            raise ValueError("previous branch state must align with dynamic branches")
        directions = tuple(
            self._direction(item, threshold)
            for item, threshold in zip(forecasts, neutral_thresholds, strict=True)
        )
        if "bear" in directions:
            states = (BranchPositionState(),) * branch_count
        else:
            states = tuple(
                self._next_position_state(direction, old, branch_count)
                for direction, old in zip(
                    directions,
                    previous,
                    strict=True,
                )
            )
        exposures = tuple(state.exposure for state in states)
        selected: list[str] = []
        target_weights: list[float] = []
        for symbols, exposure in zip(branches, exposures, strict=True):
            if exposure <= 0.0:
                continue
            if not symbols:
                raise ValueError("a branch with positive exposure must contain symbols")
            selected.extend(symbols)
            target_weights.extend((exposure / len(symbols),) * len(symbols))
        if len(selected) != len(set(selected)):
            raise ValueError("a held symbol cannot appear in both branches")
        bullish = float(
            np.mean([item.raw_state_probabilities[2] for item in forecasts])
        )
        return BoundaryPolicyDecision(
            open_position=bool(selected),
            gate_reason="/".join(directions),
            survival_conditional_bullish_probability=bullish,
            selected_symbols=tuple(selected),
            target_weights=tuple(target_weights),
            branch_directions=directions,
            held_branches=tuple(value > 0.0 for value in exposures),
            branch_target_exposures=exposures,
            branch_position_states=states,
        )

    @staticmethod
    def _next_position_state(
        direction: str, previous: BranchPositionState, branch_count: int
    ) -> BranchPositionState:
        branch_budget = 1.0 / branch_count
        if direction == "bull":
            return BranchPositionState(exposure=branch_budget)
        if previous.exposure == 0.0:
            return BranchPositionState()
        # A neutral day only unwinds the capital allocated to this branch.
        exposure = max(0.0, previous.exposure - branch_budget / 4.0)
        if exposure <= 0.0:
            return BranchPositionState()
        return BranchPositionState(
            exposure=exposure,
            consecutive_flat_signals=previous.consecutive_flat_signals + 1,
        )

    def _direction(
        self,
        forecast: BoundaryFeynmanKacForecast,
        neutral_threshold: float,
    ) -> str:
        if not np.isfinite(neutral_threshold) or neutral_threshold < 0.0:
            raise ValueError("neutral threshold must be finite and non-negative")
        if (
            forecast.significant_move_probability
            < self.config.minimum_significant_move_probability
            or forecast.drift_signal_to_noise < self.config.minimum_signal_to_noise
            or forecast.trend_survival_probability
            < self.config.minimum_survival_probability
        ):
            return "flat"
        if (
            forecast.bullish_drift_probability
            >= self.config.minimum_direction_probability
        ):
            if self.config.require_feature_evidence and (
                forecast.feature_bullish_probability
                < forecast.minimum_feature_evidence_probability
            ):
                return "flat"
            return "bull"
        if (
            forecast.bullish_drift_probability
            <= 1.0 - self.config.minimum_direction_probability
        ):
            if self.config.require_feature_evidence and (
                forecast.feature_bullish_probability
                > 1.0 - forecast.minimum_feature_evidence_probability
            ):
                return "flat"
            return "bear"
        return "flat"
