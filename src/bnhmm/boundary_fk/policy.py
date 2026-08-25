"""Long-only state machine for two independent FK branches."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .feynman_kac import BoundaryFeynmanKacForecast


FULL_BRANCH_EXPOSURE = 0.5
BRANCH_EXPOSURE_STEP = 0.125
ALLOWED_BRANCH_EXPOSURES = (0.0, 0.125, 0.25, 0.375, 0.5)


@dataclass(frozen=True, slots=True)
class BoundaryPolicyConfig:
    minimum_direction_probability: float = 0.60
    minimum_signal_to_noise: float = 0.10
    minimum_survival_probability: float = 0.35

    def __post_init__(self) -> None:
        if not 0.5 <= self.minimum_direction_probability <= 1.0:
            raise ValueError("direction probability must lie in [0.5, 1]")
        if self.minimum_signal_to_noise < 0.0:
            raise ValueError("signal-to-noise gate must be non-negative")
        if not 0.0 <= self.minimum_survival_probability <= 1.0:
            raise ValueError("survival probability must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class BranchPositionState:
    exposure: float = 0.0
    consecutive_flat_signals: int = 0

    def __post_init__(self) -> None:
        if self.exposure not in ALLOWED_BRANCH_EXPOSURES:
            raise ValueError("branch exposure must use 12.5% reduction steps")
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
    branch_directions: tuple[str, str] = ("flat", "flat")
    held_branches: tuple[bool, bool] = (False, False)
    branch_target_exposures: tuple[float, float] = (0.0, 0.0)
    branch_position_states: tuple[BranchPositionState, BranchPositionState] = (
        BranchPositionState(),
        BranchPositionState(),
    )


class BoundaryPolicy:
    def __init__(self, config: BoundaryPolicyConfig | None = None) -> None:
        self.config = config or BoundaryPolicyConfig()

    def decide(
        self,
        forecasts: tuple[BoundaryFeynmanKacForecast, BoundaryFeynmanKacForecast],
        *,
        branches: tuple[tuple[str, ...], tuple[str, ...]],
        neutral_thresholds: tuple[float, float],
        previous_branch_states: tuple[
            BranchPositionState, BranchPositionState
        ] = (BranchPositionState(), BranchPositionState()),
        branch_exposure_caps: tuple[float, float] = (
            FULL_BRANCH_EXPOSURE,
            FULL_BRANCH_EXPOSURE,
        ),
    ) -> BoundaryPolicyDecision:
        if (
            len(forecasts) != 2
            or len(branches) != 2
            or len(neutral_thresholds) != 2
        ):
            raise ValueError("two-branch policy requires exactly two branches")
        if any(cap not in ALLOWED_BRANCH_EXPOSURES for cap in branch_exposure_caps):
            raise ValueError("branch exposure caps must use supported exposure steps")
        directions = tuple(
            self._direction(item, threshold)
            for item, threshold in zip(forecasts, neutral_thresholds, strict=True)
        )
        if "bear" in directions:
            states = (BranchPositionState(), BranchPositionState())
        else:
            states = tuple(
                self._apply_exposure_cap(
                    self._next_position_state(direction, previous), cap
                )
                for direction, previous, cap in zip(
                    directions,
                    previous_branch_states,
                    branch_exposure_caps,
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
            branch_directions=(directions[0], directions[1]),
            held_branches=(exposures[0] > 0.0, exposures[1] > 0.0),
            branch_target_exposures=(exposures[0], exposures[1]),
            branch_position_states=(states[0], states[1]),
        )

    @staticmethod
    def _apply_exposure_cap(
        state: BranchPositionState, cap: float
    ) -> BranchPositionState:
        if state.exposure <= cap:
            return state
        if cap <= 0.0:
            return BranchPositionState()
        return BranchPositionState(
            exposure=cap,
            consecutive_flat_signals=state.consecutive_flat_signals,
        )

    @staticmethod
    def _next_position_state(
        direction: str, previous: BranchPositionState
    ) -> BranchPositionState:
        if direction == "bull":
            return BranchPositionState(exposure=FULL_BRANCH_EXPOSURE)
        if previous.exposure == 0.0:
            return BranchPositionState()
        exposure = round(previous.exposure - BRANCH_EXPOSURE_STEP, 3)
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
            forecast.drift_signal_to_noise < self.config.minimum_signal_to_noise
            or forecast.trend_survival_probability
            < self.config.minimum_survival_probability
        ):
            return "flat"
        if (
            forecast.bullish_drift_probability
            >= self.config.minimum_direction_probability
        ):
            return "bull"
        if (
            forecast.bullish_drift_probability
            <= 1.0 - self.config.minimum_direction_probability
        ):
            return "bear"
        return "flat"
