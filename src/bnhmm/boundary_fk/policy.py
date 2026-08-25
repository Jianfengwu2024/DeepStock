"""Long-only state machine for two independent FK branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .feynman_kac import BoundaryFeynmanKacForecast


@dataclass(frozen=True, slots=True)
class BoundaryPolicyConfig:
    mode: Literal["posterior_signal", "legacy_attenuated"] = "posterior_signal"
    minimum_direction_probability: float = 0.60
    minimum_signal_to_noise: float = 0.10
    minimum_survival_probability: float = 0.35

    def __post_init__(self) -> None:
        if self.mode not in ("posterior_signal", "legacy_attenuated"):
            raise ValueError("unknown two-branch policy mode")
        if not 0.5 <= self.minimum_direction_probability <= 1.0:
            raise ValueError("direction probability must lie in [0.5, 1]")
        if self.minimum_signal_to_noise < 0.0:
            raise ValueError("signal-to-noise gate must be non-negative")
        if not 0.0 <= self.minimum_survival_probability <= 1.0:
            raise ValueError("survival probability must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class BoundaryPolicyDecision:
    open_position: bool
    gate_reason: str
    survival_conditional_bullish_probability: float
    selected_symbols: tuple[str, ...] = ()
    target_weights: tuple[float, ...] = ()
    branch_directions: tuple[str, str] = ("flat", "flat")
    held_branches: tuple[bool, bool] = (False, False)


class BoundaryPolicy:
    def __init__(self, config: BoundaryPolicyConfig | None = None) -> None:
        self.config = config or BoundaryPolicyConfig()

    def decide(
        self,
        forecasts: tuple[BoundaryFeynmanKacForecast, BoundaryFeynmanKacForecast],
        *,
        branches: tuple[tuple[str, ...], tuple[str, ...]],
        held_branches: tuple[bool, bool],
        neutral_thresholds: tuple[float, float],
    ) -> BoundaryPolicyDecision:
        if (
            len(forecasts) != 2
            or len(branches) != 2
            or len(held_branches) != 2
            or len(neutral_thresholds) != 2
        ):
            raise ValueError("two-branch policy requires exactly two branches")
        directions = tuple(
            self._direction(item, threshold)
            for item, threshold in zip(forecasts, neutral_thresholds, strict=True)
        )
        next_held: list[bool] = []
        for direction, held in zip(directions, held_branches, strict=True):
            if direction == "bull":
                next_held.append(True)
            elif direction == "bear":
                next_held.append(False)
            else:
                next_held.append(held)
        resolved_held = (next_held[0], next_held[1])
        selected = tuple(
            symbol
            for branch, symbols in enumerate(branches)
            if resolved_held[branch]
            for symbol in symbols
        )
        if len(selected) != len(set(selected)):
            raise ValueError("a held symbol cannot appear in both branches")
        weight = 1.0 / len(selected) if selected else 0.0
        bullish = float(
            np.mean([item.raw_state_probabilities[2] for item in forecasts])
        )
        return BoundaryPolicyDecision(
            open_position=bool(selected),
            gate_reason="/".join(directions),
            survival_conditional_bullish_probability=bullish,
            selected_symbols=selected,
            target_weights=(weight,) * len(selected),
            branch_directions=(directions[0], directions[1]),
            held_branches=resolved_held,
        )

    def _direction(
        self,
        forecast: BoundaryFeynmanKacForecast,
        neutral_threshold: float,
    ) -> str:
        if not np.isfinite(neutral_threshold) or neutral_threshold < 0.0:
            raise ValueError("neutral threshold must be finite and non-negative")
        if self.config.mode == "legacy_attenuated":
            if forecast.expected_attenuated_drift > neutral_threshold:
                return "bull"
            if forecast.expected_attenuated_drift < -neutral_threshold:
                return "bear"
            return "flat"
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
