"""Validated configuration for profile construction and the future model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class RegimeState(IntEnum):
    STRONG_BEAR = -2
    WEAK_BEAR = -1
    NEUTRAL = 0
    WEAK_BULL = 1
    STRONG_BULL = 2


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    bins: int = 48
    price_span: float = 4.0
    default_log_scale: float = 0.02
    minimum_log_scale: float = 0.0025
    scale_lookback_sessions: int = 20
    relative_volume_lookback_sessions: int = 20
    relative_volume_min_observations: int = 5
    relative_volume_clip: float = 10.0
    minimum_hour_bars: int = 20
    minimum_day_bars: int = 120

    def __post_init__(self) -> None:
        if self.bins < 8:
            raise ValueError("profile bins must be at least 8")
        if self.price_span <= 0:
            raise ValueError("price_span must be positive")
        if self.minimum_log_scale <= 0:
            raise ValueError("minimum_log_scale must be positive")
        if self.default_log_scale < self.minimum_log_scale:
            raise ValueError("default_log_scale cannot be below minimum_log_scale")
        if self.scale_lookback_sessions < 2:
            raise ValueError("scale_lookback_sessions must be at least 2")
        if self.relative_volume_lookback_sessions < 2:
            raise ValueError("relative-volume lookback must be at least 2")
        if self.relative_volume_min_observations < 1:
            raise ValueError("relative-volume minimum observations must be positive")
        if self.relative_volume_clip <= 1:
            raise ValueError("relative-volume clip must exceed 1")


@dataclass(frozen=True, slots=True)
class BNHMMConfig:
    latent_states: int = 10
    output_states: tuple[RegimeState, ...] = field(
        default=(
            RegimeState.STRONG_BEAR,
            RegimeState.WEAK_BEAR,
            RegimeState.NEUTRAL,
            RegimeState.WEAK_BULL,
            RegimeState.STRONG_BULL,
        )
    )
    profile: ProfileConfig = field(default_factory=ProfileConfig)

    def __post_init__(self) -> None:
        if self.latent_states != 10:
            raise ValueError("BNHMM v2 requires exactly ten unnamed latent states")
        if tuple(self.output_states) != tuple(RegimeState):
            raise ValueError("output_states must contain the canonical five regimes")

