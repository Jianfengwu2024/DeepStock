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
    state_timeframes: tuple[str, ...] = ("1d",)
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
        if not self.state_timeframes:
            raise ValueError("at least one state timeframe is required")
        unsupported = set(self.state_timeframes) - {"60m", "1d"}
        if unsupported:
            raise ValueError(f"unsupported state timeframes: {sorted(unsupported)}")
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
class LabelConfig:
    """Forward-return definition for the canonical five regimes."""

    forward_sessions: int = 1
    strong_bear_threshold: float = -0.02
    weak_bear_threshold: float = -0.005
    weak_bull_threshold: float = 0.005
    strong_bull_threshold: float = 0.02

    def __post_init__(self) -> None:
        if self.forward_sessions < 1:
            raise ValueError("forward_sessions must be positive")
        thresholds = self.thresholds
        if any(left >= right for left, right in zip(thresholds, thresholds[1:])):
            raise ValueError("regime thresholds must be strictly increasing")

    @property
    def thresholds(self) -> tuple[float, float, float, float]:
        return (
            self.strong_bear_threshold,
            self.weak_bear_threshold,
            self.weak_bull_threshold,
            self.strong_bull_threshold,
        )


@dataclass(frozen=True, slots=True)
class SequenceConfig:
    length: int = 128
    stride: int = 64
    minimum_length: int = 32

    def __post_init__(self) -> None:
        if self.length < 2:
            raise ValueError("sequence length must be at least two")
        if self.stride < 1:
            raise ValueError("sequence stride must be positive")
        if not 2 <= self.minimum_length <= self.length:
            raise ValueError("minimum_length must be between two and length")


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
    labels: LabelConfig = field(default_factory=LabelConfig)
    sequences: SequenceConfig = field(default_factory=SequenceConfig)

    def __post_init__(self) -> None:
        if self.latent_states != 10:
            raise ValueError("BNHMM v2 requires exactly ten unnamed latent states")
        if tuple(self.output_states) != tuple(RegimeState):
            raise ValueError("output_states must contain the canonical five regimes")

