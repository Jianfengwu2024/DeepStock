"""Validated configuration for profile construction and the future model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class RegimeState(IntEnum):
    BEAR = -1
    FLAT = 0
    BULL = 1


REGIME_STATES = tuple(RegimeState)
REGIME_COUNT = len(REGIME_STATES)
REGIME_NAMES = tuple(state.name.lower() for state in REGIME_STATES)
REGIME_SCORES = tuple(float(state.value) for state in REGIME_STATES)
REGIME_PROTOCOL = "continuous-return-posterior-v2"
PROFILE_PROTOCOL = "phase-space-joint-return-v6"
DEFAULT_VELOCITY_BINS = 3
PROFILE_CHANNELS = 4 + DEFAULT_VELOCITY_BINS**2
OUTPUT_HIDDEN_STATES = 9


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    state_timeframes: tuple[str, ...] = ("1d",)
    bin_width_bps: float = 10.0
    max_profile_bins: int = 512
    profile_margin_bins: int = 1
    relative_volume_lookback_sessions: int = 10
    relative_volume_min_observations: int = 10
    relative_volume_clip: float = 10.0
    trading_minutes_per_session: int = 240
    velocity_bins: int = DEFAULT_VELOCITY_BINS
    velocity_clip: float = 4.0
    velocity_smoothing: float = 0.25
    feature_clip: float = 4.0
    minimum_hour_bars: int = 20
    minimum_day_bars: int = 120

    def __post_init__(self) -> None:
        if not self.state_timeframes:
            raise ValueError("at least one state timeframe is required")
        unsupported = set(self.state_timeframes) - {"60m", "1d"}
        if unsupported:
            raise ValueError(f"unsupported state timeframes: {sorted(unsupported)}")
        if self.bin_width_bps <= 0:
            raise ValueError("profile bin width must be positive")
        if self.max_profile_bins < 8:
            raise ValueError("maximum profile bins must be at least 8")
        if self.profile_margin_bins < 0:
            raise ValueError("profile margin bins cannot be negative")
        if self.relative_volume_lookback_sessions < 2:
            raise ValueError("relative-volume lookback must be at least 2")
        if self.relative_volume_min_observations < 1:
            raise ValueError("relative-volume minimum observations must be positive")
        if self.relative_volume_clip <= 1:
            raise ValueError("relative-volume clip must exceed 1")
        if self.relative_volume_min_observations > self.relative_volume_lookback_sessions:
            raise ValueError("relative-volume minimum cannot exceed its lookback")
        if self.trading_minutes_per_session < 1:
            raise ValueError("trading minutes per session must be positive")
        if self.velocity_bins < 3 or self.velocity_bins % 2 == 0:
            raise ValueError("velocity bins must be an odd integer of at least three")
        if self.velocity_clip <= 0 or self.feature_clip <= 0:
            raise ValueError("phase-space clipping limits must be positive")
        if self.velocity_smoothing < 0:
            raise ValueError("velocity smoothing cannot be negative")

    @property
    def profile_channels(self) -> int:
        # Four scalar fields plus the paired price/volume velocity grid.
        return 4 + self.velocity_bins**2


@dataclass(frozen=True, slots=True)
class LabelConfig:
    """Forward-return definition for bearish, flat, and bullish regimes."""

    forward_sessions: int = 1
    bear_threshold: float = -0.005
    bull_threshold: float = 0.005

    def __post_init__(self) -> None:
        if self.forward_sessions < 1:
            raise ValueError("forward_sessions must be positive")
        thresholds = self.thresholds
        if any(left >= right for left, right in zip(thresholds, thresholds[1:])):
            raise ValueError("regime thresholds must be strictly increasing")

    @property
    def thresholds(self) -> tuple[float, float]:
        return (self.bear_threshold, self.bull_threshold)


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
    latent_states: int = 50
    output_states: tuple[RegimeState, ...] = field(default=REGIME_STATES)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    sequences: SequenceConfig = field(default_factory=SequenceConfig)

    def __post_init__(self) -> None:
        if self.latent_states < self.profile.profile_channels:
            raise ValueError("latent states cannot be fewer than profile channels")
        if self.latent_states > 512:
            raise ValueError("latent states cannot exceed 512")
        if tuple(self.output_states) != REGIME_STATES:
            raise ValueError("output_states must contain bear, flat, and bull")

