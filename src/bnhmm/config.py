"""Validated configuration for profile construction and the future model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class RegimeState(IntEnum):
    STRONG_BEAR = -2
    WEAK_BEAR = -1
    WEAK_BULL = 1
    STRONG_BULL = 2


REGIME_STATES = tuple(RegimeState)
REGIME_COUNT = len(REGIME_STATES)
REGIME_NAMES = tuple(state.name.lower() for state in REGIME_STATES)
REGIME_SCORES = tuple(float(state.value) for state in REGIME_STATES)
REGIME_PROTOCOL = "volatility-standardized-prior-evidence-v4"
PROFILE_PROTOCOL = "phase-space-joint-return-v6"
SEQUENCE_PROTOCOL = "fixed-recent-endpoint-v1"
DEFAULT_VELOCITY_BINS = 3
PROFILE_CHANNELS = 4 + DEFAULT_VELOCITY_BINS**2


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
    """Leakage-safe standardized forward-return definition."""

    forward_sessions: int = 1
    standardization_lookback_sessions: int = 20
    standardization_min_observations: int = 10
    minimum_return_scale: float = 0.002
    strong_threshold_sigma: float = 0.75

    def __post_init__(self) -> None:
        if self.forward_sessions < 1:
            raise ValueError("forward_sessions must be positive")
        if self.standardization_lookback_sessions < 2:
            raise ValueError("return standardization lookback must be at least two")
        if not 2 <= self.standardization_min_observations <= self.standardization_lookback_sessions:
            raise ValueError("return standardization observations must fit the lookback")
        if self.minimum_return_scale <= 0:
            raise ValueError("minimum return scale must be positive")
        if self.strong_threshold_sigma <= 0:
            raise ValueError("strong-state sigma threshold must be positive")

    @property
    def thresholds(self) -> tuple[float, float, float]:
        return (-self.strong_threshold_sigma, 0.0, self.strong_threshold_sigma)

    @property
    def strong_bear_threshold(self) -> float:
        return -self.strong_threshold_sigma

    @property
    def direction_threshold(self) -> float:
        return 0.0

    @property
    def strong_bull_threshold(self) -> float:
        return self.strong_threshold_sigma


@dataclass(frozen=True, slots=True)
class SequenceConfig:
    length: int = 5
    stride: int = 1
    minimum_length: int = 2

    def __post_init__(self) -> None:
        if self.length < 2:
            raise ValueError("sequence length must be at least two")
        if self.length > 20:
            raise ValueError("sequence length cannot exceed twenty trading states")
        if self.stride < 1:
            raise ValueError("sequence stride must be positive")
        if not 2 <= self.minimum_length <= self.length:
            raise ValueError("minimum_length must be between two and length")


@dataclass(frozen=True, slots=True)
class BNHMMConfig:
    latent_states: int = 20
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
            raise ValueError(
                "output_states must contain strong bear, weak bear, weak bull, and strong bull"
            )

