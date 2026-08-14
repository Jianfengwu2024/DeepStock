"""Public API for the independent BNHMM engine."""

from .config import (
    REGIME_COUNT,
    REGIME_NAMES,
    REGIME_PROTOCOL,
    REGIME_SCORES,
    PROFILE_CHANNELS,
    PROFILE_PROTOCOL,
    BNHMMConfig,
    LabelConfig,
    ProfileConfig,
    RegimeState,
    SequenceConfig,
)
from .data.profile import PriceVolumeProfileBuilder
from .data.dataset import ProfileLabelBuilder, SequenceWindowBuilder
from .data.relative_volume import RelativeVolumeBaseline
from .data.source import BrokerBarSource, DuckDBBarSource
from .types import BarBatch, ProfileState

__all__ = [
    "BNHMMConfig",
    "BarBatch",
    "BrokerBarSource",
    "DuckDBBarSource",
    "LabelConfig",
    "ProfileLabelBuilder",
    "PriceVolumeProfileBuilder",
    "ProfileConfig",
    "ProfileState",
    "PROFILE_CHANNELS",
    "PROFILE_PROTOCOL",
    "REGIME_COUNT",
    "REGIME_NAMES",
    "REGIME_PROTOCOL",
    "REGIME_SCORES",
    "RegimeState",
    "RelativeVolumeBaseline",
    "SequenceConfig",
    "SequenceWindowBuilder",
]

__version__ = "0.5.0"

