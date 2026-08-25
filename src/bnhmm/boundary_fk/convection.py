"""Feature-free background potential inputs for strict FK ablation."""

from __future__ import annotations

import numpy as np

from .types import TwoBranchDataset


BACKGROUND_POTENTIAL_FEATURE_NAMES: tuple[str, ...] = ()


def background_potential_features(
    data: TwoBranchDataset,
    branch: int,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Build an intercept-only potential input without reading market features."""
    if branch not in (0, 1):
        raise ValueError("background potential branch must be zero or one")
    rows = len(data.states.sessions)
    return (
        BACKGROUND_POTENTIAL_FEATURE_NAMES,
        np.empty((rows, 0), dtype=np.float64),
        np.ones(rows, dtype=np.bool_),
    )
