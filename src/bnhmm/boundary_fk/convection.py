"""Causal convection features derived from opposing constituent minute flow."""

from __future__ import annotations

import numpy as np

from .types import TwoBranchDataset


CONVECTION_POTENTIAL_FEATURE_NAMES = (
    "reverse_flow_1d",
    "reverse_flow_persistence_2d",
    "reverse_flow_persistence_3d",
)


def convection_potential_features(
    data: TwoBranchDataset,
    branch: int,
) -> tuple[tuple[str, str, str], np.ndarray, np.ndarray]:
    """Build causal convection level and persistence features for one branch."""
    if branch not in (0, 1):
        raise ValueError("convection branch must be zero or one")
    try:
        one_day_index = data.convection_lookbacks.index(1)
    except ValueError as exc:
        raise ValueError("one-day convection data is required") from exc

    source = data.convection_strength[:, branch, one_day_index]
    source_valid = data.convection_valid[:, branch, one_day_index]
    values = np.full((len(source), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(source), dtype=np.bool_)
    for row in range(2, len(source)):
        if not np.all(source_valid[row - 2 : row + 1]):
            continue
        current = float(source[row])
        values[row] = (
            current,
            float(np.min(source[row - 1 : row + 1])),
            float(np.min(source[row - 2 : row + 1])),
        )
        valid[row] = True
    return CONVECTION_POTENTIAL_FEATURE_NAMES, values, valid
