"""Causal convection features derived from opposing constituent minute flow."""

from __future__ import annotations

import numpy as np

from .types import TwoBranchDataset


CONVECTION_POTENTIAL_FEATURE_NAMES = ("reverse_flow_1d",)


def convection_potential_features(
    data: TwoBranchDataset,
    branch: int,
) -> tuple[tuple[str], np.ndarray, np.ndarray]:
    """Build the current one-day reverse-flow external potential input."""
    if branch not in (0, 1):
        raise ValueError("convection branch must be zero or one")
    try:
        one_day_index = data.convection_lookbacks.index(1)
    except ValueError as exc:
        raise ValueError("one-day convection data is required") from exc

    source = data.convection_strength[:, branch, one_day_index]
    source_valid = data.convection_valid[:, branch, one_day_index]
    values = np.where(source_valid, source, np.nan).astype(np.float64)[:, None]
    valid = source_valid.astype(np.bool_, copy=True)
    return CONVECTION_POTENTIAL_FEATURE_NAMES, values, valid
