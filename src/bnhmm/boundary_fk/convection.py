"""Causal convection features derived from opposing constituent minute flow."""

from __future__ import annotations

import numpy as np

from .types import TwoBranchDataset


CONVECTION_POTENTIAL_FEATURE_NAMES = ("reverse_flow_1d",)
CONVECTION_DRIFT_FEATURE_NAMES = ("signed_reverse_flow_1d",)


def convection_potential_features(
    data: TwoBranchDataset,
    branch: int,
) -> tuple[tuple[str], np.ndarray, np.ndarray]:
    """Build the current one-day reverse-flow external potential input."""
    if branch < 0 or branch >= data.states.values.shape[1]:
        raise ValueError("convection branch is outside the dataset")
    try:
        one_day_index = data.convection_lookbacks.index(1)
    except ValueError as exc:
        raise ValueError("one-day convection data is required") from exc

    source = data.convection_strength[:, branch, one_day_index]
    source_valid = data.convection_valid[:, branch, one_day_index]
    values = np.where(source_valid, source, np.nan).astype(np.float64)[:, None]
    valid = source_valid.astype(np.bool_, copy=True)
    return CONVECTION_POTENTIAL_FEATURE_NAMES, values, valid


def convection_drift_features(
    data: TwoBranchDataset,
    branch: int,
) -> tuple[tuple[str], np.ndarray, np.ndarray]:
    """Give concentrated opposing flow the direction opposite the daily path."""
    _, concentration, valid = convection_potential_features(data, branch)
    path = data.states.feature("signed_path_efficiency")[:, branch]
    directed = -np.sign(path) * concentration[:, 0]
    directed_valid = valid & data.states.valid[:, branch] & (np.abs(path) > 1e-12)
    values = np.where(directed_valid, directed, np.nan)[:, None]
    return CONVECTION_DRIFT_FEATURE_NAMES, values, directed_valid
