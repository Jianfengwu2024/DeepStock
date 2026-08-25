from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bnhmm.boundary_fk.convection import (
    CONVECTION_POTENTIAL_FEATURE_NAMES,
    convection_potential_features,
)
from test_boundary_fk_pipeline import _walk_forward_data


def test_potential_features_use_causal_one_day_persistence_minima() -> None:
    base = _walk_forward_data(6)
    strength = np.zeros((6, 2, 4), dtype=np.float64)
    strength[:, 0, 0] = np.array([0.8, 0.6, 0.9, 0.4, 0.7, 0.5])
    strength[:, 1, 0] = np.array([0.2, 0.3, 0.1, 0.5, 0.4, 0.6])
    valid = np.zeros((6, 2, 4), dtype=np.bool_)
    valid[:, :, 0] = True
    data = replace(base, convection_strength=strength, convection_valid=valid)

    names, values, feature_valid = convection_potential_features(data, branch=0)

    assert names == CONVECTION_POTENTIAL_FEATURE_NAMES
    assert values[2] == pytest.approx([0.9, 0.6, 0.6])
    assert values[4] == pytest.approx([0.7, 0.4, 0.4])
    assert feature_valid.tolist() == [False, False, True, True, True, True]


def test_potential_features_do_not_read_future_convection() -> None:
    base = _walk_forward_data(6)
    strength = np.full((6, 2, 4), 0.4, dtype=np.float64)
    valid = np.zeros((6, 2, 4), dtype=np.bool_)
    valid[:, :, 0] = True
    original = replace(base, convection_strength=strength, convection_valid=valid)
    changed_strength = strength.copy()
    changed_strength[4:, 0, 0] = 0.99
    changed = replace(original, convection_strength=changed_strength)

    _, original_values, original_valid = convection_potential_features(
        original, branch=0
    )
    _, changed_values, changed_valid = convection_potential_features(changed, branch=0)

    np.testing.assert_allclose(
        changed_values[:4], original_values[:4], equal_nan=True
    )
    assert np.array_equal(changed_valid[:4], original_valid[:4])


def test_potential_features_invalidate_windows_with_missing_history() -> None:
    base = _walk_forward_data(6)
    strength = np.full((6, 2, 4), 0.5, dtype=np.float64)
    valid = np.zeros((6, 2, 4), dtype=np.bool_)
    valid[:, :, 0] = True
    valid[2, 0, 0] = False
    data = replace(base, convection_strength=strength, convection_valid=valid)

    _, values, feature_valid = convection_potential_features(data, branch=0)

    assert feature_valid.tolist() == [False, False, False, False, False, True]
    assert np.isnan(values[2:5]).all()
