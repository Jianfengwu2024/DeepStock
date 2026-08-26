from __future__ import annotations

import numpy as np

from bnhmm.boundary_fk.convection import (
    CONVECTION_POTENTIAL_FEATURE_NAMES,
    convection_drift_features,
    convection_potential_features,
)
from test_boundary_fk_pipeline import _walk_forward_data


def test_one_day_reverse_flow_is_the_only_potential_feature() -> None:
    base = _walk_forward_data(6)

    names, values, feature_valid = convection_potential_features(base, branch=0)

    assert names == CONVECTION_POTENTIAL_FEATURE_NAMES == ("reverse_flow_1d",)
    assert values.shape == (6, 1)
    np.testing.assert_allclose(values[:, 0], base.convection_strength[:, 0, 0])
    np.testing.assert_array_equal(feature_valid, base.convection_valid[:, 0, 0])


def test_one_day_reverse_flow_requires_valid_convection_data() -> None:
    base = _walk_forward_data(6)
    base.convection_strength[:] = np.nan
    base.convection_valid[:] = False

    names, values, feature_valid = convection_potential_features(base, branch=1)

    assert names == ("reverse_flow_1d",)
    assert values.shape == (6, 1)
    assert np.isnan(values).all()
    assert not feature_valid.any()


def test_signed_convection_points_against_the_current_path_direction() -> None:
    data = _walk_forward_data(6)
    column = data.states.feature_names.index("signed_path_efficiency")
    data.states.values[:, 0, column] = np.asarray(
        [0.4, -0.2, 0.4, -0.2, 0.4, -0.2]
    )
    data.convection_strength[:, 0, 0] = 0.25
    data.convection_valid[:, 0, 0] = True

    names, values, valid = convection_drift_features(data, 0)

    assert names == ("signed_reverse_flow_1d",)
    np.testing.assert_allclose(values[:4, 0], [-0.25, 0.25, -0.25, 0.25])
    assert valid[:4].all()
