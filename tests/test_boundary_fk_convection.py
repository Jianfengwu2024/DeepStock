from __future__ import annotations

import numpy as np

from bnhmm.boundary_fk.convection import (
    CONVECTION_POTENTIAL_FEATURE_NAMES,
    convection_potential_features,
)
from boundary_fk_fixtures import walk_forward_sector_data


def test_one_day_reverse_flow_is_the_only_potential_feature() -> None:
    base = walk_forward_sector_data(6)

    names, values, feature_valid = convection_potential_features(base, branch=0)

    assert names == CONVECTION_POTENTIAL_FEATURE_NAMES == ("reverse_flow_1d",)
    assert values.shape == (6, 1)
    np.testing.assert_allclose(values[:, 0], base.convection_strength[:, 0, 0])
    np.testing.assert_array_equal(feature_valid, base.convection_valid[:, 0, 0])


def test_one_day_reverse_flow_requires_valid_convection_data() -> None:
    base = walk_forward_sector_data(6)
    base.convection_strength[:] = np.nan
    base.convection_valid[:] = False

    names, values, feature_valid = convection_potential_features(base, branch=0)

    assert names == ("reverse_flow_1d",)
    assert values.shape == (6, 1)
    assert np.isnan(values).all()
    assert not feature_valid.any()
