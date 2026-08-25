from __future__ import annotations

import numpy as np

from bnhmm.boundary_fk.convection import (
    BACKGROUND_POTENTIAL_FEATURE_NAMES,
    background_potential_features,
)
from test_boundary_fk_pipeline import _walk_forward_data


def test_background_potential_has_no_market_features() -> None:
    base = _walk_forward_data(6)

    names, values, feature_valid = background_potential_features(base, branch=0)

    assert names == BACKGROUND_POTENTIAL_FEATURE_NAMES == ()
    assert values.shape == (6, 0)
    assert feature_valid.tolist() == [True, True, True, True, True, True]


def test_background_potential_does_not_depend_on_convection_data() -> None:
    base = _walk_forward_data(6)
    base.convection_strength[:] = np.nan
    base.convection_valid[:] = False

    names, values, feature_valid = background_potential_features(base, branch=1)

    assert names == ()
    assert values.shape == (6, 0)
    assert feature_valid.all()
