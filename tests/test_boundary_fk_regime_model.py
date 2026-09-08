from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bnhmm.boundary_fk.branch_model import BayesianBranchModel, BayesianBranchModelConfig
from bnhmm.boundary_fk.feynman_kac import (
    BoundaryFeynmanKacConfig,
    BranchFeynmanKacForecaster,
)
from bnhmm.boundary_fk.hurdle_model import BayesianHurdleModel, BayesianHurdleModelConfig
from bnhmm.boundary_fk.regime_model import (
    BayesianPathRegimeModel,
    BayesianPathRegimeModelConfig,
)
from test_two_branch_fk_model import _dataset, _fit_branch, _potential_inputs


def _regime_labels(*, continuation: bool):
    states, labels = _dataset(120)
    path = states.feature("signed_path_efficiency")[:, 0]
    direction = np.sign(path)
    if not continuation:
        direction *= -1.0
    returns = labels.next_return.copy()
    returns[:, 0] = direction * 0.01
    return states, replace(labels, next_return=returns)


def test_regime_model_recognizes_persistent_continuation() -> None:
    states, labels = _regime_labels(continuation=True)
    posterior = BayesianPathRegimeModel(
        BayesianPathRegimeModelConfig(minimum_rows=30, neutral_sigma=0.35)
    ).fit(states, labels, branch=0, training_cutoff=labels.target_time[100])
    path = states.feature("signed_path_efficiency")[:, 0]
    positive = int(np.flatnonzero(path > 0.0)[-1])
    negative = int(np.flatnonzero(path < 0.0)[-1])

    assert posterior.next_agreement_probability > 0.6
    assert posterior.bullish_probability(states, positive) > 0.6
    assert posterior.bullish_probability(states, negative) < 0.4


def test_regime_model_recognizes_persistent_reversal() -> None:
    states, labels = _regime_labels(continuation=False)
    posterior = BayesianPathRegimeModel(
        BayesianPathRegimeModelConfig(minimum_rows=30, neutral_sigma=0.35)
    ).fit(states, labels, branch=0, training_cutoff=labels.target_time[100])
    path = states.feature("signed_path_efficiency")[:, 0]
    positive = int(np.flatnonzero(path > 0.0)[-1])
    negative = int(np.flatnonzero(path < 0.0)[-1])

    assert posterior.next_agreement_probability < 0.4
    assert posterior.bullish_probability(states, positive) < 0.4
    assert posterior.bullish_probability(states, negative) > 0.6


def test_regime_model_cannot_see_labels_after_cutoff() -> None:
    states, labels = _regime_labels(continuation=True)
    cutoff = labels.target_time[80]
    changed_return = labels.next_return.copy()
    changed_return[82:, 0] *= -1.0
    changed = replace(labels, next_return=changed_return)
    model = BayesianPathRegimeModel(
        BayesianPathRegimeModelConfig(minimum_rows=30, neutral_sigma=0.35)
    )

    original = model.fit(states, labels, branch=0, training_cutoff=cutoff)
    repeated = model.fit(states, changed, branch=0, training_cutoff=cutoff)

    assert original.next_agreement_probability == repeated.next_agreement_probability
    assert original.training_cutoff == repeated.training_cutoff == cutoff


def test_fk_forecast_uses_regime_direction_with_hurdle_significance() -> None:
    states, labels = _regime_labels(continuation=True)
    cutoff = labels.target_time[100]
    branch_posterior = _fit_branch(
        BayesianBranchModel(
            BayesianBranchModelConfig(minimum_rows=30, neutral_sigma=0.35)
        ),
        states,
        labels,
        branch=0,
        training_cutoff=cutoff,
    )
    hurdle = BayesianHurdleModel(
        BayesianHurdleModelConfig(
            minimum_rows=30,
            minimum_direction_rows=20,
            neutral_sigma=0.35,
        )
    ).fit(states, labels, branch=0, training_cutoff=cutoff)
    regime = BayesianPathRegimeModel(
        BayesianPathRegimeModelConfig(minimum_rows=30, neutral_sigma=0.35)
    ).fit(states, labels, branch=0, training_cutoff=cutoff)
    path = states.feature("signed_path_efficiency")[:, 0]
    row = int(np.flatnonzero(path > 0.0)[-1])
    potential, _ = _potential_inputs(states)

    forecast = BranchFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=2000, seed=47)
    ).predict(
        branch_posterior,
        states,
        row,
        neutral_threshold=0.0035,
        potential_features=potential[row],
        hurdle_posterior=hurdle,
        regime_posterior=regime,
    )

    assert forecast.conditional_bullish_probability == pytest.approx(
        regime.bullish_probability(states, row)
    )
    assert forecast.regime_agreement_probability == pytest.approx(
        regime.next_agreement_probability
    )
    assert forecast.regime_agreement_probability_std == pytest.approx(
        regime.next_agreement_probability_std
    )
