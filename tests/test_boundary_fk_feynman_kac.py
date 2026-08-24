from __future__ import annotations

import numpy as np
import pytest

from bnhmm.boundary_fk.bayesian_state import BayesianBoundaryStateModel
from bnhmm.boundary_fk.feynman_kac import BoundaryFeynmanKacConfig, BoundaryFeynmanKacForecaster
from test_boundary_fk_bayesian_state import _dataset


def test_raw_probabilities_remain_separate_from_survival_mass() -> None:
    states, labels = _dataset()
    posterior = BayesianBoundaryStateModel().fit(
        states, labels, training_cutoff=labels.target_time[-1]
    )
    forecast = BoundaryFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=2000, seed=7)
    ).predict(posterior, states, len(states.sessions) - 1, labels, neutral_threshold=0.002)

    assert sum(forecast.raw_state_probabilities) == pytest.approx(1.0)
    assert sum(forecast.survival_joint_probabilities) == pytest.approx(
        forecast.trend_survival_probability
    )
    assert forecast.raw_state_probabilities != forecast.survival_joint_probabilities


def test_forecast_is_reproducible_and_splits_return_uncertainty() -> None:
    states, labels = _dataset()
    posterior = BayesianBoundaryStateModel().fit(
        states, labels, training_cutoff=labels.target_time[-1]
    )
    forecaster = BoundaryFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=1500, seed=11)
    )
    first = forecaster.predict(posterior, states, -1, labels, neutral_threshold=0.002)
    second = forecaster.predict(posterior, states, -1, labels, neutral_threshold=0.002)

    assert first == second
    assert first.epistemic_return_variance >= 0.0
    assert first.aleatoric_return_variance > 0.0
    assert first.total_return_variance == pytest.approx(
        first.epistemic_return_variance + first.aleatoric_return_variance
    )
    assert 0.0 <= first.bullish_probability_std <= 0.5


def test_neutral_threshold_partitions_each_sample_once() -> None:
    states, labels = _dataset()
    posterior = BayesianBoundaryStateModel().fit(
        states, labels, training_cutoff=labels.target_time[-1]
    )
    forecast = BoundaryFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=1000)
    ).predict(posterior, states, -1, labels, neutral_threshold=0.5)

    assert forecast.raw_state_probabilities[1] > 0.99


def test_ablation_can_disable_effective_potential_without_changing_raw_distribution() -> None:
    states, labels = _dataset()
    posterior = BayesianBoundaryStateModel().fit(
        states, labels, training_cutoff=labels.target_time[-1]
    )
    with_potential = BoundaryFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=1000, seed=13)
    ).predict(posterior, states, -1, labels, neutral_threshold=0.002)
    without_potential = BoundaryFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=1000, seed=13, use_effective_potential=False)
    ).predict(posterior, states, -1, labels, neutral_threshold=0.002)

    assert without_potential.raw_state_probabilities == with_potential.raw_state_probabilities
    assert without_potential.trend_survival_probability == pytest.approx(1.0)
    assert without_potential.survival_joint_probabilities == pytest.approx(
        without_potential.raw_state_probabilities
    )
