from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bnhmm.boundary_fk.branch_model import BayesianBranchModel, BayesianBranchModelConfig
from bnhmm.boundary_fk.feynman_kac import (
    BoundaryFeynmanKacConfig,
    BranchFeynmanKacForecaster,
)
from bnhmm.boundary_fk.hurdle_model import (
    BayesianHurdleModel,
    BayesianHurdleModelConfig,
)
from test_two_branch_fk_model import _dataset, _fit_branch, _potential_inputs


def test_fk_defaults_use_only_causal_rolling_path_features() -> None:
    branch = BayesianBranchModelConfig()
    hurdle = BayesianHurdleModelConfig()

    assert branch.drift_feature_names == (
        "rolling_log_position_60d",
        "rolling_path_efficiency_60d",
    )
    assert hurdle.direction_feature_names == branch.drift_feature_names
    assert hurdle.move_feature_names == (
        "rolling_log_position_60d",
        "rolling_path_efficiency_60d",
        "rolling_diffusion_60d",
    )


def test_hurdle_uses_neutral_sigma_for_significant_move_labels() -> None:
    states, labels = _dataset(100)
    returns = labels.next_return.copy()
    returns[:40, 0] = np.where(np.arange(40) % 2 == 0, 0.002, -0.002)
    returns[40:90, 0] = np.where(np.arange(50) % 2 == 0, 0.006, -0.006)
    labels = replace(labels, next_return=returns)
    cutoff = labels.target_time[88]

    posterior = BayesianHurdleModel(
        BayesianHurdleModelConfig(
            minimum_rows=30,
            minimum_direction_rows=20,
            neutral_sigma=0.35,
            half_life_sessions=None,
        )
    ).fit(states, labels, branch=0, training_cutoff=cutoff)

    matured = labels.valid[:, 0] & (labels.target_time <= cutoff)
    significant = matured & (
        np.abs(labels.next_return[:, 0])
        > 0.35 * labels.historical_volatility[:, 0]
    )
    assert posterior.fit_rows == int(matured.sum())
    assert posterior.direction_fit_rows == int(significant.sum())
    assert posterior.direction_label_sigma == pytest.approx(0.35)
    assert 0.5 <= posterior.minimum_feature_evidence_probability <= 0.95
    assert posterior.calibration_rows > 0


def test_hurdle_fit_cannot_see_returns_after_training_cutoff() -> None:
    states, labels = _dataset(100)
    cutoff = labels.target_time[70]
    changed_returns = labels.next_return.copy()
    changed_returns[72:, 0] *= -100.0
    changed = replace(labels, next_return=changed_returns)
    model = BayesianHurdleModel(
        BayesianHurdleModelConfig(
            minimum_rows=30,
            minimum_direction_rows=20,
            neutral_sigma=0.35,
        )
    )

    original = model.fit(states, labels, branch=0, training_cutoff=cutoff)
    repeated = model.fit(states, changed, branch=0, training_cutoff=cutoff)

    np.testing.assert_allclose(original.move_mean, repeated.move_mean)
    np.testing.assert_allclose(original.direction_mean, repeated.direction_mean)
    assert (
        original.minimum_feature_evidence_probability
        == repeated.minimum_feature_evidence_probability
    )
    assert original.training_cutoff == repeated.training_cutoff == cutoff


def test_hurdle_direction_is_conditional_on_significant_moves() -> None:
    states, labels = _dataset(120)
    cutoff = labels.target_time[110]
    posterior = BayesianHurdleModel(
        BayesianHurdleModelConfig(
            minimum_rows=30,
            minimum_direction_rows=20,
            neutral_sigma=0.35,
            half_life_sessions=None,
        )
    ).fit(states, labels, branch=0, training_cutoff=cutoff)
    path = states.feature("rolling_path_efficiency_60d")[:, 0]
    positive_row = int(np.argmax(path[:110]))
    negative_row = int(np.argmin(path[:110]))

    positive = posterior.estimate(states, positive_row)
    negative = posterior.estimate(states, negative_row)

    assert 0.0 < positive.significant_move_probability < 1.0
    assert positive.conditional_bullish_probability > negative.conditional_bullish_probability
    assert positive.feature_log_odds_delta > negative.feature_log_odds_delta


def test_fk_forecast_exposes_hurdle_probabilities_without_changing_potential() -> None:
    states, labels = _dataset(120)
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
    hurdle_posterior = BayesianHurdleModel(
        BayesianHurdleModelConfig(
            minimum_rows=30,
            minimum_direction_rows=20,
            neutral_sigma=0.35,
        )
    ).fit(states, labels, branch=0, training_cutoff=cutoff)
    potential, _ = _potential_inputs(states)

    forecast = BranchFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=4000, seed=41)
    ).predict(
        branch_posterior,
        states,
        row=90,
        potential_features=potential[90],
        neutral_threshold=0.0035,
        hurdle_posterior=hurdle_posterior,
    )

    assert sum(forecast.raw_state_probabilities) == pytest.approx(1.0)
    assert forecast.raw_state_probabilities[1] == pytest.approx(
        1.0 - forecast.significant_move_probability
    )
    assert 0.0 < forecast.significant_move_probability < 1.0
    assert 0.0 < forecast.conditional_bullish_probability < 1.0
    assert forecast.bullish_drift_probability == pytest.approx(
        forecast.conditional_bullish_probability
    )
    assert np.isfinite(forecast.feature_log_odds_delta)
    assert 0.0 <= forecast.feature_bullish_probability <= 1.0
    assert 0.5 <= forecast.minimum_feature_evidence_probability <= 0.95
    assert np.isfinite(forecast.prior_bullish_probability)
    assert forecast.expected_effective_potential > 0.0
