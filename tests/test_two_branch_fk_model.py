from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bnhmm.boundary_fk.branch_model import (
    BayesianBranchModel,
    BayesianBranchModelConfig,
    BayesianBranchPosterior,
    DRIFT_FEATURE_NAMES,
)
from bnhmm.boundary_fk.convection import CONVECTION_POTENTIAL_FEATURE_NAMES
from bnhmm.boundary_fk.feynman_kac import (
    BoundaryFeynmanKacConfig,
    BranchFeynmanKacForecaster,
)
from bnhmm.boundary_fk.types import TwoBranchDailyStates, TwoBranchLabels


def _dataset(rows: int = 80) -> tuple[TwoBranchDailyStates, TwoBranchLabels]:
    rng = np.random.default_rng(17)
    sessions = np.busday_offset(np.datetime64("2024-01-02"), np.arange(rows))
    values = rng.normal(0.0, 0.01, size=(rows, 2, 6))
    values[:, :, 3] = 0.01
    values[:, :, 4] = 0.8
    values[:, :, 5] = rng.uniform(-1.0, 1.0, size=(rows, 2))
    next_return = np.column_stack(
        (0.02 * values[:, 0, 5], -0.02 * values[:, 1, 5])
    ) + rng.normal(0.0, 0.001, size=(rows, 2))
    target = np.full(rows, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    target[:-1] = sessions[1:].astype("datetime64[ns]")
    valid = np.ones((rows, 2), dtype=np.bool_)
    valid[-1] = False
    return (
        TwoBranchDailyStates(
            sessions=sessions,
            topology_months=sessions.astype("datetime64[M]"),
            feature_names=(
                "return_1d",
                "return_mean_5",
                "return_mean_20",
                "volatility_20",
                "coherence_1d",
                "signed_path_efficiency",
            ),
            values=values,
            valid=np.ones((rows, 2), dtype=np.bool_),
        ),
        TwoBranchLabels(
            signal_sessions=sessions.copy(),
            target_time=target,
            next_return=next_return,
            historical_volatility=np.full((rows, 2), 0.01),
            valid=valid,
        ),
    )


def _intraday_dataset(
    rows: int = 80,
) -> tuple[TwoBranchDailyStates, TwoBranchLabels]:
    return _dataset(rows)


def _potential_inputs(
    states: TwoBranchDailyStates,
) -> tuple[np.ndarray, np.ndarray]:
    driver = states.feature("signed_path_efficiency")[:, 0]
    features = np.abs(driver)[:, None]
    valid = np.ones(len(features), dtype=np.bool_)
    return features, valid


def _fit_branch(
    model: BayesianBranchModel,
    states: TwoBranchDailyStates,
    labels: TwoBranchLabels,
    *,
    branch: int,
    training_cutoff: np.datetime64,
) -> BayesianBranchPosterior:
    potential_features, potential_valid = _potential_inputs(states)
    return model.fit(
        states,
        labels,
        potential_features=potential_features,
        potential_valid=potential_valid,
        branch=branch,
        training_cutoff=training_cutoff,
    )


def test_convection_disruption_potential_has_nonnegative_slopes() -> None:
    states, labels = _dataset(rows=140)
    potential_features, potential_valid = _potential_inputs(states)
    driver = states.feature("signed_path_efficiency")[:, 0]
    changed_returns = labels.next_return.copy()
    disruption = potential_features[:, 0] > np.nanmedian(potential_features[:, 0])
    changed_returns[:, 0] = np.where(disruption, -np.sign(driver), np.sign(driver)) * 0.01
    changed = replace(labels, next_return=changed_returns)

    posterior = BayesianBranchModel(
        BayesianBranchModelConfig(minimum_rows=30)
    ).fit(
        states,
        changed,
        potential_features=potential_features,
        potential_valid=potential_valid,
        branch=0,
        training_cutoff=labels.target_time[120],
    )

    assert posterior.potential_feature_names == CONVECTION_POTENTIAL_FEATURE_NAMES
    assert np.all(posterior.potential_mean[1:] >= 0.0)
    design = posterior.potential_design(np.array([0.6]))
    assert design.shape == (2,)


def test_path_efficiency_is_the_only_drift_input() -> None:
    states, labels = _intraday_dataset()
    cutoff = labels.target_time[59]

    posterior = _fit_branch(
        BayesianBranchModel(BayesianBranchModelConfig(minimum_rows=30)),
        states,
        labels,
        branch=0,
        training_cutoff=cutoff,
    )
    assert DRIFT_FEATURE_NAMES == ("signed_path_efficiency",)
    assert posterior.drift_feature_names == ("signed_path_efficiency",)
    with pytest.raises(TypeError):
        BayesianBranchModelConfig(drift_feature_set="legacy")


def test_two_branches_fit_separate_drift_posteriors_with_constant_diffusion() -> None:
    states, labels = _dataset()
    cutoff = labels.target_time[59]
    model = BayesianBranchModel(BayesianBranchModelConfig(minimum_rows=30))

    first = _fit_branch(model, states, labels, branch=0, training_cutoff=cutoff)
    second = _fit_branch(model, states, labels, branch=1, training_cutoff=cutoff)

    assert first.branch == 0 and second.branch == 1
    assert first.drift_mean[1] > 0.0
    assert second.drift_mean[1] < 0.0
    assert np.isscalar(first.diffusion_variance)
    assert first.diffusion_variance > 0.0


def test_future_labels_do_not_change_posterior_before_cutoff() -> None:
    states, labels = _dataset()
    cutoff = labels.target_time[49]
    model = BayesianBranchModel(BayesianBranchModelConfig(minimum_rows=30))
    original = _fit_branch(model, states, labels, branch=0, training_cutoff=cutoff)
    changed_returns = labels.next_return.copy()
    changed_returns[50:, 0] = 1000.0

    repeated = _fit_branch(
        model,
        states,
        replace(labels, next_return=changed_returns),
        branch=0,
        training_cutoff=cutoff,
    )

    np.testing.assert_allclose(original.drift_mean, repeated.drift_mean)
    assert original.diffusion_variance == repeated.diffusion_variance


def test_future_convection_does_not_change_potential_before_cutoff() -> None:
    states, labels = _dataset(rows=100)
    features, valid = _potential_inputs(states)
    cutoff = labels.target_time[60]
    model = BayesianBranchModel(BayesianBranchModelConfig(minimum_rows=30))
    original = model.fit(
        states,
        labels,
        potential_features=features,
        potential_valid=valid,
        branch=0,
        training_cutoff=cutoff,
    )
    changed = features.copy()
    changed[61:] = 1_000.0

    repeated = model.fit(
        states,
        labels,
        potential_features=changed,
        potential_valid=valid,
        branch=0,
        training_cutoff=cutoff,
    )

    np.testing.assert_allclose(original.potential_mean, repeated.potential_mean)
    np.testing.assert_allclose(
        original.potential_covariance, repeated.potential_covariance
    )


def test_recent_weighted_drift_follows_the_new_market_regime() -> None:
    states, labels = _dataset(rows=140)
    changed_returns = labels.next_return.copy()
    driver = states.feature("signed_path_efficiency")[:, 0]
    changed_returns[:70, 0] = driver[:70]
    changed_returns[70:139, 0] = -driver[70:139]
    changed = replace(labels, next_return=changed_returns)
    cutoff = labels.target_time[138]

    equal_weight = _fit_branch(
        BayesianBranchModel(BayesianBranchModelConfig(
            minimum_rows=30,
            drift_half_life_sessions=None,
        )),
        states,
        changed,
        branch=0,
        training_cutoff=cutoff,
    )
    recent_weight = _fit_branch(
        BayesianBranchModel(BayesianBranchModelConfig(
            minimum_rows=30,
            drift_half_life_sessions=20.0,
        )),
        states,
        changed,
        branch=0,
        training_cutoff=cutoff,
    )

    assert recent_weight.drift_mean[1] < equal_weight.drift_mean[1] - 0.001


def test_effective_potential_is_confidence_not_a_second_drift_shrinkage() -> None:
    states, labels = _dataset()
    base = BayesianBranchPosterior(
        branch=0,
        drift_feature_names=("signed_path_efficiency",),
        drift_center=np.zeros(1),
        drift_scale=np.ones(1),
        drift_mean=np.asarray([0.02, 0.0]),
        drift_covariance=np.eye(2) * 1e-12,
        diffusion_variance=1e-6,
        potential_feature_names=CONVECTION_POTENTIAL_FEATURE_NAMES,
        potential_center=np.zeros(1),
        potential_scale=np.ones(1),
        potential_mean=np.asarray([-8.0, 0.0]),
        potential_covariance=np.eye(2) * 1e-12,
        training_cutoff=np.datetime64("2024-01-01"),
        fit_rows=60,
    )
    forecaster = BranchFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=5000, seed=3)
    )
    potential_features = np.array([0.4])
    low = forecaster.predict(
        base,
        states,
        row=40,
        potential_features=potential_features,
        neutral_threshold=0.002,
    )
    high = forecaster.predict(
        replace(base, potential_mean=np.asarray([4.0, 0.0])),
        states,
        row=40,
        potential_features=potential_features,
        neutral_threshold=0.002,
    )

    assert high.trend_survival_probability < low.trend_survival_probability
    assert high.expected_return == pytest.approx(low.expected_return, abs=2e-5)
    assert abs(high.expected_attenuated_drift) < abs(low.expected_attenuated_drift)
    assert high.expected_effective_potential > low.expected_effective_potential


def test_branch_ablation_disables_convection_potential() -> None:
    states, _ = _dataset()
    posterior = BayesianBranchPosterior(
        branch=0,
        drift_feature_names=("signed_path_efficiency",),
        drift_center=np.zeros(1),
        drift_scale=np.ones(1),
        drift_mean=np.asarray([0.0015, 0.0]),
        drift_covariance=np.eye(2) * 1e-12,
        diffusion_variance=1e-6,
        potential_feature_names=CONVECTION_POTENTIAL_FEATURE_NAMES,
        potential_center=np.zeros(1),
        potential_scale=np.ones(1),
        potential_mean=np.asarray([4.0, 2.0]),
        potential_covariance=np.eye(2) * 1e-12,
        training_cutoff=np.datetime64("2024-01-01"),
        fit_rows=60,
    )
    features = np.array([0.8])
    enabled = BranchFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=5000, seed=5, use_effective_potential=True)
    ).predict(
        posterior,
        states,
        row=40,
        potential_features=features,
        neutral_threshold=0.002,
    )
    disabled = BranchFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=5000, seed=5, use_effective_potential=False)
    ).predict(
        posterior,
        states,
        row=40,
        potential_features=features,
        neutral_threshold=0.002,
    )

    assert disabled.raw_state_probabilities == enabled.raw_state_probabilities
    assert disabled.trend_survival_probability == pytest.approx(1.0)
    assert disabled.expected_effective_potential == pytest.approx(0.0)
    assert disabled.expected_attenuated_drift == pytest.approx(disabled.expected_drift)
    assert enabled.trend_survival_probability < disabled.trend_survival_probability


def test_branch_forecast_exposes_posterior_direction_strength() -> None:
    states, labels = _dataset()
    posterior = _fit_branch(
        BayesianBranchModel(BayesianBranchModelConfig(minimum_rows=30)),
        states,
        labels,
        branch=0,
        training_cutoff=labels.target_time[59],
    )

    potential_features, _ = _potential_inputs(states)
    forecast = BranchFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=5000, seed=19)
    ).predict(
        posterior,
        states,
        row=40,
        potential_features=potential_features[40],
        neutral_threshold=0.002,
    )

    assert 0.0 <= forecast.bullish_drift_probability <= 1.0
    assert forecast.drift_signal_to_noise >= 0.0
    assert np.isfinite(forecast.expected_attenuated_drift)


def test_potential_confidence_tracks_the_fitted_path_efficiency_drift() -> None:
    states, labels = _dataset(rows=120)
    posterior = _fit_branch(
        BayesianBranchModel(BayesianBranchModelConfig(minimum_rows=30)),
        states,
        labels,
        branch=1,
        training_cutoff=labels.target_time[100],
    )

    potential_features, _ = _potential_inputs(states)
    forecast = BranchFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=5000, seed=23)
    ).predict(
        posterior,
        states,
        row=90,
        potential_features=potential_features[90],
        neutral_threshold=0.002,
    )
    assert 0.0 <= forecast.trend_survival_probability <= 1.0
    with pytest.raises(TypeError):
        BayesianBranchModelConfig(potential_target="momentum_continuation")
