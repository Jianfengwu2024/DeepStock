from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bnhmm.boundary_fk.branch_model import (
    BayesianBranchModel,
    BayesianBranchModelConfig,
    BayesianBranchPosterior,
    DRIFT_FEATURE_SETS,
)
from bnhmm.boundary_fk.feynman_kac import (
    BoundaryFeynmanKacConfig,
    BranchFeynmanKacForecaster,
)
from bnhmm.boundary_fk.types import TwoBranchDailyStates, TwoBranchLabels


def _dataset(rows: int = 80) -> tuple[TwoBranchDailyStates, TwoBranchLabels]:
    rng = np.random.default_rng(17)
    sessions = np.busday_offset(np.datetime64("2024-01-02"), np.arange(rows))
    values = np.concatenate(
        (
            rng.normal(0.0, 0.01, size=(rows, 2, 5)),
            np.zeros((rows, 2, 3), dtype=np.float64),
        ),
        axis=2,
    )
    values[:, :, 3] = 0.01
    values[:, :, 4] = 0.8
    values[:, :, 5:] = 0.0
    next_return = np.column_stack(
        (0.8 * values[:, 0, 1], -0.8 * values[:, 1, 1])
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
                "closing_flow",
                "shock_recovery",
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


def test_closing_flow_profile_is_default_and_legacy_profile_stays_four_dimensional() -> None:
    states, labels = _intraday_dataset()
    cutoff = labels.target_time[59]

    posterior = BayesianBranchModel(
        BayesianBranchModelConfig(minimum_rows=30)
    ).fit(states, labels, branch=0, training_cutoff=cutoff)
    legacy = BayesianBranchModel(
        BayesianBranchModelConfig(
            minimum_rows=30,
            drift_feature_set="legacy",
        )
    ).fit(states, labels, branch=0, training_cutoff=cutoff)

    assert posterior.drift_feature_names[-1] == "closing_flow"
    assert len(posterior.drift_feature_names) == 5
    assert legacy.drift_feature_names == (
        "return_1d",
        "return_mean_5",
        "return_mean_20",
        "coherence_1d",
    )


def test_path_efficiency_only_profile_excludes_all_legacy_drift_inputs() -> None:
    states, labels = _intraday_dataset()

    posterior = BayesianBranchModel(
        BayesianBranchModelConfig(
            minimum_rows=30,
            drift_feature_set="path_efficiency_only",
        )
    ).fit(
        states,
        labels,
        branch=0,
        training_cutoff=labels.target_time[59],
    )

    assert DRIFT_FEATURE_SETS["path_efficiency_only"] == (
        "signed_path_efficiency",
    )
    assert posterior.drift_feature_names == ("signed_path_efficiency",)


@pytest.mark.parametrize(
    ("feature_set", "added_feature"),
    (
        ("path_efficiency", "signed_path_efficiency"),
        ("closing_flow", "closing_flow"),
        ("shock_recovery", "shock_recovery"),
    ),
)
def test_single_intraday_feature_profiles_are_independently_fittable(
    feature_set: str,
    added_feature: str,
) -> None:
    states, labels = _intraday_dataset()

    posterior = BayesianBranchModel(
        BayesianBranchModelConfig(
            minimum_rows=30,
            drift_feature_set=feature_set,
        )
    ).fit(
        states,
        labels,
        branch=0,
        training_cutoff=labels.target_time[59],
    )

    assert posterior.drift_feature_names[-1] == added_feature
    assert len(posterior.drift_feature_names) == 5


def test_all_intraday_profile_remains_available_for_ablation() -> None:
    states, labels = _intraday_dataset()

    posterior = BayesianBranchModel(
        BayesianBranchModelConfig(
            minimum_rows=30,
            drift_feature_set="intraday_path",
        )
    ).fit(
        states,
        labels,
        branch=0,
        training_cutoff=labels.target_time[59],
    )

    assert posterior.drift_feature_names[-3:] == (
        "signed_path_efficiency",
        "closing_flow",
        "shock_recovery",
    )


def test_two_branches_fit_separate_drift_posteriors_with_constant_diffusion() -> None:
    states, labels = _dataset()
    cutoff = labels.target_time[59]
    model = BayesianBranchModel(BayesianBranchModelConfig(minimum_rows=30))

    first = model.fit(states, labels, branch=0, training_cutoff=cutoff)
    second = model.fit(states, labels, branch=1, training_cutoff=cutoff)

    assert first.branch == 0 and second.branch == 1
    assert first.drift_mean[2] > 0.0
    assert second.drift_mean[2] < 0.0
    assert np.isscalar(first.diffusion_variance)
    assert first.diffusion_variance > 0.0


def test_future_labels_do_not_change_posterior_before_cutoff() -> None:
    states, labels = _dataset()
    cutoff = labels.target_time[49]
    model = BayesianBranchModel(BayesianBranchModelConfig(minimum_rows=30))
    original = model.fit(states, labels, branch=0, training_cutoff=cutoff)
    changed_returns = labels.next_return.copy()
    changed_returns[50:, 0] = 1000.0

    repeated = model.fit(
        states,
        replace(labels, next_return=changed_returns),
        branch=0,
        training_cutoff=cutoff,
    )

    np.testing.assert_allclose(original.drift_mean, repeated.drift_mean)
    assert original.diffusion_variance == repeated.diffusion_variance


def test_recent_weighted_drift_follows_the_new_market_regime() -> None:
    states, labels = _dataset(rows=140)
    changed_returns = labels.next_return.copy()
    driver = states.feature("return_mean_5")[:, 0]
    changed_returns[:70, 0] = driver[:70]
    changed_returns[70:139, 0] = -driver[70:139]
    changed = replace(labels, next_return=changed_returns)
    cutoff = labels.target_time[138]

    equal_weight = BayesianBranchModel(
        BayesianBranchModelConfig(
            minimum_rows=30,
            drift_half_life_sessions=None,
        )
    ).fit(states, changed, branch=0, training_cutoff=cutoff)
    recent_weight = BayesianBranchModel(
        BayesianBranchModelConfig(
            minimum_rows=30,
            drift_half_life_sessions=20.0,
        )
    ).fit(states, changed, branch=0, training_cutoff=cutoff)

    assert recent_weight.drift_mean[2] < equal_weight.drift_mean[2] - 0.001


def test_effective_potential_is_confidence_not_a_second_drift_shrinkage() -> None:
    states, labels = _dataset()
    base = BayesianBranchPosterior(
        branch=0,
        drift_feature_names=("return_1d", "return_mean_5", "return_mean_20", "coherence_1d"),
        drift_center=np.zeros(4),
        drift_scale=np.ones(4),
        drift_mean=np.asarray([0.02, 0.0, 0.0, 0.0, 0.0]),
        drift_covariance=np.eye(5) * 1e-12,
        diffusion_variance=1e-6,
        potential_feature_names=("abs_return_1d", "volatility_20", "low_coherence_1d"),
        potential_center=np.zeros(3),
        potential_scale=np.ones(3),
        potential_mean=np.asarray([-8.0, 0.0, 0.0, 0.0]),
        potential_covariance=np.eye(4) * 1e-12,
        training_cutoff=np.datetime64("2024-01-01"),
        fit_rows=60,
    )
    forecaster = BranchFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=5000, seed=3)
    )
    low = forecaster.predict(base, states, row=40, neutral_threshold=0.002)
    high = forecaster.predict(
        replace(base, potential_mean=np.asarray([4.0, 0.0, 0.0, 0.0])),
        states,
        row=40,
        neutral_threshold=0.002,
    )

    assert high.trend_survival_probability < low.trend_survival_probability
    assert high.expected_return == pytest.approx(low.expected_return, abs=2e-5)
    assert abs(high.expected_attenuated_drift) < abs(low.expected_attenuated_drift)
    assert high.expected_effective_potential > low.expected_effective_potential


def test_branch_forecast_exposes_posterior_direction_strength() -> None:
    states, labels = _dataset()
    posterior = BayesianBranchModel(
        BayesianBranchModelConfig(minimum_rows=30)
    ).fit(
        states,
        labels,
        branch=0,
        training_cutoff=labels.target_time[59],
    )

    forecast = BranchFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=5000, seed=19)
    ).predict(posterior, states, row=40, neutral_threshold=0.002)

    assert 0.0 <= forecast.bullish_drift_probability <= 1.0
    assert forecast.drift_signal_to_noise >= 0.0
    assert np.isfinite(forecast.expected_attenuated_drift)


def test_potential_confidence_tracks_the_fitted_drift_not_momentum_proxy() -> None:
    states, labels = _dataset(rows=120)
    posterior = BayesianBranchModel(
        BayesianBranchModelConfig(minimum_rows=30)
    ).fit(
        states,
        labels,
        branch=1,
        training_cutoff=labels.target_time[100],
    )

    forecast = BranchFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=5000, seed=23)
    ).predict(posterior, states, row=90, neutral_threshold=0.002)
    legacy = BayesianBranchModel(
        BayesianBranchModelConfig(
            minimum_rows=30,
            drift_half_life_sessions=None,
            potential_target="momentum_continuation",
        )
    ).fit(
        states,
        labels,
        branch=1,
        training_cutoff=labels.target_time[100],
    )
    legacy_forecast = BranchFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=5000, seed=23)
    ).predict(legacy, states, row=90, neutral_threshold=0.002)

    # Branch 2 is deliberately contrarian to return_mean_5 in this fixture.
    # Its accurate drift forecast must therefore receive high survival confidence.
    assert forecast.trend_survival_probability > 0.65
    assert legacy_forecast.trend_survival_probability < 0.5
