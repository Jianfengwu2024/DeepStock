from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bnhmm.boundary_fk.branch_model import (
    BayesianBranchModel,
    BayesianBranchModelConfig,
    BayesianBranchPosterior,
    BayesianConvectionDriftPosterior,
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
    feature_names = (
        "return_1d",
        "return_mean_5",
        "return_mean_20",
        "volatility_20",
        "coherence_1d",
        "signed_path_efficiency",
        "path_efficiency_delta_1d",
        "path_direction_consistency_3d",
        "signed_member_synchronization",
    )
    values = rng.normal(0.0, 0.01, size=(rows, 2, len(feature_names)))
    values[:, :, 3] = 0.01
    values[:, :, 4] = 0.8
    values[:, :, 5] = rng.uniform(-1.0, 1.0, size=(rows, 2))
    values[1:, :, 6] = np.diff(values[:, :, 5], axis=0)
    values[:, :, 7] = np.sign(values[:, :, 5])
    values[:, :, 8] = np.sign(values[:, :, 5]) * 0.8
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
            feature_names=feature_names,
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
    features = np.linspace(0.1, 0.9, len(states.sessions), dtype=np.float64)[:, None]
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


def test_disruption_potential_has_beta0_and_one_day_reverse_flow() -> None:
    states, labels = _dataset(rows=140)
    potential_features, potential_valid = _potential_inputs(states)
    posterior = BayesianBranchModel(
        BayesianBranchModelConfig(minimum_rows=30)
    ).fit(
        states,
        labels,
        potential_features=potential_features,
        potential_valid=potential_valid,
        branch=0,
        training_cutoff=labels.target_time[120],
    )

    assert posterior.potential_feature_names == CONVECTION_POTENTIAL_FEATURE_NAMES == ("reverse_flow_1d",)
    assert posterior.potential_mean.shape == (2,)
    design = posterior.potential_design(np.asarray([0.5]))
    assert design.shape == (2,)
    assert design[0] == 1.0


def test_c1_only_potential_fixes_beta0_to_zero() -> None:
    states, labels = _dataset(rows=140)
    potential_features, potential_valid = _potential_inputs(states)

    posterior = BayesianBranchModel(
        BayesianBranchModelConfig(minimum_rows=30, potential_mode="c1_only")
    ).fit(
        states,
        labels,
        potential_features=potential_features,
        potential_valid=potential_valid,
        branch=0,
        training_cutoff=labels.target_time[120],
    )

    assert posterior.potential_mode == "c1_only"
    assert posterior.potential_mean[0] == 0.0
    assert posterior.potential_mean[1] >= 0.0
    assert posterior.potential_covariance[0, 0] == 0.0


def test_beta0_only_potential_fixes_c1_coefficient_to_zero() -> None:
    states, labels = _dataset(rows=140)
    potential_features, potential_valid = _potential_inputs(states)

    posterior = BayesianBranchModel(
        BayesianBranchModelConfig(minimum_rows=30, potential_mode="beta0_only")
    ).fit(
        states,
        labels,
        potential_features=potential_features,
        potential_valid=potential_valid,
        branch=0,
        training_cutoff=labels.target_time[120],
    )

    assert posterior.potential_mode == "beta0_only"
    assert posterior.potential_mean[1] == 0.0
    assert posterior.potential_covariance[1, 1] == 0.0


def test_direction_model_uses_the_minimal_causal_feature_set() -> None:
    states, labels = _intraday_dataset()
    cutoff = labels.target_time[59]

    posterior = _fit_branch(
        BayesianBranchModel(BayesianBranchModelConfig(minimum_rows=30)),
        states,
        labels,
        branch=0,
        training_cutoff=cutoff,
    )
    assert DRIFT_FEATURE_NAMES == (
        "signed_path_efficiency",
        "signed_member_synchronization",
    )
    assert posterior.drift_feature_names == DRIFT_FEATURE_NAMES
    assert posterior.drift_model == "bayesian_direction_logit"
    assert posterior.drift_mean[0] == 0.0
    assert np.all(posterior.drift_covariance[0] == 0.0)
    assert np.all(posterior.drift_covariance[:, 0] == 0.0)
    assert np.all(posterior.drift_center == 0.0)
    with pytest.raises(TypeError):
        BayesianBranchModelConfig(drift_feature_set="legacy")


def test_direction_model_accepts_the_best_ordered_feature_subset() -> None:
    features = (
        "signed_path_efficiency",
        "signed_member_synchronization",
    )

    config = BayesianBranchModelConfig(drift_feature_names=features)

    assert config.drift_feature_names == features


def test_direction_model_excludes_ambiguous_training_returns() -> None:
    states, labels = _dataset(rows=100)
    driver = np.sign(states.feature("signed_path_efficiency")[:, 0])
    changed_returns = labels.next_return.copy()
    changed_returns[:79, 0] = -driver[:79] * 0.0001
    changed_returns[::4, 0] = driver[::4] * 0.01
    changed = replace(labels, next_return=changed_returns)
    cutoff = labels.target_time[78]
    features = ("signed_path_efficiency", "signed_member_synchronization")

    unfiltered = _fit_branch(
        BayesianBranchModel(
            BayesianBranchModelConfig(
                minimum_rows=15,
                drift_half_life_sessions=None,
                drift_feature_names=features,
                direction_label_sigma=0.0,
            )
        ),
        states,
        changed,
        branch=0,
        training_cutoff=cutoff,
    )
    filtered = _fit_branch(
        BayesianBranchModel(
            BayesianBranchModelConfig(
                minimum_rows=15,
                drift_half_life_sessions=None,
                drift_feature_names=features,
                direction_label_sigma=0.1,
            )
        ),
        states,
        changed,
        branch=0,
        training_cutoff=cutoff,
    )

    assert unfiltered.drift_mean[1] < 0.0
    assert filtered.drift_mean[1] > 0.0
    assert filtered.direction_label_sigma == 0.1
    assert filtered.direction_fit_rows < filtered.fit_rows


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


def test_one_day_potential_has_no_future_market_feature() -> None:
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
    repeated = model.fit(
        states,
        labels,
        potential_features=features.copy(),
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
    potential_features = np.asarray([0.2])
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


def test_signed_convection_correction_can_flip_bearish_drift_to_bullish() -> None:
    states, _ = _dataset()
    posterior = BayesianBranchPosterior(
        branch=0,
        drift_feature_names=("signed_path_efficiency",),
        drift_center=np.zeros(1),
        drift_scale=np.ones(1),
        drift_mean=np.asarray([-0.01, 0.0]),
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
    correction = BayesianConvectionDriftPosterior(
        feature_names=("signed_reverse_flow_1d",),
        scale=np.ones(1),
        mean=np.asarray([0.03]),
        covariance=np.eye(1) * 1e-12,
        fit_rows=40,
    )

    forecast = BranchFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(
            paths=5000,
            seed=31,
            use_convection_drift_correction=True,
        )
    ).predict(
        posterior,
        states,
        row=40,
        potential_features=np.asarray([0.2]),
        convection_drift_posterior=correction,
        convection_drift_features=np.asarray([1.0]),
        neutral_threshold=0.002,
    )

    assert forecast.expected_base_drift < 0.0
    assert forecast.expected_convection_drift_correction > 0.0
    assert forecast.expected_drift > 0.0
    assert forecast.bullish_drift_probability > 0.99


def test_zero_convection_correction_preserves_common_fk_random_paths() -> None:
    states, labels = _dataset()
    posterior = _fit_branch(
        BayesianBranchModel(BayesianBranchModelConfig(minimum_rows=30)),
        states,
        labels,
        branch=0,
        training_cutoff=labels.target_time[59],
    )
    correction = BayesianConvectionDriftPosterior(
        feature_names=("signed_reverse_flow_1d",),
        scale=np.ones(1),
        mean=np.zeros(1),
        covariance=np.zeros((1, 1)),
        fit_rows=40,
    )
    kwargs = {
        "posterior": posterior,
        "states": states,
        "row": 40,
        "potential_features": np.asarray([0.2]),
        "neutral_threshold": 0.002,
    }

    baseline = BranchFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(paths=5000, seed=37)
    ).predict(**kwargs)
    corrected = BranchFeynmanKacForecaster(
        BoundaryFeynmanKacConfig(
            paths=5000,
            seed=37,
            use_convection_drift_correction=True,
        )
    ).predict(
        **kwargs,
        convection_drift_posterior=correction,
        convection_drift_features=np.asarray([1.0]),
    )

    assert corrected.raw_state_probabilities == baseline.raw_state_probabilities
    assert corrected.expected_effective_potential == baseline.expected_effective_potential


def test_branch_ablation_disables_one_day_potential() -> None:
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
        potential_mean=np.asarray([4.0, 0.0]),
        potential_covariance=np.eye(2) * 1e-12,
        training_cutoff=np.datetime64("2024-01-01"),
        fit_rows=60,
    )
    features = np.asarray([0.2])
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
