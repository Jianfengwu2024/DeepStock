from __future__ import annotations

from dataclasses import replace

import numpy as np

from bnhmm.boundary_fk.branch_model import (
    BayesianBranchModel,
    BayesianBranchModelConfig,
    BayesianBranchPosterior,
)
from bnhmm.boundary_fk.feynman_kac import (
    BoundaryFeynmanKacConfig,
    BranchFeynmanKacForecaster,
)
from bnhmm.boundary_fk.types import TwoBranchDailyStates, TwoBranchLabels


def _dataset(rows: int = 80) -> tuple[TwoBranchDailyStates, TwoBranchLabels]:
    rng = np.random.default_rng(17)
    sessions = np.busday_offset(np.datetime64("2024-01-02"), np.arange(rows))
    values = rng.normal(0.0, 0.01, size=(rows, 2, 5))
    values[:, :, 3] = 0.01
    values[:, :, 4] = 0.8
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


def test_effective_potential_attenuates_drift_toward_noise() -> None:
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

    assert low.raw_state_probabilities[2] > high.raw_state_probabilities[2]
    assert abs(high.expected_return) < abs(low.expected_return)
    assert high.expected_effective_potential > low.expected_effective_potential
