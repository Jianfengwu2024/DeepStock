from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bnhmm.boundary_fk.bayesian_state import (
    BayesianBoundaryStateModel,
    BoundaryStateLabels,
)
from bnhmm.boundary_fk.types import BoundaryDailyStates


FEATURES = (
    "velocity_close",
    "velocity_last30",
    "velocity_last60",
    "velocity_integral",
    "velocity_quadratic_variation",
    "momentum_close",
    "momentum_last60",
    "momentum_integral",
    "water_close",
    "water_last60",
    "water_integral",
    "coherence_close",
    "coherence_last60",
    "coherence_high_occupancy",
    "turbulence_close",
    "turbulence_last60",
    "turbulence_high_occupancy",
    "decoupling_close",
    "decoupling_last60",
    "decoupling_integral",
    "thickness_close",
    "thickness_last60",
    "thickness_slope_close",
    "viscosity_close",
)


def _dataset(rows: int = 80) -> tuple[BoundaryDailyStates, BoundaryStateLabels]:
    rng = np.random.default_rng(19)
    sessions = np.busday_offset(np.datetime64("2023-01-02"), np.arange(rows))
    values = rng.normal(size=(rows, len(FEATURES)))
    index = {name: column for column, name in enumerate(FEATURES)}
    values[:, index["coherence_last60"]] = rng.uniform(0.1, 0.95, rows)
    values[:, index["turbulence_last60"]] = rng.lognormal(-0.2, 0.4, rows)
    values[:, index["thickness_last60"]] = rng.lognormal(-0.4, 0.3, rows)
    values[:, index["viscosity_close"]] = rng.uniform(0.2, 1.8, rows)
    target = (
        0.004 * values[:, index["velocity_last60"]]
        + 0.003 * values[:, index["momentum_last60"]]
        - 0.002 * values[:, index["decoupling_last60"]]
        + rng.normal(0.0, 0.008, rows)
    )
    survival = (
        values[:, index["coherence_last60"]]
        > 0.45 + 0.1 * values[:, index["thickness_last60"]]
    ).astype(np.float64)
    states = BoundaryDailyStates(
        sessions=sessions,
        topology_months=sessions.astype("datetime64[M]"),
        feature_names=FEATURES,
        units=tuple("unit" for _ in FEATURES),
        values=values,
        valid=np.ones(rows, dtype=np.bool_),
        branch_degraded=np.ones(rows, dtype=np.bool_),
    )
    labels = BoundaryStateLabels(
        signal_sessions=sessions,
        target_time=(sessions + np.timedelta64(1, "D")).astype("datetime64[ns]"),
        next_return=target,
        historical_volatility=np.full(rows, 0.01),
        survival=survival,
        return_valid=np.ones(rows, dtype=np.bool_),
        survival_valid=np.ones(rows, dtype=np.bool_),
    )
    return states, labels


def test_fit_rejects_unmatured_targets() -> None:
    states, labels = _dataset()
    cutoff = labels.target_time[-2]

    with pytest.raises(ValueError, match="target_time"):
        BayesianBoundaryStateModel().fit(states, labels, training_cutoff=cutoff)


def test_posterior_covariances_are_positive_semidefinite() -> None:
    states, labels = _dataset()
    posterior = BayesianBoundaryStateModel().fit(
        states,
        labels,
        training_cutoff=labels.target_time[-1],
    )

    for covariance in (
        posterior.drift_covariance,
        posterior.diffusion_covariance,
        posterior.potential_covariance,
    ):
        assert np.linalg.eigvalsh(covariance).min() >= -1e-10
    assert posterior.drift_noise_variance > 0.0
    assert posterior.diffusion_noise_variance > 0.0


def test_model_uses_only_named_low_dimensional_features() -> None:
    states, labels = _dataset()
    posterior = BayesianBoundaryStateModel().fit(
        states,
        labels,
        training_cutoff=labels.target_time[-1],
    )

    assert set(posterior.drift_feature_names).issubset(set(FEATURES))
    named_or_derived = set(FEATURES) | {
        "abs_decoupling_last60",
        "low_coherence_last60",
    }
    assert set(posterior.potential_feature_names).issubset(named_or_derived)
    assert "historical_index_volatility" in posterior.diffusion_feature_names
    assert len(posterior.drift_feature_names) <= 10
    assert len(posterior.diffusion_feature_names) <= 6
    assert len(posterior.potential_feature_names) <= 6


def test_invalid_future_rows_cannot_change_training_posterior() -> None:
    states, labels = _dataset()
    split = 60
    valid = np.arange(len(states.sessions)) < split
    bounded_states = replace(states, valid=valid)
    bounded_labels = replace(
        labels,
        return_valid=valid,
        survival_valid=valid,
    )
    cutoff = labels.target_time[split - 1]
    base = BayesianBoundaryStateModel().fit(
        bounded_states,
        bounded_labels,
        training_cutoff=cutoff,
    )
    changed_values = states.values.copy()
    changed_values[split:] = 1_000_000.0
    changed_states = replace(bounded_states, values=changed_values)

    repeated = BayesianBoundaryStateModel().fit(
        changed_states,
        bounded_labels,
        training_cutoff=cutoff,
    )

    np.testing.assert_allclose(base.drift_mean, repeated.drift_mean)
    np.testing.assert_allclose(base.diffusion_mean, repeated.diffusion_mean)
    np.testing.assert_allclose(base.potential_mean, repeated.potential_mean)


def test_posterior_design_vectors_are_finite_and_include_intercept() -> None:
    states, labels = _dataset()
    posterior = BayesianBoundaryStateModel().fit(
        states,
        labels,
        training_cutoff=labels.target_time[-1],
    )

    for component in ("drift", "diffusion", "potential"):
        design = posterior.design_vector(component, states, -1, labels)
        assert design[0] == pytest.approx(1.0)
        assert np.isfinite(design).all()
