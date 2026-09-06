from __future__ import annotations

import numpy as np
import pytest

from bnhmm.percolation_branching.filter import CriticalStateFilter
from bnhmm.percolation_branching.simulator import PercolationSimulator
from bnhmm.percolation_branching.types import (
    BranchingPosterior,
    CriticalStateForecast,
    DailyBranchingObservation,
)


def _posterior(direction: str, symbols: int = 5) -> BranchingPosterior:
    adjacency = np.ones((symbols, symbols), dtype=np.bool_)
    np.fill_diagonal(adjacency, False)
    normalized = adjacency.astype(np.float64) / (symbols - 1)
    return BranchingPosterior(
        direction=direction,
        symbols=tuple(f"S{i}" for i in range(symbols)),
        adjacency=adjacency,
        mean=normalized.copy(),
        variance=np.full((symbols, symbols), 1e-4) * adjacency,
        normalized_mean=normalized,
        lag_weights=np.asarray([0.5, 0.3, 0.2]),
        external_rate=np.full(symbols, 0.01),
        self_rate=np.full(symbols, 0.02),
        training_cutoff=np.datetime64("2025-01-30"),
    )


def _forecast(reproduction: float) -> CriticalStateForecast:
    return CriticalStateForecast(
        log_reproduction_mean=float(np.log(reproduction)),
        log_reproduction_variance=0.005,
        slope_mean=0.0,
        slope_variance=0.001,
        probability_supercritical=float(reproduction > 1.0),
        external_mean=0.04,
        external_variance=0.001,
    )


def _isolated_posterior(direction: str, symbols: int = 5) -> BranchingPosterior:
    empty = np.zeros((symbols, symbols), dtype=np.float64)
    return BranchingPosterior(
        direction=direction,
        symbols=tuple(f"S{i}" for i in range(symbols)),
        adjacency=np.zeros((symbols, symbols), dtype=np.bool_),
        mean=empty.copy(),
        variance=empty.copy(),
        normalized_mean=empty.copy(),
        lag_weights=np.asarray([0.5, 0.3, 0.2]),
        external_rate=np.full(symbols, 0.2),
        self_rate=np.zeros(symbols),
        training_cutoff=np.datetime64("2025-01-30"),
    )


def test_supercritical_observations_raise_crossing_probability() -> None:
    model = CriticalStateFilter()
    state = model.initialize([])
    for day in range(8):
        state = model.update(
            state,
            DailyBranchingObservation(
                parent_exposure=20,
                offspring=32,
                external=2,
                session=np.datetime64("2025-01-01") + np.timedelta64(day, "D"),
            ),
        )

    forecast = model.predict(state)

    assert forecast.probability_supercritical > 0.8


def test_external_rain_does_not_raise_reproduction_number() -> None:
    model = CriticalStateFilter()
    initial = model.initialize([])
    quiet = model.update(
        initial, DailyBranchingObservation(20, 10, 2)
    )
    rainy = model.update(
        initial, DailyBranchingObservation(20, 10, 50)
    )

    assert rainy.mean[0] == pytest.approx(quiet.mean[0])
    assert rainy.external_shape > quiet.external_shape


def test_supercritical_simulation_grows_larger_clusters_deterministically() -> None:
    bull = _posterior("bull")
    bear = _posterior("bear")
    simulator = PercolationSimulator(paths=800, maximum_events=500)
    kwargs = dict(
        bull_posterior=bull,
        bear_posterior=bear,
        bear_forecast=_forecast(0.4),
        bull_threshold=0.6,
        bear_threshold=0.6,
        signal_time=np.datetime64("2025-01-30T15:01"),
        available_at=np.datetime64("2025-01-31T09:25"),
        target_session=np.datetime64("2025-01-31"),
    )

    subcritical = simulator.predict(
        **kwargs,
        bull_forecast=_forecast(0.4),
        rng=np.random.default_rng(17),
    )
    supercritical = simulator.predict(
        **kwargs,
        bull_forecast=_forecast(1.8),
        rng=np.random.default_rng(17),
    )
    repeated = simulator.predict(
        **kwargs,
        bull_forecast=_forecast(1.8),
        rng=np.random.default_rng(17),
    )

    assert supercritical.bull_giant_fraction_mean > subcritical.bull_giant_fraction_mean
    assert supercritical.phase_probabilities == repeated.phase_probabilities
    np.testing.assert_allclose(
        supercritical.bull_membership_probability,
        repeated.bull_membership_probability,
    )
    assert sum(supercritical.phase_probabilities) == pytest.approx(1.0)


def test_independent_external_seeds_do_not_merge_into_a_giant_cluster() -> None:
    rainy = CriticalStateForecast(
        log_reproduction_mean=float(np.log(1.5)),
        log_reproduction_variance=0.0,
        slope_mean=0.0,
        slope_variance=0.0,
        probability_supercritical=1.0,
        external_mean=100.0,
        external_variance=1.0,
    )
    prediction = PercolationSimulator(paths=100).predict(
        _isolated_posterior("bull"),
        _isolated_posterior("bear"),
        rainy,
        rainy,
        bull_threshold=0.5,
        bear_threshold=0.5,
        signal_time=np.datetime64("2025-01-30T15:01"),
        available_at=np.datetime64("2025-01-31T09:25"),
        target_session=np.datetime64("2025-01-31"),
        rng=np.random.default_rng(19),
    )

    assert prediction.bull_giant_fraction_mean == pytest.approx(0.2)
    assert prediction.bear_giant_fraction_mean == pytest.approx(0.2)
    assert prediction.phase_probabilities == pytest.approx((0.0, 1.0, 0.0, 0.0))


def test_prediction_rejects_noncausal_timing() -> None:
    simulator = PercolationSimulator(paths=20)
    with pytest.raises(ValueError, match="causal"):
        simulator.predict(
            _posterior("bull"),
            _posterior("bear"),
            _forecast(1.1),
            _forecast(0.8),
            bull_threshold=0.6,
            bear_threshold=0.6,
            signal_time=np.datetime64("2025-02-01T10:00"),
            available_at=np.datetime64("2025-01-31T09:25"),
            target_session=np.datetime64("2025-01-31"),
            rng=np.random.default_rng(1),
        )
