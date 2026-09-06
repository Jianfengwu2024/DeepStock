from __future__ import annotations

import numpy as np
import pytest

from bnhmm.percolation_branching.labels import (
    NullThresholdCalibrator,
    classify_phase,
    temporal_reachability,
)


def _full_support(symbols: int) -> np.ndarray:
    support = np.ones((symbols, symbols), dtype=np.bool_)
    np.fill_diagonal(support, False)
    return support


def test_temporal_chain_reaches_three_distinct_stocks() -> None:
    events = np.zeros((8, 3), dtype=np.bool_)
    events[0, 0] = True
    events[1, 1] = True
    events[3, 2] = True

    result = temporal_reachability(events, _full_support(3), maximum_lag=3)

    assert result.giant_fraction == 1.0
    assert result.source_membership[0].tolist() == [True, True, True]
    assert result.edge_count == 2


def test_same_minute_common_jump_does_not_form_a_chain() -> None:
    events = np.zeros((8, 3), dtype=np.bool_)
    events[5, :] = True

    result = temporal_reachability(events, _full_support(3), maximum_lag=3)

    assert result.giant_fraction == pytest.approx(1 / 3)
    assert result.edge_count == 0


def test_phase_labels_cover_all_four_states() -> None:
    assert classify_phase(0.8, 0.2, 0.6, 0.6) == "bull"
    assert classify_phase(0.2, 0.8, 0.6, 0.6) == "bear"
    assert classify_phase(0.2, 0.2, 0.6, 0.6) == "none"
    assert classify_phase(0.8, 0.8, 0.6, 0.6) == "conflict"


def test_null_threshold_calibration_is_seeded_and_causal() -> None:
    sessions = np.arange(
        np.datetime64("2025-01-01"), np.datetime64("2025-01-09")
    ).astype("datetime64[D]")
    timestamps = np.concatenate(
        [session + np.arange(12).astype("timedelta64[m]") for session in sessions]
    ).astype("datetime64[ns]")
    events = np.zeros((len(timestamps), 4), dtype=np.bool_)
    for day in range(len(sessions)):
        offset = day * 12
        events[offset + 1, day % 4] = True
        events[offset + 3, (day + 1) % 4] = True
        events[offset + 6, (day + 2) % 4] = True
    calibrator = NullThresholdCalibrator(permutations=24, density_bins=4)

    first = calibrator.fit(
        events,
        timestamps,
        _full_support(4),
        sessions,
        direction="bull",
        month=np.datetime64("2025-02", "M"),
        rng=np.random.default_rng(7),
    )
    second = calibrator.fit(
        events,
        timestamps,
        _full_support(4),
        sessions,
        direction="bull",
        month=np.datetime64("2025-02", "M"),
        rng=np.random.default_rng(7),
    )

    np.testing.assert_allclose(first.thresholds, second.thresholds)
    assert first.training_cutoff == sessions[-1]
    assert first.training_cutoff < np.datetime64("2025-02-01")
    assert first.permutations == 24
    assert 0.0 <= first.threshold_for(0.2) <= 1.0

