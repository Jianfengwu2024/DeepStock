from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bnhmm.boundary_fk.convection import (
    ConvectionGate,
    ConvectionGateDecision,
    ConvectionPersistenceGate,
)
from test_boundary_fk_pipeline import _walk_forward_data


def test_gate_quantiles_use_only_rows_matured_before_training_cutoff() -> None:
    base = _walk_forward_data(80)
    strength = np.full((80, 2, 4), 0.99, dtype=np.float64)
    training = np.linspace(0.10, 0.49, 40)
    strength[:40, 0, 2] = training
    strength[:40, 1, 2] = training[::-1]
    data = replace(
        base,
        convection_strength=strength,
        convection_valid=np.ones((80, 2, 4), dtype=np.bool_),
    )

    gate = ConvectionGate.fit(
        data,
        training_cutoff=base.labels.target_time[39],
        minimum_rows=30,
    )

    expected = tuple(float(item) for item in np.quantile(training, (0.50, 0.75, 0.90)))
    assert gate.thresholds[0] == pytest.approx(expected)
    assert gate.thresholds[1] == pytest.approx(expected)
    assert gate.training_rows == (40, 40)
    assert gate.lookback_sessions == 3


@pytest.mark.parametrize(
    ("strength", "band", "cap"),
    [
        (0.20, "low", 0.5),
        (0.55, "elevated", 0.375),
        (0.75, "high", 0.25),
        (0.95, "extreme", 0.0),
    ],
)
def test_gate_maps_convection_to_fixed_exposure_caps(
    strength: float, band: str, cap: float
) -> None:
    base = _walk_forward_data(40)
    values = np.full((40, 2, 4), strength, dtype=np.float64)
    data = replace(
        base,
        convection_strength=values,
        convection_valid=np.ones((40, 2, 4), dtype=np.bool_),
    )
    gate = ConvectionGate(
        thresholds=((0.50, 0.70, 0.90), (0.50, 0.70, 0.90)),
        training_rows=(40, 40),
        lookback_sessions=3,
    )

    decision = gate.decide(data, row=20, branch=0)

    assert decision.band == band
    assert decision.exposure_cap == cap
    assert decision.strength == strength


def test_invalid_or_uncalibrated_convection_does_not_limit_exposure() -> None:
    base = _walk_forward_data(40)
    gate = ConvectionGate(
        thresholds=(None, None), training_rows=(0, 0), lookback_sessions=3
    )

    decision = gate.decide(base, row=20, branch=0)

    assert decision.band == "unavailable"
    assert decision.exposure_cap == 0.5
    assert decision.strength is None


def test_gate_selects_requested_precomputed_window() -> None:
    base = _walk_forward_data(40)
    values = np.full((40, 2, 4), 0.2, dtype=np.float64)
    values[:, :, 0] = 0.95
    data = replace(
        base,
        convection_strength=values,
        convection_valid=np.ones((40, 2, 4), dtype=np.bool_),
    )
    gate = ConvectionGate(
        thresholds=((0.50, 0.70, 0.90), (0.50, 0.70, 0.90)),
        training_rows=(40, 40),
        lookback_sessions=1,
    )

    decision = gate.decide(data, row=20, branch=0)

    assert decision.strength == 0.95
    assert decision.band == "extreme"


def test_persistence_gate_waits_for_two_days_then_exits_on_three() -> None:
    gate = ConvectionPersistenceGate()
    event = ConvectionGateDecision(0.8, "high", 0.25, "above_q75")
    quiet = ConvectionGateDecision(0.2, "low", 0.5, "below_median")

    first = gate.apply((event, quiet))
    second = gate.apply((event, quiet))
    third = gate.apply((event, quiet))

    assert first[0].exposure_cap == 0.5
    assert second[0].exposure_cap == 0.25
    assert third[0].exposure_cap == 0.0
    assert first[1].exposure_cap == second[1].exposure_cap == 0.5


def test_persistence_gate_resets_after_convection_disappears() -> None:
    gate = ConvectionPersistenceGate()
    event = ConvectionGateDecision(0.8, "high", 0.25, "above_q75")
    quiet = ConvectionGateDecision(0.2, "low", 0.5, "below_median")

    gate.apply((event, quiet))
    reduced = gate.apply((event, quiet))
    reset = gate.apply((quiet, quiet))
    restarted = gate.apply((event, quiet))

    assert reduced[0].exposure_cap == 0.25
    assert reset[0].exposure_cap == 0.5
    assert restarted[0].exposure_cap == 0.5
