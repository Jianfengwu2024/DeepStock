from __future__ import annotations

import numpy as np
import pytest

from bnhmm.boundary_fk.moments import BoundaryFieldConfig, BoundaryFieldExtractor
from bnhmm.boundary_fk.types import (
    MonthlyIndexWeights,
    NormalizedBoundaryPanel,
    SyntheticSectorIndex,
)


def _inputs(
    *,
    raw: list[list[float]],
    standardized: list[list[float]],
    density: list[list[float]],
    index_velocity: list[float],
    symbols: tuple[str, ...] = ("A", "B", "C"),
    weights: list[float] | None = None,
    feature_valid: list[list[bool]] | None = None,
) -> tuple[NormalizedBoundaryPanel, SyntheticSectorIndex]:
    rows = len(raw)
    timestamps = np.asarray(
        [np.datetime64("2024-01-02T09:31") + np.timedelta64(i, "m") for i in range(rows)],
        dtype="datetime64[ns]",
    )
    raw_array = np.asarray(raw, dtype=np.float64)
    z = np.asarray(standardized, dtype=np.float64)
    rho = np.asarray(density, dtype=np.float64)
    valid = (
        np.asarray(feature_valid, dtype=np.bool_)
        if feature_valid is not None
        else np.ones(raw_array.shape, dtype=np.bool_)
    )
    normalized = NormalizedBoundaryPanel(
        symbols=symbols,
        timestamps=timestamps,
        raw_velocity=raw_array,
        standardized_velocity=z,
        relative_volume=rho,
        velocity_scale=np.ones(raw_array.shape),
        minute_volume_baseline=np.ones(raw_array.shape),
        raw_velocity_valid=valid.copy(),
        velocity_valid=valid.copy(),
        density_valid=valid.copy(),
    )
    frozen = MonthlyIndexWeights(
        month=np.datetime64("2024-01", "M"),
        weight_date=np.datetime64("2023-12-29"),
        symbols=symbols,
        weights=np.asarray(weights or [1.0 / 3.0] * 3, dtype=np.float64),
        training_cutoff=np.datetime64("2023-12-29"),
        excluded_symbols=(),
    )
    index_values = np.asarray(index_velocity, dtype=np.float64)
    sessions = np.asarray(["2024-01-02"], dtype="datetime64[D]")
    index = SyntheticSectorIndex(
        timestamps=timestamps,
        velocity=index_values,
        valid=np.isfinite(index_values),
        active_weight=np.ones(rows),
        active_fraction=np.ones(rows),
        monthly_weights=(frozen,),
        daily_sessions=sessions,
        daily_open_to_close_return=np.asarray([0.0]),
        daily_valid=np.asarray([True]),
        daily_active_weight=np.asarray([1.0]),
        daily_active_fraction=np.asarray([1.0]),
        excluded_symbols=(),
    )
    return normalized, index


def _extract(
    normalized: NormalizedBoundaryPanel,
    index: SyntheticSectorIndex,
) -> object:
    return BoundaryFieldExtractor(
        BoundaryFieldConfig(
            minimum_symbol_fraction=0.5,
            viscosity_window=10,
            minimum_viscosity_transitions=0,
            thickness_slope_window=3,
        )
    ).extract(
        normalized,
        index,
        index_velocity_scale=0.01,
        thickness_stabilizer=0.25,
    )


def test_high_volume_same_direction_flow_has_high_coherence() -> None:
    normalized, index = _inputs(
        raw=[[0.01, 0.01, 0.01]],
        standardized=[[1.0, 1.0, 1.0]],
        density=[[4.0, 4.0, 4.0]],
        index_velocity=[0.01],
    )

    moments = _extract(normalized, index)

    assert moments.valid[-1]
    assert moments.water[-1] == pytest.approx(4.0)
    assert moments.momentum[-1] == pytest.approx(0.04)
    assert moments.velocity[-1] == pytest.approx(0.01)
    assert moments.coherence[-1] > 0.999
    assert moments.turbulence[-1] < 1e-12
    assert moments.breadth[-1] == pytest.approx(1.0)
    assert moments.effective_constituents[-1] == pytest.approx(3.0)


def test_opposing_flow_has_low_coherence_and_high_turbulence() -> None:
    normalized, index = _inputs(
        raw=[[0.01, -0.01, 0.0]],
        standardized=[[1.0, -1.0, 0.0]],
        density=[[1.0, 1.0, 1.0]],
        index_velocity=[0.0],
    )

    moments = _extract(normalized, index)

    assert moments.coherence[-1] < 1e-12
    assert moments.turbulence[-1] == pytest.approx(2.0 / 3.0)
    assert moments.breadth[-1] == pytest.approx(0.0)
    assert moments.thickness[-1] > 3.0


def test_more_participation_in_rising_stock_reduces_index_decoupling() -> None:
    weak, weak_index = _inputs(
        raw=[[0.02, 0.0, 0.0]],
        standardized=[[2.0, 0.0, 0.0]],
        density=[[0.1, 1.0, 1.0]],
        index_velocity=[0.01],
    )
    strong, strong_index = _inputs(
        raw=[[0.02, 0.0, 0.0]],
        standardized=[[2.0, 0.0, 0.0]],
        density=[[4.0, 1.0, 1.0]],
        index_velocity=[0.01],
    )

    weak_moments = _extract(weak, weak_index)
    strong_moments = _extract(strong, strong_index)

    assert abs(strong_moments.decoupling[-1]) < abs(weak_moments.decoupling[-1])
    assert strong_moments.momentum[-1] > weak_moments.momentum[-1]


def test_missing_stock_is_not_zero_filled_and_available_mass_is_preserved() -> None:
    normalized, index = _inputs(
        raw=[[0.01, 99.0, 0.01]],
        standardized=[[1.0, 99.0, 1.0]],
        density=[[1.0, 99.0, 1.0]],
        index_velocity=[0.01],
        feature_valid=[[True, False, True]],
    )

    moments = _extract(normalized, index)

    assert moments.valid[-1]
    assert moments.active_fraction[-1] == pytest.approx(2.0 / 3.0)
    assert moments.velocity[-1] == pytest.approx(0.01)
    assert moments.coherence[-1] > 0.999


def test_symbol_permutation_does_not_change_fluid_moments() -> None:
    base, base_index = _inputs(
        raw=[[0.02, -0.01, 0.005]],
        standardized=[[2.0, -1.0, 0.5]],
        density=[[3.0, 1.0, 2.0]],
        index_velocity=[0.008],
        weights=[0.2, 0.3, 0.5],
    )
    changed, changed_index = _inputs(
        raw=[[0.005, 0.02, -0.01]],
        standardized=[[0.5, 2.0, -1.0]],
        density=[[2.0, 3.0, 1.0]],
        index_velocity=[0.008],
        symbols=("C", "A", "B"),
        weights=[0.5, 0.2, 0.3],
    )

    left = _extract(base, base_index)
    right = _extract(changed, changed_index)

    for name in (
        "water",
        "momentum",
        "velocity",
        "coherence",
        "turbulence",
        "breadth",
        "concentration",
        "effective_constituents",
        "decoupling",
        "thickness",
    ):
        np.testing.assert_allclose(getattr(left, name), getattr(right, name))


def test_low_active_fraction_invalidates_the_minute() -> None:
    normalized, index = _inputs(
        raw=[[0.01, 0.0, 0.0]],
        standardized=[[1.0, 0.0, 0.0]],
        density=[[1.0, 0.0, 0.0]],
        index_velocity=[0.01],
        feature_valid=[[True, False, False]],
    )

    moments = _extract(normalized, index)

    assert not moments.valid[-1]
    assert np.isnan(moments.velocity[-1])
