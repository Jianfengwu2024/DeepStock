from __future__ import annotations

import numpy as np
import pytest

from bnhmm.percolation_branching.backbone import SlowBackboneBuilder
from bnhmm.percolation_branching.branching import (
    BayesianBranchingModel,
    spectral_radius,
)
from bnhmm.percolation_branching.types import (
    ActivationPanel,
    MonthlyBackboneSupport,
)


def _activation_panel(
    *, mode: str, sessions: int = 72, symbols: int = 4, minutes: int = 12
) -> ActivationPanel:
    days = np.arange(
        np.datetime64("2024-10-01"),
        np.datetime64("2024-10-01") + np.timedelta64(sessions, "D"),
    ).astype("datetime64[D]")
    timestamps = np.concatenate(
        [day + np.arange(minutes).astype("timedelta64[m]") for day in days]
    ).astype("datetime64[ns]")
    shape = (len(timestamps), symbols)
    bull = np.zeros(shape, dtype=np.bool_)
    for day in range(sessions):
        offset = day * minutes
        if mode == "chain":
            for symbol in range(symbols):
                bull[offset + 2 + symbol, symbol] = True
        elif mode == "common":
            bull[offset + 4, :] = True
        elif mode == "self":
            bull[offset + 2 : offset + 7, 0] = True
        else:
            raise ValueError(mode)
    velocity = np.where(bull, 2.0, 0.0)
    return ActivationPanel(
        symbols=tuple(f"S{index}" for index in range(symbols)),
        timestamps=timestamps,
        standardized_velocity=velocity,
        velocity_scale=np.ones(shape),
        relative_volume=np.ones(shape),
        volume_baseline=np.ones(shape),
        marks=np.ones(shape),
        bull=bull,
        bear=np.zeros(shape, dtype=np.bool_),
        valid=np.ones(shape, dtype=np.bool_),
    )


def _full_support(panel: ActivationPanel) -> MonthlyBackboneSupport:
    count = len(panel.symbols)
    adjacency = np.ones((count, count), dtype=np.bool_)
    np.fill_diagonal(adjacency, False)
    return MonthlyBackboneSupport(
        month=np.datetime64("2025-01", "M"),
        training_cutoff=np.datetime64("2024-12-31"),
        symbols=panel.symbols,
        adjacency=adjacency,
        similarity=adjacency.astype(np.float64),
        session_count=60,
        completeness=np.ones(count),
    )


def test_backbone_uses_only_sessions_before_month() -> None:
    panel = _activation_panel(mode="chain")

    support = SlowBackboneBuilder(lookback_sessions=60, candidate_neighbors=3).build(
        panel, np.datetime64("2025-01", "M")
    )

    assert support.training_cutoff < np.datetime64("2025-01-01")
    assert not np.any(np.diag(support.adjacency))
    assert np.all(support.adjacency == support.adjacency.T)
    assert support.session_count == 60


def test_branching_recovers_directed_chain_and_unit_spectral_shape() -> None:
    panel = _activation_panel(mode="chain")
    sessions = np.unique(panel.timestamps.astype("datetime64[D]"))[-60:]

    posterior = BayesianBranchingModel(iterations=16).fit(
        panel,
        support=_full_support(panel),
        training_sessions=sessions,
        direction="bull",
    )

    assert posterior.mean[0, 1] > posterior.mean[1, 0]
    assert spectral_radius(posterior.normalized_mean) == pytest.approx(1.0)


def test_common_shocks_stay_external_and_self_runs_stay_self() -> None:
    common = _activation_panel(mode="common")
    repeated = _activation_panel(mode="self")
    sessions = np.unique(common.timestamps.astype("datetime64[D]"))[-60:]
    model = BayesianBranchingModel(iterations=12)

    common_fit = model.fit(
        common, _full_support(common), sessions, direction="bull"
    )
    self_fit = model.fit(
        repeated, _full_support(repeated), sessions, direction="bull"
    )

    assert float(np.mean(common_fit.external_rate)) > float(np.mean(common_fit.mean))
    assert self_fit.self_rate[0] > float(np.max(self_fit.mean))
    assert spectral_radius(self_fit.normalized_mean) == pytest.approx(1.0)

