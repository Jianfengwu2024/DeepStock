from __future__ import annotations

import numpy as np

from bnhmm.boundary_fk.sector_state import SECTOR_ROLLING_FEATURE_NAMES
from bnhmm.boundary_fk.types import (
    TwoBranchDailyStates,
    TwoBranchDataset,
    TwoBranchLabels,
)


def walk_forward_sector_data(rows: int = 180) -> TwoBranchDataset:
    rng = np.random.default_rng(41)
    sessions = np.busday_offset(np.datetime64("2024-01-02"), np.arange(rows))
    driver = rng.normal(size=rows)
    target = np.sign(driver) * 0.01 + rng.normal(0.0, 0.001, rows)
    values = np.zeros(
        (rows, 1, len(SECTOR_ROLLING_FEATURE_NAMES)), dtype=np.float64
    )
    index = {
        name: column for column, name in enumerate(SECTOR_ROLLING_FEATURE_NAMES)
    }
    values[:, 0, index["rolling_log_position_60d"]] = driver * 0.10
    values[:, 0, index["rolling_drift_60d"]] = driver * 0.001
    values[:, 0, index["rolling_diffusion_60d"]] = np.abs(
        rng.normal(0.001, 0.0001, rows)
    )
    values[:, 0, index["rolling_path_efficiency_60d"]] = driver
    target_time = np.full(rows, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    target_time[:-1] = sessions[1:].astype("datetime64[ns]")
    label_valid = np.ones((rows, 1), dtype=np.bool_)
    label_valid[-1] = False
    symbols = ("A", "B", "C", "D")
    member_returns = np.repeat(target[:, None], len(symbols), axis=1)
    member_valid = np.ones_like(member_returns, dtype=np.bool_)
    convection = np.full((rows, 1, 4), 0.2, dtype=np.float64)
    return TwoBranchDataset(
        states=TwoBranchDailyStates(
            sessions=sessions,
            topology_months=sessions.astype("datetime64[M]"),
            feature_names=SECTOR_ROLLING_FEATURE_NAMES,
            values=values,
            valid=np.ones((rows, 1), dtype=np.bool_),
        ),
        labels=TwoBranchLabels(
            signal_sessions=sessions,
            target_time=target_time,
            next_return=target[:, None],
            historical_volatility=np.full((rows, 1), 0.01),
            valid=label_valid,
        ),
        memberships=(),
        symbols=symbols,
        member_next_returns=member_returns,
        member_next_valid=member_valid,
        convection_strength=convection,
        convection_valid=np.ones_like(convection, dtype=np.bool_),
        convection_total_flow=convection.copy(),
        convection_observations=np.full_like(convection, 100, dtype=np.int64),
        evaluation_start=sessions[120] if rows > 120 else None,
        evaluation_end=sessions[-1],
    )
