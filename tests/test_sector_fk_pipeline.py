from __future__ import annotations

import json

import numpy as np

from bnhmm.boundary_fk.branch_state import BRANCH_FEATURE_NAMES
from bnhmm.boundary_fk.pipeline import BoundaryFlowPipeline, BoundaryFlowPipelineConfig
from bnhmm.boundary_fk.types import TwoBranchDailyStates, TwoBranchDataset, TwoBranchLabels


def _direct_dataset(rows: int = 180) -> TwoBranchDataset:
    rng = np.random.default_rng(41)
    sessions = np.busday_offset(np.datetime64("2024-01-02"), np.arange(rows))
    driver = rng.normal(size=rows)
    target = np.sign(driver) * 0.01 + rng.normal(0.0, 0.001, rows)
    values = np.zeros((rows, 1, len(BRANCH_FEATURE_NAMES)), dtype=np.float64)
    index = {name: column for column, name in enumerate(BRANCH_FEATURE_NAMES)}
    values[:, 0, index["return_1d"]] = rng.normal(0.0, 0.01, rows)
    values[:, 0, index["return_mean_5"]] = rng.normal(0.0, 0.005, rows)
    values[:, 0, index["return_mean_20"]] = rng.normal(0.0, 0.003, rows)
    values[:, 0, index["volatility_20"]] = 0.01
    values[:, 0, index["coherence_1d"]] = 0.8
    values[:, 0, index["signed_path_efficiency"]] = driver
    values[1:, 0, index["path_efficiency_delta_1d"]] = np.diff(driver)
    values[:, 0, index["path_direction_consistency_3d"]] = np.sign(driver)
    values[:, 0, index["signed_member_synchronization"]] = np.sign(driver) * 0.75
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
            feature_names=BRANCH_FEATURE_NAMES,
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
        evaluation_start=sessions[120],
        evaluation_end=sessions[-1],
    )


def test_pipeline_predicts_one_sector_without_branch_memberships(tmp_path) -> None:
    result = BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(
            minimum_train_sessions=30,
            max_folds=1,
            fk_paths=1000,
            direction_label_sigma=0.0,
            minimum_direction_probability=0.51,
            minimum_signal_to_noise=0.0,
            minimum_survival_probability=0.0,
        )
    ).run(_direct_dataset(), tmp_path)

    assert result.model_names == (
        "SectorFK-PathOnly",
        "SectorFK-PathSync",
        "SectorFK-PathMemory",
        "SectorFK-PathFull",
    )
    assert all(stage.metrics and stage.metrics.observations > 0 for stage in result.stages)
    predictions = json.loads((tmp_path / "feynman-kac-predictions.json").read_text())
    assert predictions
    assert "sector" in predictions[0]
    assert "branches" not in predictions[0]
    opened = [item for item in predictions if item["selected_symbols"]]
    assert opened
    assert all(len(item["selected_symbols"]) == 2 for item in opened)


def test_sector_pipeline_reports_direction_baselines(tmp_path) -> None:
    BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(
            minimum_train_sessions=30,
            max_folds=1,
            fk_paths=1000,
            direction_label_sigma=0.0,
            minimum_signal_to_noise=0.0,
            minimum_survival_probability=0.0,
        )
    ).run(_direct_dataset(), tmp_path)

    report = json.loads((tmp_path / "sector-direction-evaluation.json").read_text())
    assert set(report["baselines"]) == {
        "majority_direction",
        "previous_day_direction",
        "path_efficiency_direction",
    }
    assert len(report["models"]) == 4
