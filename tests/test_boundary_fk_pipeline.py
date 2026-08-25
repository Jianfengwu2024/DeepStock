from __future__ import annotations

import json

import numpy as np

from bnhmm.boundary_fk.pipeline import (
    BOUNDARY_MODEL_NAMES,
    BoundaryFlowPipeline,
    BoundaryFlowPipelineConfig,
)
from bnhmm.boundary_fk.types import (
    MonthlyBranchMembership,
    TwoBranchDataset,
)
from test_two_branch_fk_model import _dataset


def _walk_forward_data(rows: int = 180) -> TwoBranchDataset:
    states, labels = _dataset(rows)
    symbols = ("A", "B", "C", "D")
    member_returns = np.column_stack(
        (
            labels.next_return[:, 0],
            labels.next_return[:, 0],
            labels.next_return[:, 1],
            labels.next_return[:, 1],
        )
    )
    member_valid = np.isfinite(member_returns) & labels.valid.any(axis=1)[:, None]
    memberships = tuple(
        MonthlyBranchMembership(
            month=month,
            training_cutoff=month.astype("datetime64[ns]") - np.timedelta64(1, "ns"),
            branches=(("A", "B"), ("C", "D")),
        )
        for month in np.unique(states.topology_months)
    )
    return TwoBranchDataset(
        states=states,
        labels=labels,
        memberships=memberships,
        symbols=symbols,
        member_next_returns=member_returns,
        member_next_valid=member_valid,
    )


def test_pipeline_runs_one_two_branch_fk_stage_and_writes_causal_artifacts(tmp_path) -> None:
    data = _walk_forward_data()
    result = BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(
            minimum_train_sessions=60,
            max_folds=1,
            fk_paths=1000,
        )
    ).run(data, tmp_path)

    assert result.model_names == BOUNDARY_MODEL_NAMES == ("TwoBranchFK",)
    assert len(result.stages) == 1
    assert result.stages[0].status == "completed"
    predictions = json.loads(
        (tmp_path / "feynman-kac-predictions.json").read_text(encoding="utf-8")
    )
    assert predictions
    assert len(predictions[0]["branches"]) == 2
    assert "effective_drift" in predictions[0]["branches"][0]
    assert {
        "signal_time",
        "available_at",
        "target_time",
        "training_cutoff",
        "model_version",
        "selected_symbols",
        "gross_strategy_return",
    }.issubset(predictions[0])


def test_pipeline_portfolio_return_comes_from_selected_constituents(tmp_path) -> None:
    data = _walk_forward_data()
    BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(
            minimum_train_sessions=60,
            max_folds=1,
            fk_paths=1000,
        )
    ).run(data, tmp_path)
    predictions = json.loads(
        (tmp_path / "feynman-kac-predictions.json").read_text(encoding="utf-8")
    )
    opened = next(item for item in predictions if item["selected_symbols"])
    row = int(
        np.flatnonzero(
            data.states.sessions
            == np.datetime64(opened["signal_time"], "D")
        )[0]
    )
    columns = [data.symbols.index(symbol) for symbol in opened["selected_symbols"]]
    expected = float(np.mean(data.member_next_returns[row, columns]))

    assert opened["gross_strategy_return"] == expected
