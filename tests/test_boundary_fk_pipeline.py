from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from bnhmm.boundary_fk.pipeline import (
    BOUNDARY_MODEL_NAMES,
    BoundaryFlowPipeline,
    BoundaryFlowPipelineConfig,
    BoundaryWalkForwardData,
)
from test_boundary_fk_bayesian_state import _dataset


def _walk_forward_data() -> BoundaryWalkForwardData:
    states, labels = _dataset(180)
    return BoundaryWalkForwardData(
        states=states,
        labels=labels,
        thickness_risk_quantile=np.full(len(states.sessions), 0.4),
        gross_strategy_returns=labels.next_return.copy(),
    )


def test_pipeline_runs_named_ablations_on_the_same_fold_and_writes_artifacts(tmp_path) -> None:
    result = BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(
            minimum_train_sessions=60,
            max_folds=1,
            fk_paths=1000,
        ),
        stage_gate=lambda previous, current, name: True,
    ).run(_walk_forward_data(), tmp_path)

    assert result.model_names == BOUNDARY_MODEL_NAMES
    assert all(stage.status == "passed" for stage in result.stages)
    assert len({stage.test_months for stage in result.stages}) == 1
    assert result.stages[0].drift_feature_names != result.stages[1].drift_feature_names
    assert result.stages[1].drift_feature_names != result.stages[2].drift_feature_names
    assert (tmp_path / "boundary-ablation-report.md").is_file()
    assert (tmp_path / "feynman-kac-predictions.json").is_file()
    predictions = json.loads((tmp_path / "feynman-kac-predictions.json").read_text())
    assert predictions
    assert {
        "signal_time",
        "available_at",
        "target_time",
        "training_cutoff",
        "model_version",
    }.issubset(predictions[0])


def test_failed_stage_does_not_stop_later_diagnostics_or_restore_promotion(tmp_path) -> None:
    def fail_global(previous, current, name):
        return name != "BFK-Global"

    result = BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(
            minimum_train_sessions=60,
            max_folds=1,
            fk_paths=1000,
        ),
        stage_gate=fail_global,
    ).run(_walk_forward_data(), tmp_path)

    assert result.stages[0].status == "passed"
    assert result.stages[1].status == "failed"
    assert all(stage.status == "passed" for stage in result.stages[2:])
    assert all(stage.metrics is not None for stage in result.stages)
    assert [stage.promotion_eligible for stage in result.stages] == [
        True,
        False,
        False,
        False,
        False,
    ]

    predictions = json.loads((tmp_path / "feynman-kac-predictions.json").read_text())
    assert {item["model_version"] for item in predictions} == set(BOUNDARY_MODEL_NAMES)
    report = (tmp_path / "boundary-ablation-report.md").read_text(encoding="utf-8")
    assert "后续级别仍继续运行" in report
    assert "后续级别停止" not in report


def test_minimum_training_sessions_counts_only_mature_targets(tmp_path) -> None:
    data = _walk_forward_data()
    matured = data.labels.return_valid.copy()
    matured[:20] = False
    labels = replace(
        data.labels,
        return_valid=matured,
        survival_valid=data.labels.survival_valid & matured,
    )
    data = replace(data, labels=labels)

    result = BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(
            minimum_train_sessions=60,
            max_folds=1,
            fk_paths=1000,
        ),
        stage_gate=lambda previous, current, name: True,
    ).run(data, tmp_path)

    assert all(stage.minimum_fit_rows >= 60 for stage in result.stages)
