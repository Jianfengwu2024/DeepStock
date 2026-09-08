from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

import bnhmm.boundary_fk as boundary_fk
from bnhmm.boundary_fk.pipeline import (
    BoundaryFlowPipeline,
    BoundaryFlowPipelineConfig,
    _passes_sector_direction_gate,
)
from bnhmm.boundary_fk.types import (
    MonthlyBranchMembership,
)
from boundary_fk_fixtures import walk_forward_sector_data


def test_boundary_fk_package_does_not_export_removed_model_catalog() -> None:
    removed = {
        "BOUNDARY_MODEL_NAMES",
        "BayesianBoundaryPosterior",
        "BayesianBoundaryStateConfig",
        "BayesianBoundaryStateModel",
        "BoundaryStateLabels",
        "BoundaryFeynmanKacForecaster",
        "BoundaryForecastObservation",
        "evaluate_boundary_forecasts",
        "BoundaryDailyStates",
    }

    assert removed.isdisjoint(boundary_fk.__all__)


def test_pipeline_predicts_one_sector_without_branch_memberships(tmp_path) -> None:
    result = BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(
            minimum_train_sessions=30,
            max_folds=1,
            fk_paths=1000,
            neutral_sigma=1e-12,
            minimum_direction_probability=0.51,
            minimum_signal_to_noise=0.0,
            minimum_survival_probability=0.0,
        )
    ).run(walk_forward_sector_data(), tmp_path)

    assert result.model_names == ("FK-Bayesian-Rolling60",)
    assert all(stage.metrics and stage.metrics.observations > 0 for stage in result.stages)
    predictions = json.loads((tmp_path / "feynman-kac-predictions.json").read_text())
    assert predictions
    assert "sector" in predictions[0]
    assert "branches" not in predictions[0]
    opened = [item for item in predictions if item["selected_symbols"]]
    assert opened
    assert all(len(item["selected_symbols"]) == 2 for item in opened)
    hurdle_predictions = [
        item
        for item in predictions
        if item["model_version"] == "FK-Bayesian-Rolling60"
    ]
    assert hurdle_predictions
    assert {
        "significant_move_probability",
        "conditional_bullish_probability",
        "prior_bullish_probability",
        "feature_log_odds_delta",
    }.issubset(hurdle_predictions[0]["sector"])
    posteriors = json.loads(
        (tmp_path / "boundary-model-posterior.json").read_text()
    )
    hurdle_posteriors = [
        item for item in posteriors if item["model_name"] == "FK-Bayesian-Rolling60"
    ]
    assert hurdle_posteriors
    assert hurdle_posteriors[0]["model_type"] == "bayesian_hurdle"
    assert hurdle_posteriors[0]["direction_label_sigma"] == 1e-12
    assert 0.5 <= hurdle_posteriors[0]["minimum_feature_evidence_probability"] <= 0.95


def test_pipeline_rejects_removed_branch_memberships(tmp_path) -> None:
    data = walk_forward_sector_data()
    first_session = data.states.sessions[0].astype("datetime64[D]")
    membership = MonthlyBranchMembership(
        effective_session=first_session,
        training_cutoff=first_session.astype("datetime64[ns]")
        - np.timedelta64(1, "ns"),
        branches=(("A", "B"), ("C", "D")),
    )

    with pytest.raises(ValueError, match="must not contain branch memberships"):
        BoundaryFlowPipeline(
            BoundaryFlowPipelineConfig(
                minimum_train_sessions=30,
                max_folds=1,
                fk_paths=1000,
                neutral_sigma=1e-12,
            )
        ).run(replace(data, memberships=(membership,)), tmp_path)


def test_sector_pipeline_reports_direction_baselines(tmp_path) -> None:
    BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(
            minimum_train_sessions=30,
            max_folds=1,
            fk_paths=1000,
            neutral_sigma=1e-12,
            minimum_signal_to_noise=0.0,
            minimum_survival_probability=0.0,
        )
    ).run(walk_forward_sector_data(), tmp_path)

    report = json.loads((tmp_path / "sector-direction-evaluation.json").read_text())
    assert set(report["baselines"]) == {
        "previous_day_direction",
        "path_efficiency_direction",
    }
    assert len(report["models"]) == 1
    assert report["preferred_model"] == "FK-Bayesian-Rolling60"
    assert report["selection_rule"] == "fixed_predeclared_baseline"
    for model in report["models"]:
        metrics = model["metrics"]
        expected = (
            metrics["direction_signal_count"] >= 30
            and metrics["direction_signal_bull_count"] >= 15
            and metrics["direction_signal_bear_count"] >= 15
            and metrics["direction_signal_accuracy"]
            > metrics["direction_signal_majority_baseline"]
            and metrics["direction_signal_balanced_accuracy"] > 0.5
        )
        assert model["passed"] is expected
        assert model["promotion_eligible"] is expected
        assert model["monthly_direction_accuracy"]

    markdown = (tmp_path / "path-efficiency-fk-report.md").read_text(
        encoding="utf-8"
    )
    assert "预测多" in markdown
    assert "预测空" in markdown
    assert "失败原因" in markdown
    assert "FK-Bayesian-Rolling60" in markdown
    assert "不再保留候选、正式或旧模型" in markdown


def test_sector_direction_gate_rejects_one_sided_predictions(tmp_path) -> None:
    result = BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(
            minimum_train_sessions=30,
            max_folds=1,
            fk_paths=1000,
            neutral_sigma=1e-12,
            minimum_signal_to_noise=0.0,
            minimum_survival_probability=0.0,
        )
    ).run(walk_forward_sector_data(), tmp_path)
    metrics = result.stages[0].metrics
    assert metrics is not None
    one_sided = replace(
        metrics,
        direction_signal_count=64,
        direction_signal_accuracy=0.70,
        direction_signal_majority_baseline=0.60,
        direction_signal_balanced_accuracy=0.55,
        direction_signal_bull_count=63,
        direction_signal_bear_count=1,
    )

    assert not _passes_sector_direction_gate(one_sided)
