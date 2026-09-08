from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from bnhmm.boundary_fk.branch_state import BRANCH_FEATURE_NAMES
from bnhmm.boundary_fk.pipeline import (
    BoundaryFlowPipeline,
    BoundaryFlowPipelineConfig,
    _passes_sector_direction_gate,
)
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
            neutral_sigma=1e-12,
            minimum_direction_probability=0.51,
            minimum_signal_to_noise=0.0,
            minimum_survival_probability=0.0,
        )
    ).run(_direct_dataset(), tmp_path)

    assert result.model_names == (
        "SectorFK-LabelAligned",
        "SectorFK-BayesianHurdle",
        "SectorFK-FeatureEvidence",
        "SectorFK-Regime",
    )
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
        if item["model_version"] == "SectorFK-BayesianHurdle"
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
        item for item in posteriors if item["model_name"] == "SectorFK-BayesianHurdle"
    ]
    assert hurdle_posteriors
    assert hurdle_posteriors[0]["model_type"] == "bayesian_hurdle"
    assert hurdle_posteriors[0]["direction_label_sigma"] == 1e-12
    evidence_posteriors = [
        item for item in posteriors if item["model_name"] == "SectorFK-FeatureEvidence"
    ]
    assert evidence_posteriors
    assert 0.5 <= evidence_posteriors[0]["minimum_feature_evidence_probability"] <= 0.95
    regime_posteriors = [
        item for item in posteriors if item["model_name"] == "SectorFK-Regime"
    ]
    assert regime_posteriors
    assert regime_posteriors[0]["model_type"] == "bayesian_path_regime"
    assert 0.0 <= regime_posteriors[0]["next_agreement_probability"] <= 1.0


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
    ).run(_direct_dataset(), tmp_path)

    report = json.loads((tmp_path / "sector-direction-evaluation.json").read_text())
    assert set(report["baselines"]) == {
        "previous_day_direction",
        "path_efficiency_direction",
    }
    assert len(report["models"]) == 4
    assert report["preferred_model"] == "SectorFK-FeatureEvidence"
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
    assert "SectorFK-FeatureEvidence" in markdown
    assert "SectorFK-Regime` 已作为实验否决" in markdown


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
    ).run(_direct_dataset(), tmp_path)
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
