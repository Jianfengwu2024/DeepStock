from __future__ import annotations

from dataclasses import replace
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
    convection_shape = (rows, 2, 4)
    convection_strength = np.zeros(convection_shape, dtype=np.float64)
    convection_strength[:, :, 0] = np.linspace(0.1, 0.9, rows)[:, None]
    convection_valid = np.zeros(convection_shape, dtype=np.bool_)
    convection_valid[:, :, 0] = True
    return TwoBranchDataset(
        states=states,
        labels=labels,
        memberships=memberships,
        symbols=symbols,
        member_next_returns=member_returns,
        member_next_valid=member_valid,
        convection_strength=convection_strength,
        convection_valid=convection_valid,
    )


def test_pipeline_compares_baseline_with_one_day_potential_fk(tmp_path) -> None:
    base = _walk_forward_data()
    shape = (len(base.states.sessions), 2, 4)
    data = replace(
        base,
        labels=replace(
            base.labels,
            historical_volatility=np.full_like(
                base.labels.historical_volatility, 1e-6
            ),
        ),
        convection_strength=np.full(shape, 0.2),
        convection_valid=np.ones(shape, dtype=np.bool_),
    )
    result = BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(
            minimum_train_sessions=60,
            max_folds=1,
            fk_paths=1000,
        )
    ).run(data, tmp_path)

    assert result.model_names == BOUNDARY_MODEL_NAMES == (
        "PathEfficiencyFK-60D-Beta0PlusC1-Baseline",
        "PathEfficiencyFK-60D-StrongestThird",
    )
    assert len(result.stages) == 2
    assert result.stages[0].status == "completed"
    predictions = json.loads(
        (tmp_path / "feynman-kac-predictions.json").read_text(encoding="utf-8")
    )
    assert predictions
    assert len(predictions[0]["branches"]) == 2
    assert "effective_drift" in predictions[0]["branches"][0]
    assert abs(predictions[0]["branches"][0]["effective_drift"]) <= abs(
        predictions[0]["branches"][0]["expected_drift"]
    )
    assert {
        "bullish_drift_probability",
        "drift_signal_to_noise",
        "survival_confidence",
    }.issubset(predictions[0]["branches"][0])
    assert predictions[0]["branches"][0]["convection_potential_features"] == {
        "reverse_flow_1d": 0.2
    }
    assert "expected_effective_potential" in predictions[0]["branches"][0]
    assert {
        "signal_time",
        "available_at",
        "target_time",
        "training_cutoff",
        "model_version",
        "selected_symbols",
        "branch_target_exposures",
        "branch_flat_signal_counts",
        "gross_strategy_return",
    }.issubset(predictions[0])
    assert not (tmp_path / "boundary-policy-ablation.json").exists()
    assert all(
        round(sum(item["target_weights"]), 8)
        in {0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0}
        for item in predictions
    )
    posteriors = json.loads(
        (tmp_path / "boundary-model-posterior.json").read_text(encoding="utf-8")
    )
    assert {item["policy"] for item in posteriors} == {"path_efficiency_fk"}
    assert {item["potential_mode"] for item in posteriors} == {"beta0_c1"}
    assert all(item["potential_feature_names"] == ["reverse_flow_1d"] for item in posteriors)
    assert all(len(item["potential_mean"]) == 2 for item in posteriors)
    expected_drift_features = [
        "signed_path_efficiency",
        "signed_member_synchronization",
    ]
    assert all(
        item["drift_feature_names"] == expected_drift_features
        for item in posteriors
    )
    assert all(item["drift_model"] == "bayesian_direction_logit" for item in posteriors)
    assert all(item["drift_baseline_mode"] == "bayesian_intercept" for item in posteriors)
    assert all(np.isfinite(item["drift_mean"][0]) for item in posteriors)
    assert result.stages[0].drift_feature_names == tuple(expected_drift_features)
    assert 0.0 <= result.stages[0].metrics.direction_signal_coverage <= 1.0
    assert 0.0 <= result.stages[0].metrics.direction_signal_accuracy <= 1.0
    assert (tmp_path / "path-efficiency-fk-report.md").exists()
    assert (tmp_path / "path-efficiency-convection-potential-report.md").exists()
    diagnostics = json.loads(
        (tmp_path / "convection-potential-evaluation.json").read_text(encoding="utf-8")
    )
    assert "mean_effective_potential" in diagnostics
    assert "mean_trend_survival_probability" in diagnostics
    assert set(diagnostics["models"]) == set(BOUNDARY_MODEL_NAMES)
    assert (tmp_path / "boundary-ablation-report.md").exists()
    report = (tmp_path / "boundary-ablation-report.md").read_text(encoding="utf-8")
    assert "相对强弱" in report
    assert "最大回撤" in report
    assert "涨停未买入" in report
    assert 0.0 <= result.stages[0].metrics.average_exposure <= 1.0

    grouped: dict[str, list[dict[str, object]]] = {
        model: [item for item in predictions if item["model_version"] == model]
        for model in BOUNDARY_MODEL_NAMES
    }
    baseline_rows = grouped[BOUNDARY_MODEL_NAMES[0]]
    assert all(len(grouped[name]) == len(baseline_rows) for name in BOUNDARY_MODEL_NAMES)
    for row_index, baseline in enumerate(baseline_rows):
        for model_name in BOUNDARY_MODEL_NAMES[1:]:
            potential = grouped[model_name][row_index]
            assert baseline["signal_time"] == potential["signal_time"]
            assert [item["expected_drift"] for item in baseline["branches"]] == [
                item["expected_drift"] for item in potential["branches"]
            ]
            assert all(
                0.0 <= item["survival_confidence"] <= 1.0
                for item in potential["branches"]
            )
            assert len(potential["selected_symbols"]) <= len(baseline["selected_symbols"])


def test_walk_forward_skips_folds_without_enough_direction_labels() -> None:
    base = _walk_forward_data(240)
    returns = base.labels.next_return.copy()
    returns[:100] = 1e-6
    data = replace(base, labels=replace(base.labels, next_return=returns))
    pipeline = BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(
            minimum_train_sessions=60,
            max_folds=1,
            fk_paths=1000,
            direction_label_sigma=0.1,
        )
    )

    folds = pipeline._mature_monthly_folds(data)

    assert folds
    first_training_rows = folds[0][2]
    for branch in range(2):
        significant = np.abs(data.labels.next_return[first_training_rows, branch]) > (
            0.1 * data.labels.historical_volatility[first_training_rows, branch]
        )
        assert int(significant.sum()) >= 60


def test_strongest_third_ranking_is_causal_and_keeps_one_of_three() -> None:
    data = _walk_forward_data(180)
    symbols = ("A", "B", "C", "D", "E", "F")
    returns = np.zeros((180, 6), dtype=np.float64)
    returns[:, :3] = (0.01, 0.005, -0.005)
    returns[:, 3:] = (0.008, 0.002, -0.004)
    memberships = tuple(
        MonthlyBranchMembership(
            month=item.month,
            training_cutoff=item.training_cutoff,
            branches=(("A", "B", "C"), ("D", "E", "F")),
        )
        for item in data.memberships
    )
    ranked_data = replace(
        data,
        symbols=symbols,
        memberships=memberships,
        member_next_returns=returns,
        member_next_valid=np.ones_like(returns, dtype=np.bool_),
        member_next_hold_returns=returns,
        member_next_buy_returns=returns,
        member_next_buy_executable=np.ones_like(returns, dtype=np.bool_),
        member_next_sell_returns=np.zeros_like(returns),
        member_next_sell_executable=np.ones_like(returns, dtype=np.bool_),
    )
    pipeline = BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(relative_strength_lookback_sessions=20)
    )
    row = 100

    selected, scores = pipeline._strongest_third_branches(
        ranked_data, row, memberships[0].branches
    )
    changed = returns.copy()
    changed[row, 2] = 10.0
    repeated, _ = pipeline._strongest_third_branches(
        replace(ranked_data, member_next_returns=changed),
        row,
        memberships[0].branches,
    )

    assert selected == (("A",), ("D",))
    assert repeated == selected
    assert scores["A"] > scores["B"] > scores["C"]


def test_pipeline_portfolio_return_comes_from_selected_constituents(tmp_path) -> None:
    data = _walk_forward_data()
    BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(
            minimum_train_sessions=60,
            max_folds=1,
            fk_paths=1000,
                minimum_direction_probability=0.5,
                minimum_signal_to_noise=0.0,
                minimum_survival_probability=0.0,
                direction_label_sigma=0.0,
            )
    ).run(data, tmp_path)
    predictions = json.loads(
        (tmp_path / "feynman-kac-predictions.json").read_text(encoding="utf-8")
    )
    opened = next(
        item
        for item in predictions
        if item["model_version"] == BOUNDARY_MODEL_NAMES[0]
        and item["selected_symbols"]
    )
    row = int(
        np.flatnonzero(
            data.states.sessions
            == np.datetime64(opened["signal_time"], "D")
        )[0]
    )
    expected = sum(
        weight * float(data.member_next_returns[row, data.symbols.index(symbol)])
        for symbol, weight in zip(
            opened["selected_symbols"], opened["target_weights"], strict=True
        )
    )

    np.testing.assert_allclose(opened["gross_strategy_return"], expected)
