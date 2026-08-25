from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from bnhmm.boundary_fk.direct_pipeline import (
    DirectPathPipeline,
    DirectPathPipelineConfig,
)
from test_boundary_fk_pipeline import _walk_forward_data


def _direct_data():
    data = _walk_forward_data(90)
    values = data.states.values.copy()
    column = data.states.feature_names.index("signed_path_efficiency")
    pattern = np.asarray((0.6, -0.4, 0.0), dtype=np.float64)
    values[:, 0, column] = np.resize(pattern, len(values))
    values[:, 1, column] = np.resize(-pattern, len(values))
    states = replace(data.states, values=values)
    return replace(data, states=states)


def test_direct_pipeline_writes_causal_sign_predictions_and_costed_metrics(tmp_path) -> None:
    data = _direct_data()
    result = DirectPathPipeline(
        DirectPathPipelineConfig(transaction_cost_bps=10.0)
    ).run(data, tmp_path)

    predictions = json.loads(
        (tmp_path / "path-efficiency-predictions.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (tmp_path / "path-efficiency-evaluation.json").read_text(encoding="utf-8")
    )

    assert result.model_name == "PathEfficiency"
    assert predictions
    assert evaluation[0]["model_name"] == "PathEfficiency"
    assert {
        "signal_time",
        "available_at",
        "target_time",
        "training_cutoff",
        "model_version",
        "branches",
        "branch_target_exposures",
        "selected_symbols",
        "target_weights",
        "gross_strategy_return",
        "turnover",
    }.issubset(predictions[0])
    assert {
        "path_efficiency",
        "direction",
        "actual_return",
        "target_exposure",
    }.issubset(predictions[0]["branches"][0])
    assert all(
        np.datetime64(item["signal_time"]) < np.datetime64(item["available_at"])
        < np.datetime64(item["target_time"])
        for item in predictions
    )

    net_returns = np.asarray(
        [
            item["gross_strategy_return"]
            - item["turnover"] * 10.0 / 10_000.0
            for item in predictions
        ]
    )
    expected_net = float(np.prod(1.0 + net_returns) - 1.0)
    assert evaluation[0]["metrics"]["net_total_return"] == pytest.approx(
        expected_net
    )
    assert 0.0 <= evaluation[0]["metrics"]["average_exposure"] <= 1.0
    assert evaluation[0]["benchmark"]["average_exposure"] == pytest.approx(1.0)
    assert (tmp_path / "path-efficiency-report.md").exists()


def test_zero_signal_keeps_the_previous_pipeline_position(tmp_path) -> None:
    DirectPathPipeline().run(_direct_data(), tmp_path)
    predictions = json.loads(
        (tmp_path / "path-efficiency-predictions.json").read_text(encoding="utf-8")
    )
    zero_day = next(
        item
        for item in predictions
        if item["branches"][0]["path_efficiency"] == 0.0
    )

    assert zero_day["branches"][0]["direction"] == "hold"
    assert zero_day["branches"][0]["target_exposure"] == pytest.approx(0.0)
    assert zero_day["branches"][1]["target_exposure"] == pytest.approx(0.5)
