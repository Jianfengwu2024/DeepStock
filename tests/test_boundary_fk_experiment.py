from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np

from bnhmm.boundary_fk.pipeline import BoundaryFlowPipeline, BoundaryFlowPipelineConfig
from test_boundary_fk_pipeline import _walk_forward_data


def test_experiment_reads_precomputed_data_without_opening_minute_source(
    tmp_path, monkeypatch
) -> None:
    import bnhmm.boundary_fk.experiment as module

    database = tmp_path / "market.duckdb"
    database.write_bytes(b"source")
    loaded = _walk_forward_data(180)

    class FakeCache:
        def __init__(self, root):
            assert root == str(tmp_path / "cache")

        def load(self, key, **kwargs):
            assert key.lookback_sessions == 30
            return loaded

    class FakePipeline:
        def __init__(self, config):
            assert not hasattr(config, "convection_lookback_sessions")

        def run(self, data, output):
            assert data is loaded
            return SimpleNamespace(
                model_names=("PathEfficiencyFK",),
                stages=(SimpleNamespace(model_name="PathEfficiencyFK", status="completed", failure_reason=""),),
            )

    monkeypatch.setattr(module, "BoundaryDataCache", FakeCache)
    monkeypatch.setattr(module, "BoundaryFlowPipeline", FakePipeline)
    monkeypatch.setattr(
        module,
        "DuckDBBarSource",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("minute source opened")),
    )
    args = SimpleNamespace(
        symbols="A,B,C,D",
        start="2024-01-01",
        end="2024-12-31",
        output_dir=str(tmp_path / "out"),
        duckdb_path=str(database),
        dataset_id="semiconductor",
        cache_root=str(tmp_path / "cache"),
        topology_lookback_sessions=30,
        minimum_train_sessions=60,
        max_folds=1,
        fk_paths=1000,
        seed=7,
        neutral_sigma=0.35,
        drift_half_life_sessions=60.0,
        minimum_direction_probability=0.6,
        minimum_signal_to_noise=0.1,
        minimum_survival_probability=0.35,
        transaction_cost_bps=10.0,
    )

    module.run_boundary_flow_experiment(args)

    assert (tmp_path / "out" / "run-summary.json").is_file()
    assert '"convection_lookback_sessions"' not in (
        tmp_path / "out" / "run-summary.json"
    ).read_text(encoding="utf-8")


def test_requested_evaluation_range_limits_sample_out_months(tmp_path) -> None:
    data = _walk_forward_data(220)
    start = np.datetime64("2024-07-01")
    end = np.datetime64("2024-09-30")
    ranged = type(data)(
        states=data.states,
        labels=data.labels,
        memberships=data.memberships,
        symbols=data.symbols,
        member_next_returns=data.member_next_returns,
        member_next_valid=data.member_next_valid,
        convection_strength=data.convection_strength,
        convection_valid=data.convection_valid,
        convection_total_flow=data.convection_total_flow,
        convection_observations=data.convection_observations,
        evaluation_start=start,
        evaluation_end=end,
    )

    result = BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(minimum_train_sessions=60, max_folds=None, fk_paths=1000)
    ).run(ranged, tmp_path)

    assert result.stages[0].test_months
    assert all("2024-07" <= month <= "2024-09" for month in result.stages[0].test_months)
