"""Real-data construction and CLI execution for PathEfficiencyFK."""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ..data.source import DuckDBBarSource
from ..sector_flow.panel import SectorMinutePanelBuilder
from ..sector_flow.topology import TopologyConfig
from .branch_state import TwoBranchDailyStateBuilder
from .cache import BoundaryDataCache, BoundaryDataCacheKey
from .daily_state import BoundaryTopologyBuilder
from .normalization import CausalBoundaryNormalizer
from .pipeline import BoundaryFlowPipeline, BoundaryFlowPipelineConfig
from .reporting import write_json
from .types import TwoBranchDataset


def run_boundary_flow_experiment(args: Any) -> None:
    started = time.perf_counter()
    symbols = tuple(item.strip() for item in args.symbols.split(",") if item.strip())
    if len(symbols) < 4:
        raise ValueError("PathEfficiencyFK requires at least four symbols")
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end <= start:
        raise ValueError("PathEfficiencyFK end must follow start")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    key = BoundaryDataCacheKey.from_source(
        dataset_id=args.dataset_id,
        source_path=args.duckdb_path,
        symbols=symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        lookback_sessions=args.topology_lookback_sessions,
    )
    data = BoundaryDataCache(args.cache_root).load(
        key,
        evaluation_start=start.isoformat(),
        evaluation_end=end.isoformat(),
    )
    convection_lookback_sessions = int(
        getattr(args, "convection_lookback_sessions", 1)
    )
    result = BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(
            minimum_train_sessions=args.minimum_train_sessions,
            max_folds=args.max_folds,
            fk_paths=args.fk_paths,
            seed=args.seed,
            neutral_sigma=args.neutral_sigma,
            drift_half_life_sessions=args.drift_half_life_sessions,
            minimum_direction_probability=args.minimum_direction_probability,
            minimum_signal_to_noise=args.minimum_signal_to_noise,
            minimum_survival_probability=args.minimum_survival_probability,
            transaction_cost_bps=args.transaction_cost_bps,
            convection_lookback_sessions=convection_lookback_sessions,
        )
    ).run(data, output)
    summary = {
        "strategy": "boundary-flow-fk",
        "display_name": "路径效率 FK",
        "dataset_id": args.dataset_id,
        "symbols": len(symbols),
        "data_source": "precomputed",
        "topology_lookback_sessions": args.topology_lookback_sessions,
        "convection_lookback_sessions": convection_lookback_sessions,
        "elapsed_seconds": time.perf_counter() - started,
        "model_names": list(result.model_names),
        "stages": [
            {
                "model_name": stage.model_name,
                "status": stage.status,
                "failure_reason": stage.failure_reason,
            }
            for stage in result.stages
        ],
    }
    write_json(output / "run-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_boundary_experiment_data(
    source: DuckDBBarSource,
    symbols: tuple[str, ...],
    start: date,
    end: date,
    *,
    topology_lookback_sessions: int,
) -> tuple[TwoBranchDataset, dict[str, Any]]:
    warmup_start = start - timedelta(days=max(450, topology_lookback_sessions * 5))
    bars = {
        symbol: source.fetch(symbol, "1m", warmup_start, end)
        for symbol in symbols
    }
    panel = SectorMinutePanelBuilder().build(bars)
    normalized = CausalBoundaryNormalizer().transform(panel)
    topology_start = start - timedelta(days=300)
    months = tuple(
        np.arange(
            np.datetime64(topology_start, "M"),
            np.datetime64(end, "M") + np.timedelta64(1, "M"),
            dtype="datetime64[M]",
        )
    )
    topologies = BoundaryTopologyBuilder(
        TopologyConfig(
            lookback_sessions=topology_lookback_sessions,
            maximum_bands=2,
            minimum_band_size=max(2, int(np.ceil(len(symbols) * 0.10))),
        )
    ).build(normalized, months)
    built = TwoBranchDailyStateBuilder().build(normalized, topologies)
    data = TwoBranchDataset(
        states=built.states,
        labels=built.labels,
        memberships=built.memberships,
        symbols=built.symbols,
        member_next_returns=built.member_next_returns,
        member_next_valid=built.member_next_valid,
        evaluation_start=np.datetime64(start),
        evaluation_end=np.datetime64(end),
    )
    return data, {"topologies": topologies, "memberships": built.memberships}


def _write_upstream_artifacts(output: Path, context: dict[str, Any]) -> None:
    topologies = context["topologies"]
    memberships = context["memberships"]
    write_json(
        output / "branch-topologies.json",
        [
            {
                "month": str(item.month),
                "training_cutoff": str(item.training_cutoff),
                "symbols": list(item.symbols),
                "assigned": item.assigned.tolist(),
                "band_ids": item.band_ids.tolist(),
                "row_order": item.row_order.tolist(),
            }
            for item in topologies
        ],
    )
    write_json(
        output / "branch-memberships.json",
        [
            {
                "month": str(item.month),
                "training_cutoff": str(item.training_cutoff),
                "branch_1": list(item.branches[0]),
                "branch_2": list(item.branches[1]),
            }
            for item in memberships
        ],
    )
