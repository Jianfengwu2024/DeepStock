"""Real-data construction and CLI execution for PathEfficiencyFK."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ..data.source import DuckDBBarSource
from ..sector_flow.panel import SectorMinutePanelBuilder
from .cache import SECTOR_CACHE_SLOT, BoundaryDataCache, BoundaryDataCacheKey
from .execution import build_member_execution_constraints
from .normalization import CausalBoundaryNormalizer
from .pipeline import BoundaryFlowPipeline, BoundaryFlowPipelineConfig
from .reporting import write_json
from .sector_state import SectorDailyStateBuilder
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
        lookback_sessions=SECTOR_CACHE_SLOT,
    )
    data = BoundaryDataCache(args.cache_root).load(
        key,
        evaluation_start=start.isoformat(),
        evaluation_end=end.isoformat(),
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
            relative_strength_lookback_sessions=(
                args.relative_strength_lookback_sessions
            ),
            transaction_cost_bps=args.transaction_cost_bps,
            maximum_drawdown=getattr(
                args, "maximum_drawdown", 0.05
            ),
        )
    ).run(data, output)
    summary = {
        "strategy": "boundary-flow-fk",
        "display_name": "板块方向 FK",
        "dataset_id": args.dataset_id,
        "symbols": len(symbols),
        "data_source": "precomputed",
        "relative_strength_lookback_sessions": (
            args.relative_strength_lookback_sessions
        ),
        "maximum_drawdown": getattr(
            args, "maximum_drawdown", 0.05
        ),
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
) -> tuple[TwoBranchDataset, dict[str, Any]]:
    warmup_start = start - timedelta(days=450)
    bars = {
        symbol: source.fetch(symbol, "1m", warmup_start, end)
        for symbol in symbols
    }
    panel = SectorMinutePanelBuilder().build(bars)
    normalized = CausalBoundaryNormalizer().transform(panel)
    execution = build_member_execution_constraints(panel)
    built = SectorDailyStateBuilder().build(normalized)
    if not np.array_equal(built.states.sessions, execution.sessions):
        raise ValueError("execution constraints do not align with daily states")
    data = replace(
        built,
        member_next_hold_returns=execution.hold_returns,
        member_next_buy_returns=execution.buy_returns,
        member_next_buy_executable=execution.buy_executable,
        member_next_sell_returns=execution.sell_returns,
        member_next_sell_executable=execution.sell_executable,
        evaluation_start=np.datetime64(start),
        evaluation_end=np.datetime64(end),
    )
    return data, {}
