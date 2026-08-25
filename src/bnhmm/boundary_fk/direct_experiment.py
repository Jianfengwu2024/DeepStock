"""Real-data execution for the direct path-efficiency baseline."""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Any

from ..data.source import DuckDBBarSource
from .direct_pipeline import DirectPathPipeline, DirectPathPipelineConfig
from .experiment import build_boundary_experiment_data, _write_upstream_artifacts
from .reporting import write_json


def run_direct_path_experiment(args: Any) -> None:
    started = time.perf_counter()
    symbols = tuple(item.strip() for item in args.symbols.split(",") if item.strip())
    if len(symbols) < 4:
        raise ValueError("PathEfficiency requires at least four symbols")
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end <= start:
        raise ValueError("PathEfficiency end must follow start")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data, context = build_boundary_experiment_data(
        DuckDBBarSource(args.duckdb_path),
        symbols,
        start,
        end,
        topology_lookback_sessions=args.topology_lookback_sessions,
    )
    _write_upstream_artifacts(output, context)
    result = DirectPathPipeline(
        DirectPathPipelineConfig(
            transaction_cost_bps=args.transaction_cost_bps,
        )
    ).run(data, output)
    summary = {
        "strategy": "path-efficiency-direct",
        "display_name": "路径效率基准",
        "dataset_id": args.dataset_id,
        "symbols": len(symbols),
        "elapsed_seconds": time.perf_counter() - started,
        "model_names": [result.model_name],
        "stages": [
            {
                "model_name": result.model_name,
                "status": "completed",
                "failure_reason": "",
            }
        ],
    }
    write_json(output / "run-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
