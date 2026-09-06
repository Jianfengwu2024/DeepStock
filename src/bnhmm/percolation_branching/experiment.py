"""CLI adapter for PercolationBranching experiments."""

from __future__ import annotations

import json

from ..data.source import DuckDBBarSource
from .pipeline import PercolationPipeline, PercolationPipelineConfig


def run_percolation_experiment(args: object) -> object:
    symbols = tuple(
        item.strip() for item in str(getattr(args, "symbols")).split(",") if item.strip()
    )
    config = PercolationPipelineConfig(
        symbols=symbols,
        dataset_id=str(getattr(args, "dataset_id")),
        start=str(getattr(args, "start")),
        end=str(getattr(args, "end")),
        activation_threshold=float(getattr(args, "activation_threshold")),
        maximum_lag_minutes=int(getattr(args, "maximum_lag_minutes")),
        topology_lookback_sessions=int(getattr(args, "topology_lookback_sessions")),
        null_permutations=int(getattr(args, "null_permutations")),
        posterior_paths=int(getattr(args, "posterior_paths")),
        phase_probability_threshold=float(getattr(args, "phase_probability_threshold")),
        transaction_cost_bps=float(getattr(args, "transaction_cost_bps")),
        random_seed=int(getattr(args, "seed")),
        max_folds=getattr(args, "max_folds"),
    )
    result = PercolationPipeline(config).run(
        DuckDBBarSource(str(getattr(args, "duckdb_path"))),
        str(getattr(args, "output_dir")),
    )
    summary = {
        "strategy": "percolation-branching",
        "predictions": len(result.predictions),
        "balanced_accuracy": result.metrics.balanced_accuracy,
        "macro_f1": result.metrics.macro_f1,
        "net_total_return": result.metrics.net_total_return,
        "output_dir": str(getattr(args, "output_dir")),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return result
