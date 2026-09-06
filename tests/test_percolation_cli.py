from __future__ import annotations

from bnhmm.cli import build_parser


def test_cli_exposes_fixed_physical_defaults() -> None:
    args = build_parser().parse_args(
        [
            "percolation-branching-walk-forward",
            "--duckdb-path",
            "market.duckdb",
            "--symbols",
            "A,B,C,D",
            "--dataset-id",
            "synthetic",
            "--start",
            "2024-01-01",
            "--end",
            "2025-01-01",
            "--output-dir",
            "/tmp/percolation",
        ]
    )

    assert args.activation_threshold == 1.5
    assert args.maximum_lag_minutes == 3
    assert args.topology_lookback_sessions == 60
    assert args.phase_probability_threshold == 0.60
    assert args.null_permutations == 500
    assert args.posterior_paths == 10_000
    assert args.transaction_cost_bps == 5.0

