from __future__ import annotations

from bnhmm.cli import build_parser


def test_boundary_flow_cli_exposes_only_current_model_parameters() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "boundary-flow-fk-walk-forward",
            "--duckdb-path",
            "/data/semiconductor.duckdb",
            "--symbols",
            "000001,000002,000003",
            "--dataset-id",
            "semiconductor",
            "--start",
            "2024-01-01",
            "--end",
            "2025-01-01",
            "--output-dir",
            "/tmp/bfk",
        ]
    )

    assert args.branch_count == 4
    assert args.topology_lookback_sessions == 60
    assert args.viscosity_window == 60
    assert args.minimum_train_sessions == 120
    assert args.neutral_sigma == 0.35
    assert args.fk_paths == 10_000
    assert not hasattr(args, "epochs")
    assert not hasattr(args, "layers")
