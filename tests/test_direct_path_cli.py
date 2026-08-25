from __future__ import annotations

from bnhmm.cli import build_parser


def test_direct_path_cli_only_exposes_data_topology_and_cost_parameters() -> None:
    args = build_parser().parse_args(
        [
            "path-efficiency-walk-forward",
            "--duckdb-path",
            "/data/semiconductor.duckdb",
            "--symbols",
            "000001,000002,000003,000004",
            "--dataset-id",
            "semiconductor",
            "--start",
            "2024-01-01",
            "--end",
            "2025-01-01",
            "--output-dir",
            "/tmp/path-efficiency",
        ]
    )

    assert args.topology_lookback_sessions == 60
    assert args.transaction_cost_bps == 10.0
    for excluded in (
        "minimum_train_sessions",
        "neutral_sigma",
        "fk_paths",
        "drift_half_life_sessions",
        "minimum_direction_probability",
        "minimum_signal_to_noise",
        "minimum_survival_probability",
        "seed",
        "epochs",
    ):
        assert not hasattr(args, excluded)
