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

    assert args.topology_lookback_sessions == 60
    assert args.cache_root.endswith("path-efficiency-data")
    assert args.minimum_train_sessions == 120
    assert args.neutral_sigma == 0.35
    assert args.fk_paths == 10_000
    assert args.drift_half_life_sessions == 60.0
    assert args.minimum_direction_probability == 0.60
    assert args.minimum_signal_to_noise == 0.10
    assert args.minimum_survival_probability == 0.35
    for obsolete in (
        "branch_count",
        "viscosity_window",
        "minimum_raw_bullish_probability",
        "minimum_survival_bullish_probability",
        "minimum_trend_survival_probability",
        "maximum_bullish_probability_std",
        "maximum_thickness_risk_quantile",
        "top_k",
    ):
        assert not hasattr(args, obsolete)
    assert not hasattr(args, "epochs")
    assert not hasattr(args, "layers")


def test_boundary_flow_cli_accepts_120_day_window() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "boundary-flow-fk-walk-forward",
            "--duckdb-path", "/data/semiconductor.duckdb",
            "--symbols", "000001,000002,000003",
            "--dataset-id", "semiconductor",
            "--start", "2024-01-01",
            "--end", "2025-01-01",
            "--output-dir", "/tmp/bfk-120",
            "--topology-lookback-sessions", "120",
        ]
    )

    assert args.topology_lookback_sessions == 120


def test_boundary_data_build_cli_constructs_all_standard_windows() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "boundary-data-build",
            "--duckdb-path",
            "/data/semiconductor.duckdb",
            "--symbols",
            "000001,000002,000003,000004",
            "--dataset-id",
            "semiconductor",
            "--start",
            "2024-01-01",
            "--end",
            "2025-12-31",
        ]
    )

    assert args.lookbacks == "10,20,30,40,60,90,120"
    assert args.cache_root.endswith("path-efficiency-data")
