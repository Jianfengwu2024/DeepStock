from __future__ import annotations

from bnhmm.boundary_fk.reporting import write_baseline_report, write_convection_report


def test_convection_report_explains_gate_in_plain_chinese(tmp_path) -> None:
    path = tmp_path / "convection.md"

    write_convection_report(
        path,
        [
            {
                "model_name": "PathEfficiencyFK-60D-Baseline",
                "metrics": {"net_total_return": 0.02, "maximum_drawdown": 0.03},
            },
            {
                "model_name": "PathEfficiencyFK-60D-ConvectionGate",
                "metrics": {"net_total_return": 0.03, "maximum_drawdown": 0.02},
            },
        ],
        {
            "gate_triggers": 12,
            "path_efficiency_rank_correlation": -0.4,
            "bands": {
                "low": {"count": 3, "mean_next_return": 0.01},
                "elevated": {"count": 3, "mean_next_return": 0.0},
                "high": {"count": 3, "mean_next_return": -0.01},
                "extreme": {"count": 3, "mean_next_return": -0.02},
            },
        },
    )

    report = path.read_text(encoding="utf-8")
    assert "反向流量集中度" in report
    assert "股票 × 分钟" in report
    assert "反向流量占比" not in report
    assert "只能减仓或清仓" in report
    assert "PathEfficiencyFK-60D-ConvectionGate" in report
    assert "12" in report


def test_report_describes_pure_path_position_baseline(tmp_path) -> None:
    path = tmp_path / "report.md"
    metrics = {
        "observations": 12,
        "balanced_accuracy": 0.4,
        "macro_f1": 0.35,
        "open_rate": 0.5,
        "average_exposure": 0.625,
        "average_turnover": 0.1,
        "net_total_return": 0.02,
        "maximum_drawdown": 0.01,
        "direction_counts": [2, 6, 4],
    }

    write_baseline_report(
        path,
        [{"model_name": "PathEfficiencyFK", "status": "completed", "metrics": metrics}],
    )

    report = path.read_text(encoding="utf-8")
    assert "signed_path_efficiency" in report
    assert "连续四次看平清仓" in report
    assert "任一支流看空则全部清仓" in report
    assert "平均仓位" in report
    assert "62.50%" in report
    assert "10.00%" in report
    assert "消融" not in report
    assert "旧四项" not in report
