from __future__ import annotations

from bnhmm.boundary_fk.reporting import write_ablation_report


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

    write_ablation_report(
        path,
        [{"model_name": "TwoBranchFK", "status": "completed", "metrics": metrics}],
    )

    report = path.read_text(encoding="utf-8")
    assert "纯路径效率" in report
    assert "多 50% / 平 25% / 空 0%（每条支流）" in report
    assert "平均仓位" in report
    assert "62.50%" in report
    assert "10.00%" in report
    assert "日内物理特征消融" not in report
    assert "旧四项" not in report
