from __future__ import annotations

from bnhmm.boundary_fk.reporting import write_ablation_report


def test_report_compares_posterior_and_legacy_policy(tmp_path) -> None:
    path = tmp_path / "report.md"
    metrics = {
        "observations": 12,
        "balanced_accuracy": 0.4,
        "macro_f1": 0.35,
        "open_rate": 0.5,
        "average_turnover": 0.1,
        "net_total_return": 0.02,
        "maximum_drawdown": 0.01,
        "direction_counts": [2, 6, 4],
    }

    write_ablation_report(
        path,
        [],
        {
            "posterior_signal": metrics,
            "path_efficiency_only": metrics,
            "path_efficiency": metrics,
            "legacy_attenuated": {**metrics, "open_rate": 0.0},
        },
    )

    report = path.read_text(encoding="utf-8")
    assert "近期后验信号" in report
    assert "纯路径效率" in report
    assert "旧四项 + 路径效率" in report
    assert "| 开仓率 | 换手 |" in report
    assert "10.00%" in report
    assert "旧衰减漂移" in report
    assert "空 / 平 / 多" in report
