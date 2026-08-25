"""Chinese artifacts for the PathEfficiencyFK baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_baseline_report(
    path: Path,
    stages: list[dict[str, Any]],
) -> None:
    lines = [
        "# 路径效率 FK 样本外报告",
        "",
        "模型只使用 signed_path_efficiency 估计漂移，随后由 Feynman-Kac 计算多、平、空后验与有效势。",
        "仓位规则：看多恢复满仓；无看空时，每次看平减去总满仓的 25%，连续四次看平清仓；任一支流看空则全部清仓。",
        "",
        "| 模型 | 漂移输入 | 样本数 | 平均仓位 | 开仓率 | 换手 | 净收益 | 最大回撤 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in stages:
        metrics = stage.get("metrics") or {}
        lines.append(
            "| {model} | 路径效率 | {count} | {exposure} | {open_rate} | {turnover} | {net_return} | {drawdown} |".format(
                model=stage["model_name"],
                count=metrics.get("observations", "-"),
                exposure=_percent(metrics.get("average_exposure")),
                open_rate=_percent(metrics.get("open_rate")),
                turnover=_percent(metrics.get("average_turnover")),
                net_return=_percent(metrics.get("net_total_return")),
                drawdown=_percent(metrics.get("maximum_drawdown")),
            )
        )
    lines.extend(
        (
            "",
            "结论必须同时参考漂移方向命中、概率校准、换手和回撤，不能只看单次收益。",
            "本报告仅说明严格样本外结果，不代表策略已经具有稳定交易价值。",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _percent(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100.0:.2f}%"


def write_background_potential_report(
    path: Path,
    stages: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> None:
    lines = [
        "# PathEfficiencyFK 60 日背景外势报告",
        "",
        "路径效率是唯一漂移输入，决定趋势方向。背景外势只判断这股趋势还能否延续。",
        "",
        "外势不读取反向流量或其他市场特征，只在每个支流、每个滚动训练窗口独立估计背景外势 β0。",
        "有效势 V = softplus(β0)，趋势存活率 S = exp(-V)。它只能削弱趋势，不能制造反向信号。",
        "",
        "## 策略结果",
        "",
        "| 模型 | 成本后收益 | 最大回撤 | 平均仓位 | 平均换手 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for stage in stages:
        metrics = stage.get("metrics") or {}
        lines.append(
            "| {model} | {ret} | {drawdown} | {exposure} | {turnover} |".format(
                model=stage["model_name"],
                ret=_percent(metrics.get("net_total_return")),
                drawdown=_percent(metrics.get("maximum_drawdown")),
                exposure=_percent(metrics.get("average_exposure")),
                turnover=_percent(metrics.get("average_turnover")),
            )
        )
    lines.extend(
        (
            "",
            "## 背景外势统计",
            "",
            f"样本数：{int(diagnostics.get('observations', 0))}",
            f"平均有效势：{float(diagnostics.get('mean_effective_potential', 0.0)):.4f}",
            f"中位有效势：{float(diagnostics.get('median_effective_potential', 0.0)):.4f}",
            "平均趋势存活率："
            f"{_percent(diagnostics.get('mean_trend_survival_probability'))}",
            "最低趋势存活率："
            f"{_percent(diagnostics.get('minimum_trend_survival_probability'))}",
            "",
            "只有严格样本外收益、回撤和换手的综合结果优于纯路径效率基线，才能认为背景外势有实际价值。",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")
