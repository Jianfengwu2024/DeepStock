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


def write_convection_report(
    path: Path,
    stages: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> None:
    lines = [
        f"本次使用 {int(diagnostics.get('convection_lookback_sessions', 1))} 日反向流量集中度。",
        "持续规则：首次出现不动作，连续 2 日减半，连续 3 日清仓；中断后重新计数。",
        "# PathEfficiencyFK 60 日对流门控报告",
        "",
        "方向仍然只由路径效率和 FK 漂移决定。对流使用股票 × 分钟二维平面上的反向流量集中度，只能减仓或清仓，不能反向开仓。",
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
            "## 反向流量集中度统计",
            "",
            f"门控触发 {int(diagnostics.get('gate_triggers', 0))} 次。",
            "",
            "| 集中度档位 | 样本数 | 下一日平均收益 | 趋势延续率 | 反转率 |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    labels = {
        "low": "较弱",
        "elevated": "偏强",
        "high": "很强",
        "extreme": "极强",
    }
    for name in ("low", "elevated", "high", "extreme"):
        item = (diagnostics.get("bands") or {}).get(name, {})
        lines.append(
            "| {label} | {count} | {ret} | {continuation} | {reversal} |".format(
                label=labels[name],
                count=int(item.get("count", 0)),
                ret=_percent(item.get("mean_next_return")),
                continuation=_percent(item.get("continuation_rate")),
                reversal=_percent(item.get("reversal_rate")),
            )
        )
    lines.extend(
        (
            "",
            "反向流量集中度与路径效率绝对值的 Rank 相关系数："
            f"{float(diagnostics.get('path_efficiency_rank_correlation', 0.0)):.4f}。",
            "",
            "只有真实样本外收益和回撤同时改善，才能认为对流门控有效。",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")
