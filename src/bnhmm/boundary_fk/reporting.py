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
