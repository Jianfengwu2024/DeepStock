"""Chinese artifacts for TwoBranchFK walk-forward experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_ablation_report(
    path: Path,
    stages: list[dict[str, Any]],
) -> None:
    lines = [
        "# 两支流 FK 样本外报告",
        "",
        "正式漂移输入固定为纯路径效率（signed_path_efficiency），不再混入其他信号。",
        "仓位规则：多 50% / 平 25% / 空 0%（每条支流）。两条支流合计仓位为 0%、25%、50%、75% 或 100%。",
        "",
        "| 模型 | 漂移输入 | 仓位规则 | 样本数 | 平均仓位 | 开仓率 | 换手 | 净收益 | 最大回撤 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in stages:
        metrics = stage.get("metrics") or {}
        lines.append(
            "| {model} | 纯路径效率 | 多50% / 平25% / 空0% | {count} | {exposure} | {open_rate} | {turnover} | {net_return} | {drawdown} |".format(
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
            "判断策略是否有效，仍要同时看漂移方向命中、漂移秩相关、换手和回撤，不能只看一次回测收益。",
            "该报告只说明样本外结果，不代表策略已经具有稳定交易价值。",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _number(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def _percent(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100.0:.2f}%"
