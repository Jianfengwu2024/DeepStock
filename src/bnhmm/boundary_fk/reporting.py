"""Chinese artifacts for BoundaryFlowFK walk-forward experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_ablation_report(path: Path, stages: list[dict[str, Any]]) -> None:
    lines = [
        "# BoundaryFlowFK 固定五级消融报告",
        "",
        "五个模型使用相同样本外月份和随机种子。某一级失败后，后续级别停止。",
        "",
        "| 模型 | 状态 | Brier | Log Loss | ECE | 样本数 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for stage in stages:
        metrics = stage.get("metrics") or {}
        lines.append(
            "| {model} | {status} | {brier} | {log_loss} | {ece} | {count} |".format(
                model=stage["model_name"],
                status=stage["status"],
                brier=_number(metrics.get("brier_score")),
                log_loss=_number(metrics.get("log_loss")),
                ece=_number(metrics.get("ece")),
                count=metrics.get("observations", "-"),
            )
        )
    lines.extend(
        (
            "",
            "`passed` 只表示通过当前工程门槛，不表示策略已经具有交易价值。",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _number(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6f}"
