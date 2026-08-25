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
    policy_ablations: dict[str, dict[str, Any]] | None = None,
) -> None:
    lines = [
        "# 两支流 FK 样本外报告",
        "",
        "模型把板块股票按过去 60 个交易日的分钟同步性分成两组，并在每月内固定。",
        "两组分别估计漂移和有效势；预测多时买入整组，预测平时保持，预测空时清仓。",
        "全部实际持股统一等权，收益来自这些股票的下一交易日真实收益。",
        "",
        "| 模型 | 状态 | Brier | Log Loss | ECE | 样本数 | 开仓率 | 换手 | 净收益 | 最大回撤 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in stages:
        metrics = stage.get("metrics") or {}
        lines.append(
            "| {model} | {status} | {brier} | {log_loss} | {ece} | {count} | {open_rate} | {turnover} | {net_return} | {drawdown} |".format(
                model=stage["model_name"],
                status=stage["status"],
                brier=_number(metrics.get("brier_score")),
                log_loss=_number(metrics.get("log_loss")),
                ece=_number(metrics.get("ece")),
                count=metrics.get("observations", "-"),
                open_rate=_percent(metrics.get("open_rate")),
                turnover=_percent(metrics.get("average_turnover")),
                net_return=_percent(metrics.get("net_total_return")),
                drawdown=_percent(metrics.get("maximum_drawdown")),
            )
        )
    if policy_ablations:
        labels = {
            "posterior_signal": "近期后验信号",
            "legacy_attenuated": "旧衰减漂移",
        }
        lines.extend(
            (
                "",
                "## 信号消融",
                "",
                "两种方法使用同一组预测、月份和交易成本。收益平区间只用于评估，不控制近期后验信号的交易方向。",
                "",
                "| 方法 | 空 / 平 / 多 | Balanced Accuracy | Macro F1 | 开仓率 | 净收益 | 最大回撤 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for name, metrics in policy_ablations.items():
            counts = metrics.get("direction_counts") or (0, 0, 0)
            lines.append(
                "| {name} | {counts} | {balanced} | {macro_f1} | {open_rate} | {net_return} | {drawdown} |".format(
                    name=labels.get(name, name),
                    counts=" / ".join(str(int(value)) for value in counts),
                    balanced=_percent(metrics.get("balanced_accuracy")),
                    macro_f1=_percent(metrics.get("macro_f1")),
                    open_rate=_percent(metrics.get("open_rate")),
                    net_return=_percent(metrics.get("net_total_return")),
                    drawdown=_percent(metrics.get("maximum_drawdown")),
                )
            )
    lines.extend(
        (
            "",
            "该报告只说明样本外结果，不代表策略已经具有稳定交易价值。",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _number(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def _percent(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100.0:.2f}%"
