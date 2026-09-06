"""Human-readable Chinese reporting for percolation experiments."""

from __future__ import annotations

from .types import PercolationMetrics


def render_chinese_report(
    *,
    dataset_id: str,
    observations: int,
    metrics: PercolationMetrics,
    baseline_accuracy: dict[str, float],
    smoke_only: bool,
) -> str:
    lines = [
        "# 渗流分枝策略实验报告",
        "",
        f"- 数据集：`{dataset_id}`",
        f"- 样本外预测：{observations} 个交易日",
        f"- Balanced Accuracy：{metrics.balanced_accuracy:.2%}",
        f"- Macro F1：{metrics.macro_f1:.2%}",
        f"- 样本外净收益：{metrics.net_total_return:.2%}",
        f"- 最大回撤：{metrics.maximum_drawdown:.2%}",
        f"- 平均仓位：{metrics.average_exposure:.2%}",
        "",
        "## 这次实验在做什么",
        "",
        "模型把分钟级显著上涨和下跌看作两套独立传播过程。它只允许事件在随后1至3分钟传到其他股票，同分钟共振算外部冲击，不算股票之间互相传播。模型预测的是下一交易日多头、空头、未贯通或多空冲突四种板块状态。",
        "",
        "## 基线对照",
        "",
        "| 基线 | 相态命中率 |",
        "|---|---:|",
    ]
    labels = {
        "majority": "多数相",
        "persistence": "今日相态延续",
        "transition": "历史转移矩阵",
        "simple_reproduction": "只看再生数",
    }
    for name, value in baseline_accuracy.items():
        lines.append(f"| {labels.get(name, name)} | {value:.2%} |")
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "本报告先验证实现、因果顺序和物理过程是否正确。只有相态预测同时超过多数相和状态延续基线，才值得继续评价交易收益。",
        ]
    )
    if smoke_only:
        lines.extend(
            [
                "",
                "> 本次使用了降低后的随机置换次数或后验路径数，只是功能冒烟测试，不能据此判断策略是否有效。",
            ]
        )
    return "\n".join(lines) + "\n"
