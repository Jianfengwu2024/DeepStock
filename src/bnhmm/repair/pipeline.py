"""Report orchestration for sequential Holo repair experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .variants import RepairVariantCatalog


_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "brier_score",
    "log_loss",
)


def _number(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    return float(value) if value is not None else 0.0


def _percent(value: object) -> str:
    return f"{float(value or 0.0) * 100:.2f}%"


class RepairExperimentPipeline:
    """Collect immutable stage artifacts and render a direct-predecessor report."""

    @staticmethod
    def load_stage(output_dir: str | Path) -> dict[str, Any]:
        path = Path(output_dir) / "run-summary.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        result["artifact_dir"] = str(path.parent)
        return result

    @classmethod
    def write_stage_report(
        cls, output_dir: str | Path, output_path: str | Path | None = None
    ) -> Path:
        row = cls.load_stage(output_dir)
        model_name = row.get("repair_variant") or "unknown"
        variant = RepairVariantCatalog.resolve(model_name)
        baselines = row.get("baselines") or {}
        backtest = row.get("backtest") or {}
        return_residual = row.get("return_residual") or {}
        model = row.get("model_config") or {}
        trainer = row.get("trainer_config") or {}
        walk_forward = row.get("walk_forward_config") or {}
        folds = row.get("folds") or []
        lines = [
            f"# {model_name} Stage Report",
            "",
            f"- Repair stage: {variant.stage}",
            f"- Causal change: {variant.description}",
            f"- Dataset: {row.get('dataset_id') or 'unspecified'}",
            f"- Symbols: {len(row.get('symbols') or [])}",
            f"- Completed folds: {len(folds)}",
            f"- OOS predictions: {int(row.get('oos_predictions') or 0):,}",
            "",
            "## Fixed Contract",
            "",
            f"- Control mode: `{model.get('control_mode', variant.control_mode)}`",
            f"- Sequence length: {model.get('max_sequence_length', 'unknown')}",
            f"- Epochs: {trainer.get('epochs', 'unknown')}",
            f"- Batch size: {trainer.get('batch_size', 'unknown')}",
            f"- Seed: {trainer.get('seed', 'unknown')}",
            f"- Minimum train sessions: {walk_forward.get('minimum_train_sessions', 'unknown')}",
            "",
            "## OOS Results",
            "",
            "| Metric | Model | Reference | Delta |",
            "|---|---:|---:|---:|",
            (
                f"| Accuracy | {_percent(row.get('accuracy'))} | "
                f"{_percent(row.get('majority_baseline'))} | "
                f"{_percent(_number(row, 'accuracy') - _number(row, 'majority_baseline'))} |"
            ),
            f"| Balanced accuracy | {_percent(row.get('balanced_accuracy'))} | 25.00% | {_percent(_number(row, 'balanced_accuracy') - 0.25)} |",
            f"| Macro F1 | {_number(row, 'macro_f1'):.4f} | - | - |",
            f"| Brier score | {_number(row, 'brier_score'):.4f} | {float(baselines.get('class_prior_brier') or 0.0):.4f} | {_number(row, 'brier_score') - float(baselines.get('class_prior_brier') or 0.0):.4f} |",
            f"| Log loss | {_number(row, 'log_loss'):.4f} | {float(baselines.get('class_prior_log_loss') or 0.0):.4f} | {_number(row, 'log_loss') - float(baselines.get('class_prior_log_loss') or 0.0):.4f} |",
            f"| Return correlation | {float(return_residual.get('correlation') or 0.0):.4f} | 0.0000 | {float(return_residual.get('correlation') or 0.0):.4f} |",
            f"| Strategy return | {_percent(backtest.get('total_return'))} | {_percent(backtest.get('benchmark_return'))} | {_percent(float(backtest.get('total_return') or 0.0) - float(backtest.get('benchmark_return') or 0.0))} |",
            "",
            "## Diagnostics",
            "",
            f"- Class support: `{row.get('class_support')}`",
            f"- Predicted class counts: `{row.get('predicted_class_counts')}`",
            f"- Validation gate: `{row.get('validation_gate')}`",
            "",
            "This report is descriptive. Promotion requires comparison against the other HoloProbe controls under the same folds and budget.",
        ]
        destination = Path(output_path) if output_path else Path(output_dir) / "repair-stage-report.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def write_report(
        cls, stage_dirs: Iterable[str | Path], output_path: str | Path
    ) -> Path:
        rows = [cls.load_stage(path) for path in stage_dirs]
        order = {name: index for index, name in enumerate(RepairVariantCatalog.names())}
        rows.sort(key=lambda row: order.get(row.get("repair_variant", ""), 10_000))
        lines = ["# HoloBNHMM Sequential Repair Report", ""]
        lines.append("| Model | OOS | Accuracy | vs Majority | Balanced | Macro F1 | Brier | Log Loss |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                "| {name} | {oos:,} | {accuracy:.4f} | {accuracy_delta:+.4f} | {balanced_accuracy:.4f} | "
                "{macro_f1:.4f} | {brier_score:.4f} | {log_loss:.4f} |".format(
                    name=row.get("repair_variant", "unknown"),
                    oos=row.get("oos_predictions", 0),
                    accuracy_delta=_number(row, "accuracy")
                    - _number(row, "majority_baseline"),
                    **{metric: float(row.get(metric, 0.0)) for metric in _METRICS},
                )
            )
        lines.extend(
            (
                "",
                "## Interpretation Rule",
                "",
                "The image branch has incremental signal only when HoloProbe-Image consistently beats HoloProbe-Prior, HoloProbe-Metadata, and HoloProbe-Shuffled on balanced accuracy and probabilistic loss under identical folds.",
            )
        )
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return destination
