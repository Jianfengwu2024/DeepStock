"""All-stock conditional and rolling validation of holographic image clusters."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from analyze_visual_clusters import (
    balanced_accuracy,
    center_images_on_density,
    chi_square,
    fit_pca,
    image_features,
    kmeans,
    load_observations,
    standardize_channels,
    stratified_permutation_p,
    assign,
)
from bnhmm.visual_dataset import VisualDatasetConfig


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def fit_logistic(
    features: np.ndarray,
    outcomes: np.ndarray,
    *,
    ridge: float = 1.0,
    iterations: int = 60,
) -> np.ndarray:
    coefficients = np.zeros(features.shape[1], dtype=np.float64)
    penalty = np.ones(features.shape[1], dtype=np.float64)
    penalty[0] = 0.0
    for _ in range(iterations):
        probability = sigmoid(features @ coefficients)
        weight = np.maximum(probability * (1.0 - probability), 1e-5)
        gradient = features.T @ (probability - outcomes) + ridge * penalty * coefficients
        hessian = (features.T * weight) @ features
        hessian.flat[:: len(coefficients) + 1] += ridge * penalty + 1e-8
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return coefficients


def one_hot(values: np.ndarray, categories: int, *, drop_first: bool = True) -> np.ndarray:
    result = np.eye(categories, dtype=np.float64)[values]
    return result[:, 1:] if drop_first else result


def baseline_features(
    symbols: np.ndarray,
    current_returns: np.ndarray,
    symbol_order: list[str],
    return_thresholds: np.ndarray,
    return_mean: float,
    return_scale: float,
) -> np.ndarray:
    symbol_index = {symbol: index for index, symbol in enumerate(symbol_order)}
    symbol_values = np.asarray([symbol_index[item] for item in symbols], dtype=np.int64)
    bins = np.digitize(current_returns, return_thresholds).astype(np.int64)
    standardized = (current_returns - return_mean) / max(return_scale, 1e-8)
    return np.column_stack(
        (
            np.ones(len(symbols)),
            one_hot(symbol_values, len(symbol_order)),
            one_hot(bins, len(return_thresholds) + 1),
            standardized,
            np.abs(standardized),
            np.square(np.clip(standardized, -5, 5)),
        )
    )


def cluster_features(baseline: np.ndarray, labels: np.ndarray, clusters: int) -> np.ndarray:
    return np.column_stack((baseline, one_hot(labels, clusters)))


def log_loss(outcomes: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return float(-np.mean(outcomes * np.log(clipped) + (1 - outcomes) * np.log(1 - clipped)))


def metric_summary(outcomes: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(np.int64)
    return {
        "log_loss": log_loss(outcomes, probability),
        "brier": float(np.mean(np.square(probability - outcomes))),
        "accuracy": float(np.mean(predicted == outcomes)),
        "balanced_accuracy": balanced_accuracy(outcomes, predicted),
    }


def paired_day_bootstrap(
    sessions: np.ndarray,
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    unique = np.unique(sessions)
    daily = np.asarray(
        [float(differences[sessions == session].mean()) for session in unique]
    )
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = rng.choice(daily, len(daily), replace=True).mean()
    return {
        "mean_difference": float(differences.mean()),
        "daily_mean_difference": float(daily.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "one_sided_p_improvement": float((np.sum(means >= 0.0) + 1) / (samples + 1)),
        "trading_days": len(unique),
    }


def fit_cluster_labels(
    images: np.ndarray,
    fit_mask: np.ndarray,
    args: argparse.Namespace,
    seed_offset: int,
) -> tuple[np.ndarray, dict[str, float]]:
    flat, _ = standardize_channels(images[fit_mask], images)
    train_scores, scores, explained = fit_pca(
        flat[fit_mask], flat, args.pca_components, args.device, args.seed + seed_offset
    )
    centers, _, inertia = kmeans(
        train_scores,
        args.clusters,
        restarts=args.restarts,
        seed=args.seed + seed_offset + 10,
        device=args.device,
    )
    return assign(scores, centers), {
        "pca_explained_variance": explained,
        "train_inertia": inertia,
    }


def rolling_compare(
    name: str,
    labels: np.ndarray,
    baseline: np.ndarray,
    sessions: np.ndarray,
    next_sessions: np.ndarray,
    outcomes: np.ndarray,
    *,
    clusters: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    quarters = (
        ("2025Q1", "2025-01-01", "2025-04-01"),
        ("2025Q2", "2025-04-01", "2025-07-01"),
        ("2025Q3", "2025-07-01", "2025-10-01"),
        ("2025Q4", "2025-10-01", "2026-01-01"),
    )
    base_probability = np.full(len(outcomes), np.nan, dtype=np.float64)
    full_probability = np.full(len(outcomes), np.nan, dtype=np.float64)
    folds: list[dict[str, Any]] = []
    full = cluster_features(baseline, labels, clusters)
    for fold, start, end in quarters:
        train = next_sessions < start
        test = (sessions >= start) & (sessions < end) & (next_sessions < end)
        base_coef = fit_logistic(baseline[train], outcomes[train])
        full_coef = fit_logistic(full[train], outcomes[train])
        base_probability[test] = sigmoid(baseline[test] @ base_coef)
        full_probability[test] = sigmoid(full[test] @ full_coef)
        folds.append(
            {
                "fold": fold,
                "train": int(train.sum()),
                "test": int(test.sum()),
                "baseline": metric_summary(outcomes[test], base_probability[test]),
                "with_cluster": metric_summary(outcomes[test], full_probability[test]),
            }
        )
    valid = np.isfinite(base_probability) & np.isfinite(full_probability)
    base_metrics = metric_summary(outcomes[valid], base_probability[valid])
    full_metrics = metric_summary(outcomes[valid], full_probability[valid])
    base_loss = -(
        outcomes[valid] * np.log(np.clip(base_probability[valid], 1e-7, 1 - 1e-7))
        + (1 - outcomes[valid])
        * np.log(np.clip(1 - base_probability[valid], 1e-7, 1 - 1e-7))
    )
    full_loss = -(
        outcomes[valid] * np.log(np.clip(full_probability[valid], 1e-7, 1 - 1e-7))
        + (1 - outcomes[valid])
        * np.log(np.clip(1 - full_probability[valid], 1e-7, 1 - 1e-7))
    )
    return {
        "variant": name,
        "samples": int(valid.sum()),
        "baseline": base_metrics,
        "with_cluster": full_metrics,
        "log_loss_delta": full_metrics["log_loss"] - base_metrics["log_loss"],
        "brier_delta": full_metrics["brier"] - base_metrics["brier"],
        "log_loss_paired_day_bootstrap": paired_day_bootstrap(
            sessions[valid], full_loss - base_loss,
            samples=bootstrap_samples, seed=seed
        ),
        "folds": folds,
    }


def conditional_groups(
    symbols: np.ndarray,
    sessions: np.ndarray,
    current_returns: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bins = np.digitize(current_returns, thresholds)
    symbol_return = np.asarray(
        [f"{symbol}:{bucket}" for symbol, bucket in zip(symbols, bins, strict=True)]
    )
    coarse_thresholds = thresholds[1::2]
    coarse = np.digitize(current_returns, coarse_thresholds)
    date_return = np.asarray(
        [f"{session}:{bucket}" for session, bucket in zip(sessions, coarse, strict=True)]
    )
    return symbol_return, date_return


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(Path(args.visual_root, "index.sqlite3")) as connection:
        symbols = [row[0] for row in connection.execute("SELECT DISTINCT symbol FROM images ORDER BY symbol")]
    config = VisualDatasetConfig(start="2024-01-01", end="2026-01-01", volume_bins=21)
    observations = load_observations(
        args.duckdb_path, output_dir, symbols, config
    )
    images = np.stack([item.image for item in observations])
    sessions = np.asarray([item.session for item in observations])
    next_sessions = np.asarray([item.next_session for item in observations])
    symbol_values = np.asarray([item.symbol for item in observations])
    current_returns = np.asarray([item.current_return for item in observations])
    next_returns = np.asarray([item.next_return for item in observations])
    outcomes = (next_returns > 0).astype(np.int64)
    fit_mask = np.char.startswith(sessions, "2024-")
    test_mask = np.char.startswith(sessions, "2025-") & np.char.startswith(next_sessions, "2025-")

    return_thresholds = np.quantile(
        current_returns[fit_mask], np.linspace(0, 1, args.return_bins + 1)[1:-1]
    )
    baseline = baseline_features(
        symbol_values,
        current_returns,
        symbols,
        return_thresholds,
        float(current_returns[fit_mask].mean()),
        float(current_returns[fit_mask].std()),
    )
    centered = center_images_on_density(images)
    variants: dict[str, tuple[np.ndarray, dict[str, float]]] = {}
    variants["position-centered-density-and-vectors"] = fit_cluster_labels(
        centered, fit_mask, args, 100
    )
    variants["position-centered-density-only"] = fit_cluster_labels(
        centered[:, :1], fit_mask, args, 200
    )
    variants["position-centered-vectors-only"] = fit_cluster_labels(
        centered[:, 1:], fit_mask, args, 300
    )

    comparisons = [
        rolling_compare(
            name,
            labels,
            baseline,
            sessions,
            next_sessions,
            outcomes,
            clusters=args.clusters,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + index * 100,
        )
        for index, (name, (labels, _)) in enumerate(variants.items(), start=1)
    ]

    primary_labels = variants["position-centered-density-and-vectors"][0]
    symbol_groups, date_groups = conditional_groups(
        symbol_values[test_mask], sessions[test_mask], current_returns[test_mask], return_thresholds
    )
    _, symbol_p = stratified_permutation_p(
        primary_labels[test_mask], outcomes[test_mask], symbol_groups, args.clusters,
        chi_square, permutations=args.permutations, seed=args.seed + 900
    )
    _, date_p = stratified_permutation_p(
        primary_labels[test_mask], outcomes[test_mask], date_groups, args.clusters,
        chi_square, permutations=args.permutations, seed=args.seed + 901
    )

    cluster_rows = []
    for cluster in range(args.clusters):
        selected = test_mask & (primary_labels == cluster)
        cluster_rows.append(
            {
                "cluster": cluster + 1,
                "n": int(selected.sum()),
                "up_rate": float(outcomes[selected].mean()),
                "mean_next_return": float(next_returns[selected].mean()),
                "mean_current_return": float(current_returns[selected].mean()),
            }
        )
    result = {
        "symbols": len(symbols),
        "samples": {
            "all": len(observations),
            "fit_2024": int(fit_mask.sum()),
            "test_2025": int(test_mask.sum()),
        },
        "design": {
            "cluster_fit": "2024 only",
            "rolling_test": "2025 quarterly expanding window",
            "baseline": "stock fixed effects + current-return deciles + signed/absolute/squared current return",
            "clusters": args.clusters,
            "return_bins": args.return_bins,
        },
        "representations": {name: metadata for name, (_, metadata) in variants.items()},
        "conditional_primary_association": {
            "symbol_x_current_return_bin_permutation_p": symbol_p,
            "date_x_current_return_coarse_bin_permutation_p": date_p,
        },
        "clusters_2025": cluster_rows,
        "rolling_comparisons": comparisons,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "all-stock-conditional-report.md").write_text(
        markdown_report(result), encoding="utf-8"
    )
    return result


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# All-stock conditional holographic cluster report",
        "",
        f"- Symbols: {result['symbols']}",
        f"- 2024 cluster-fit images: {result['samples']['fit_2024']}",
        f"- 2025 rolling-test images: {result['samples']['test_2025']}",
        "- Baseline: stock identity plus nonlinear current-day return controls",
        "",
        "## Conditional association",
        "",
        f"- Symbol x current-return-bin permutation p: {result['conditional_primary_association']['symbol_x_current_return_bin_permutation_p']:.4f}",
        f"- Date x current-return-bin permutation p: {result['conditional_primary_association']['date_x_current_return_coarse_bin_permutation_p']:.4f}",
        "",
        "## Rolling incremental prediction",
        "",
        "| Representation | Baseline log loss | Full log loss | Delta | 95% daily CI | Improvement p | Baseline BA | Full BA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["rolling_comparisons"]:
        bootstrap = item["log_loss_paired_day_bootstrap"]
        lines.append(
            f"| {item['variant']} | {item['baseline']['log_loss']:.6f} | "
            f"{item['with_cluster']['log_loss']:.6f} | {item['log_loss_delta']:+.6f} | "
            f"{bootstrap['ci95_low']:+.6f} to {bootstrap['ci95_high']:+.6f} | "
            f"{bootstrap['one_sided_p_improvement']:.4f} | "
            f"{item['baseline']['balanced_accuracy']:.2%} | "
            f"{item['with_cluster']['balanced_accuracy']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Negative log-loss delta means the cluster adds predictive information beyond the current-return baseline.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb-path", required=True)
    parser.add_argument("--visual-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--return-bins", type=int, default=10)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=29)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    print(json.dumps(analyze(arguments), ensure_ascii=False, indent=2))
