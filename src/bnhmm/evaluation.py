"""Out-of-sample posterior evaluation and a separate position policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .config import REGIME_COUNT
from .data.dataset import ProfileObservations, ProfileSequence
from .model import BNHMMNetwork
from .training import collate_profile_sequences


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    symbol: str
    timeframe_id: int
    signal_time: np.datetime64
    execution_time: np.datetime64
    target_time: np.datetime64
    target: int
    forward_return: float
    probability_mean: tuple[float, ...]
    aleatoric_variance: tuple[float, ...]
    epistemic_variance: tuple[float, ...]
    expected_score: float
    expected_return: float
    return_variance: float
    aleatoric_return_variance: float
    epistemic_return_variance: float
    context_length: int


@dataclass(frozen=True, slots=True)
class DecisionPolicyConfig:
    uncertainty_aversion: float = 2.0
    transaction_cost_bps: float = 10.0

    def __post_init__(self) -> None:
        if self.uncertainty_aversion < 0 or self.transaction_cost_bps < 0:
            raise ValueError("policy costs and uncertainty aversion cannot be negative")


class DecisionPolicy:
    """Apply a stateful long-only policy to the three-state posterior."""

    def __init__(self, config: DecisionPolicyConfig | None = None) -> None:
        self.config = config or DecisionPolicyConfig()

    def target_position(
        self, prediction: PredictionRecord, previous_position: float = 0.0
    ) -> float:
        total_variance = np.asarray(prediction.aleatoric_variance) + np.asarray(
            prediction.epistemic_variance
        )
        uncertainty = float(np.sqrt(np.maximum(total_variance.sum(), 0.0)))
        confidence = float(
            np.clip(1.0 - self.config.uncertainty_aversion * uncertainty, 0.0, 1.0)
        )
        predicted_state = int(np.argmax(prediction.probability_mean))
        if predicted_state == 2:  # Bull: open or add, never reduce an existing long.
            return max(float(np.clip(previous_position, 0.0, 1.0)), confidence)
        if predicted_state == 1:  # Flat: retain the existing position.
            return float(np.clip(previous_position, 0.0, 1.0))
        return 0.0  # Bear: close; it never opens a short position.


def infer_sequences(
    model: BNHMMNetwork,
    sequences: Sequence[ProfileSequence],
    *,
    device: str | torch.device,
    mc_samples: int = 16,
) -> list[PredictionRecord]:
    """Predict one next-session return from each fixed recent-state window."""
    selected: dict[tuple[str, int, int], PredictionRecord] = {}
    for sequence in sequences:
        batch = collate_profile_sequences([sequence])
        posterior = model.predict_posterior(
            batch["profiles"].to(device),
            batch["timeframe_ids"].to(device),
            batch["state_features"].to(device),
            batch["bin_coordinates"].to(device),
            batch["profile_mask"].to(device),
            valid_mask=batch["valid_mask"].to(device),
            mc_samples=mc_samples,
        )
        for index in (len(sequence) - 1,):
            signal_time = sequence.signal_times[index]
            timeframe_id = int(sequence.timeframe_ids[index])
            key = (sequence.symbol, int(signal_time.astype(np.int64)), timeframe_id)
            record = PredictionRecord(
                symbol=sequence.symbol,
                timeframe_id=timeframe_id,
                signal_time=signal_time,
                execution_time=sequence.execution_times[index],
                target_time=sequence.target_times[index],
                target=int(sequence.targets[index]),
                forward_return=float(sequence.forward_returns[index]),
                probability_mean=tuple(
                    float(value) for value in posterior.probability_mean[0, index].cpu()
                ),
                aleatoric_variance=tuple(
                    float(value) for value in posterior.aleatoric_variance[0, index].cpu()
                ),
                epistemic_variance=tuple(
                    float(value) for value in posterior.epistemic_variance[0, index].cpu()
                ),
                expected_score=float(posterior.expected_score[0, index].cpu()),
                expected_return=float(
                    posterior.expected_return[0, index].cpu()
                ),
                return_variance=float(
                    posterior.return_variance[0, index].cpu()
                ),
                aleatoric_return_variance=float(
                    posterior.aleatoric_return_variance[0, index].cpu()
                ),
                epistemic_return_variance=float(
                    posterior.epistemic_return_variance[0, index].cpu()
                ),
                context_length=index + 1,
            )
            selected[key] = record
    return sorted(
        selected.values(), key=lambda item: (item.signal_time, item.symbol, item.timeframe_id)
    )


def estimate_regime_return_moments(
    observations: Iterable[ProfileObservations],
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.concatenate([item.targets for item in observations])
    returns = np.concatenate([item.forward_returns for item in observations])
    global_mean = float(returns.mean())
    global_variance = float(returns.var())
    means = np.empty(REGIME_COUNT, dtype=np.float64)
    variances = np.empty(REGIME_COUNT, dtype=np.float64)
    for state in range(REGIME_COUNT):
        values = returns[targets == state]
        means[state] = float(values.mean()) if len(values) else global_mean
        variances[state] = float(values.var()) if len(values) > 1 else global_variance
    return means, np.maximum(variances, 1e-10)


def evaluate_predictions(
    predictions: Sequence[PredictionRecord],
    regime_return_means: np.ndarray | Mapping[str, Sequence[float]],
    regime_return_variances: np.ndarray | Mapping[str, Sequence[float]],
    *,
    policy: DecisionPolicy | None = None,
) -> dict[str, object]:
    if not predictions:
        raise ValueError("predictions cannot be empty")
    probability = np.asarray([item.probability_mean for item in predictions])
    targets = np.asarray([item.target for item in predictions], dtype=np.int64)
    returns = np.asarray([item.forward_return for item in predictions])
    if probability.ndim != 2 or probability.shape[1] != REGIME_COUNT:
        raise ValueError("predictions must contain three-state probabilities")
    if np.any((targets < 0) | (targets >= REGIME_COUNT)):
        raise ValueError("prediction targets must use three-state indices")
    return_mean_rows = _moment_rows(
        regime_return_means, predictions, "regime return means"
    )
    return_variance_rows = _moment_rows(
        regime_return_variances, predictions, "regime return variances"
    )
    one_hot = np.eye(REGIME_COUNT)[targets]
    probability_residuals = one_hot - probability
    ranked_probability_score = np.square(
        probability.cumsum(axis=1)[:, :-1] - one_hot.cumsum(axis=1)[:, :-1]
    ).mean(axis=1)
    predicted_returns = np.asarray(
        [item.expected_return for item in predictions], dtype=np.float64
    )
    probability_variance = np.asarray(
        [
            np.asarray(item.aleatoric_variance)
            + np.asarray(item.epistemic_variance)
            for item in predictions
        ]
    )
    posterior_return_variance = np.asarray(
        [item.return_variance for item in predictions], dtype=np.float64
    )
    return_residuals = returns - predicted_returns
    standardized_residuals = return_residuals / np.sqrt(
        np.maximum(posterior_return_variance, 1e-10)
    )
    confidence = probability.max(axis=1)
    predicted = probability.argmax(axis=1)
    correct = predicted == targets
    confusion = np.zeros((REGIME_COUNT, REGIME_COUNT), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    class_support = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    recall = np.divide(
        np.diag(confusion),
        class_support,
        out=np.zeros(REGIME_COUNT, dtype=np.float64),
        where=class_support > 0,
    )
    precision = np.divide(
        np.diag(confusion),
        predicted_counts,
        out=np.zeros(REGIME_COUNT, dtype=np.float64),
        where=predicted_counts > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(REGIME_COUNT, dtype=np.float64),
        where=(precision + recall) > 0,
    )

    report: dict[str, object] = {
        "observation_count": len(predictions),
        "accuracy": float(correct.mean()),
        "majority_baseline": float(class_support.max() / len(targets)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "class_support": class_support.tolist(),
        "predicted_class_counts": predicted_counts.tolist(),
        "confusion_matrix": confusion.tolist(),
        "brier_score": float(np.square(probability_residuals).sum(axis=1).mean()),
        "log_loss": float(-np.log(np.clip(probability[np.arange(len(targets)), targets], 1e-12, 1.0)).mean()),
        "ranked_probability_score": float(ranked_probability_score.mean()),
        "probability_residual_mean": probability_residuals.mean(axis=0).tolist(),
        "probability_residual_rmse": np.sqrt(
            np.square(probability_residuals).mean(axis=0)
        ).tolist(),
        "return_residual": {
            "mean": float(return_residuals.mean()),
            "std": float(return_residuals.std()),
            "rmse": float(np.sqrt(np.square(return_residuals).mean())),
            "mae": float(np.abs(return_residuals).mean()),
            "correlation": float(
                np.corrcoef(predicted_returns, returns)[0, 1]
                if predicted_returns.std() > 0 and returns.std() > 0
                else 0.0
            ),
        },
        "standardized_residual": {
            "mean": float(standardized_residuals.mean()),
            "std": float(standardized_residuals.std()),
            "coverage_68": float((np.abs(standardized_residuals) <= 1.0).mean()),
            "coverage_95": float((np.abs(standardized_residuals) <= 1.96).mean()),
        },
        "calibration": _calibration(confidence, correct),
        "residual_autocorrelation": _residual_autocorrelation(
            predictions, return_residuals
        ),
        "posterior_predictive_check": {
            "observed_return_mean": float(returns.mean()),
            "predicted_return_mean": float(predicted_returns.mean()),
            "observed_return_std": float(returns.std()),
            "predicted_return_std": float(
                np.sqrt(
                    max(
                        np.mean(posterior_return_variance + predicted_returns**2)
                        - predicted_returns.mean() ** 2,
                        0.0,
                    )
                )
            ),
        },
    }
    baselines = _prediction_baselines(predictions, targets, returns)
    report["baselines"] = baselines
    report["validation_gate"] = _validation_gate(report, baselines)
    report["backtest"] = _backtest(predictions, policy or DecisionPolicy())
    report["series"] = {
        "signal_times": [str(item.signal_time) for item in predictions],
        "return_residuals": return_residuals.tolist(),
        "standardized_residuals": standardized_residuals.tolist(),
    }
    return report


def _prediction_baselines(
    predictions: Sequence[PredictionRecord],
    targets: np.ndarray,
    returns: np.ndarray,
) -> dict[str, object]:
    """Calculate point-in-time baselines without crossing symbol boundaries."""
    previous_indices: list[int] = []
    current_indices: list[int] = []
    latest: dict[tuple[str, int], int] = {}
    for index, item in enumerate(predictions):
        key = (item.symbol, item.timeframe_id)
        previous = latest.get(key)
        if previous is not None:
            previous_indices.append(previous)
            current_indices.append(index)
        latest[key] = index

    zero_rmse = float(np.sqrt(np.square(returns).mean()))
    zero_mae = float(np.abs(returns).mean())
    if not current_indices:
        return {
            "comparison_count": 0,
            "persistence_accuracy": None,
            "reversal_accuracy": None,
            "prior_return_correlation": None,
            "prior_return_rmse": None,
            "prior_return_mae": None,
            "zero_return_rmse": zero_rmse,
            "zero_return_mae": zero_mae,
        }

    previous_index = np.asarray(previous_indices, dtype=np.int64)
    current_index = np.asarray(current_indices, dtype=np.int64)
    previous_targets = targets[previous_index]
    current_targets = targets[current_index]
    previous_returns = returns[previous_index]
    current_returns = returns[current_index]
    correlation = (
        float(np.corrcoef(previous_returns, current_returns)[0, 1])
        if previous_returns.std() > 0 and current_returns.std() > 0
        else 0.0
    )
    return {
        "comparison_count": int(len(current_index)),
        "persistence_accuracy": float((previous_targets == current_targets).mean()),
        "reversal_accuracy": float(((2 - previous_targets) == current_targets).mean()),
        "prior_return_correlation": correlation,
        "prior_return_rmse": float(
            np.sqrt(np.square(current_returns - previous_returns).mean())
        ),
        "prior_return_mae": float(np.abs(current_returns - previous_returns).mean()),
        "zero_return_rmse": zero_rmse,
        "zero_return_mae": zero_mae,
    }


def _validation_gate(
    report: Mapping[str, object], baselines: Mapping[str, object]
) -> dict[str, object]:
    residual = report["return_residual"]
    assert isinstance(residual, Mapping)
    checks = {
        "beats_majority_accuracy": bool(
            float(report["accuracy"]) > float(report["majority_baseline"])
        ),
        "balanced_accuracy_above_random": bool(
            float(report["balanced_accuracy"]) > 1.0 / REGIME_COUNT + 0.01
        ),
        "beats_zero_return_rmse": bool(
            float(residual["rmse"]) < float(baselines["zero_return_rmse"])
        ),
        "positive_return_correlation": bool(float(residual["correlation"]) > 0.0),
        "minimum_oos_observations": bool(int(report["observation_count"]) >= 200),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "status": "passed" if all(checks.values()) else "research-only",
    }


def _moment_rows(
    values: np.ndarray | Mapping[str, Sequence[float]],
    predictions: Sequence[PredictionRecord],
    name: str,
) -> np.ndarray:
    if isinstance(values, Mapping):
        missing = sorted({item.symbol for item in predictions} - set(values))
        if missing:
            raise ValueError(f"{name} are missing symbols: {missing}")
        rows = np.asarray([values[item.symbol] for item in predictions], dtype=np.float64)
    else:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (REGIME_COUNT,):
            raise ValueError(f"{name} must contain three values")
        rows = np.broadcast_to(vector, (len(predictions), REGIME_COUNT))
    if rows.shape != (len(predictions), REGIME_COUNT):
        raise ValueError(f"{name} must contain three values per symbol")
    return rows


def save_evaluation_report(output_dir: str | Path, report: dict[str, object]) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_report(destination, report)


def records_as_json(predictions: Sequence[PredictionRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prediction in predictions:
        row = asdict(prediction)
        for key in ("signal_time", "execution_time", "target_time"):
            row[key] = str(row[key])
        rows.append(row)
    return rows


def _calibration(confidence: np.ndarray, correct: np.ndarray) -> list[dict[str, float | int]]:
    result: list[dict[str, float | int]] = []
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == 9 else confidence < edges[index + 1]
        )
        if mask.any():
            result.append(
                {
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "count": int(mask.sum()),
                    "mean_confidence": float(confidence[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                }
            )
    return result


def _residual_autocorrelation(
    predictions: Sequence[PredictionRecord], residuals: np.ndarray, max_lag: int = 20
) -> list[float]:
    by_symbol: dict[str, list[tuple[np.datetime64, float]]] = {}
    for prediction, residual in zip(predictions, residuals, strict=True):
        by_symbol.setdefault(prediction.symbol, []).append(
            (prediction.signal_time, float(residual))
        )
    result: list[float] = []
    for lag in range(1, max_lag + 1):
        values: list[float] = []
        for rows in by_symbol.values():
            series = np.asarray([value for _, value in sorted(rows)])
            if len(series) <= lag or series[:-lag].std() == 0 or series[lag:].std() == 0:
                continue
            values.append(float(np.corrcoef(series[:-lag], series[lag:])[0, 1]))
        result.append(float(np.mean(values)) if values else 0.0)
    return result


def _backtest(
    predictions: Sequence[PredictionRecord], policy: DecisionPolicy
) -> dict[str, object]:
    daily = sorted(
        (item for item in predictions if item.timeframe_id == 1),
        key=lambda item: (item.execution_time, item.symbol),
    )
    prior_position: dict[str, float] = {}
    grouped: dict[str, list[tuple[float, float]]] = {}
    positions: list[float] = []
    turnovers: list[float] = []
    cost_rate = policy.config.transaction_cost_bps / 10_000.0
    for item in daily:
        previous_position = prior_position.get(item.symbol, 0.0)
        position = policy.target_position(item, previous_position)
        turnover = abs(position - previous_position)
        net_return = position * item.forward_return - turnover * cost_rate
        positions.append(position)
        turnovers.append(turnover)
        date = str(item.execution_time.astype("datetime64[D]"))
        grouped.setdefault(date, []).append((net_return, item.forward_return))
        prior_position[item.symbol] = position
    dates = sorted(grouped)
    strategy_returns = np.asarray(
        [np.mean([row[0] for row in grouped[date]]) for date in dates]
    )
    benchmark_returns = np.asarray(
        [np.mean([row[1] for row in grouped[date]]) for date in dates]
    )
    equity = np.cumprod(1.0 + strategy_returns)
    benchmark = np.cumprod(1.0 + benchmark_returns)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0 if len(equity) else equity
    return {
        "dates": dates,
        "equity": equity.tolist(),
        "benchmark": benchmark.tolist(),
        "drawdown": drawdown.tolist(),
        "total_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "benchmark_return": float(benchmark[-1] - 1.0) if len(benchmark) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "average_abs_position": float(np.mean(np.abs(positions))) if positions else 0.0,
        "active_position_ratio": float(np.mean(np.abs(positions) >= 0.01)) if positions else 0.0,
        "average_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
    }


def _format_bimonthly_date_axis(axis: Any) -> None:
    from matplotlib import dates as mdates

    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.tick_params(axis="x", rotation=0)


def _plot_report(destination: Path, report: dict[str, object]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:  # pragma: no cover - optional report dependency
        return

    backtest = report["backtest"]
    dates = [date.fromisoformat(value) for value in backtest["dates"]]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(dates, backtest["equity"], label="Decision policy")
    axes[0].plot(dates, backtest["benchmark"], label="Benchmark")
    axes[0].set_ylabel("Net value")
    axes[0].legend()
    axes[1].fill_between(dates, backtest["drawdown"], 0.0)
    axes[1].set_ylabel("Drawdown")
    _format_bimonthly_date_axis(axes[1])
    figure.tight_layout()
    figure.savefig(destination / "equity_drawdown.png", dpi=150)
    plt.close(figure)

    series = report["series"]
    residuals = np.asarray(series["return_residuals"])
    standardized = np.asarray(series["standardized_residuals"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(residuals, linewidth=0.8)
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_title("Return residuals")
    axes[1].hist(standardized, bins=40)
    axes[1].set_title("Standardized residuals")
    figure.tight_layout()
    figure.savefig(destination / "residuals.png", dpi=150)
    plt.close(figure)

    calibration = report["calibration"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black")
    axes[0].plot(
        [row["mean_confidence"] for row in calibration],
        [row["accuracy"] for row in calibration],
        marker="o",
    )
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Calibration")
    autocorrelation = report["residual_autocorrelation"]
    axes[1].bar(np.arange(1, len(autocorrelation) + 1), autocorrelation)
    axes[1].set_xlabel("Lag")
    axes[1].set_title("Residual autocorrelation")
    figure.tight_layout()
    figure.savefig(destination / "calibration_acf.png", dpi=150)
    plt.close(figure)
