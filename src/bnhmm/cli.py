"""Operational commands for data smoke checks and storage migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import LabelConfig, ProfileConfig, SequenceConfig
from .data.dataset import ProfileLabelBuilder, SequenceWindowBuilder
from .data.profile import PriceVolumeProfileBuilder
from .data.source import BrokerBarSource, DuckDBBarSource
from .storage import apply_migrations


def _state_timeframes(mode: str) -> tuple[str, ...]:
    return ("1d",) if mode == "daily" else ("60m", "1d")


def _profile_smoke(args: argparse.Namespace) -> int:
    source = BrokerBarSource(args.broker_url, timeout_seconds=args.timeout)
    bars = source.fetch(args.symbol, "1m", args.start, args.end)
    builder = PriceVolumeProfileBuilder(ProfileConfig())
    states = builder.build(bars)
    summary = {
        "symbol": args.symbol,
        "bars": len(bars),
        "first_bar": str(bars.timestamp[0]) if len(bars) else None,
        "last_bar": str(bars.timestamp[-1]) if len(bars) else None,
        "hour_states": sum(state.timeframe == "60m" for state in states),
        "day_states": sum(state.timeframe == "1d" for state in states),
        "profile_bins": len(states[0].distribution) if states else 0,
        "all_profiles_normalized": all(
            abs(float(state.distribution.sum()) - 1.0) < 1e-9 for state in states
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if states else 2


def _migrate(args: argparse.Namespace) -> int:
    applied = apply_migrations(args.dsn)
    print(json.dumps({"applied": applied}, indent=2))
    return 0


def _train(args: argparse.Namespace) -> int:
    import torch

    from .training import (
        BNHMMTrainer,
        ModelConfig,
        TrainerConfig,
        collate_profile_sequences,
        save_checkpoint,
    )

    profile_config = ProfileConfig(
        bins=args.profile_bins,
        state_timeframes=_state_timeframes(args.state_timeframe),
    )
    sequence_config = SequenceConfig(
        length=args.sequence_length,
        stride=args.sequence_stride,
        minimum_length=args.minimum_sequence_length,
    )
    source = DuckDBBarSource(args.duckdb_path)
    bars = source.fetch(args.symbol, "1m", args.start, args.end)
    states = PriceVolumeProfileBuilder(profile_config).build(bars)
    observations = ProfileLabelBuilder(
        LabelConfig(forward_sessions=args.forward_sessions)
    ).build(states)
    sessions = np.unique(observations.signal_times.astype("datetime64[D]"))
    if len(sessions) < 10:
        raise ValueError("training requires at least ten complete sessions")
    cutoff_index = int(np.clip(len(sessions) * args.train_fraction, 1, len(sessions) - 1))
    cutoff = sessions[cutoff_index]
    train_observations, validation_observations = observations.temporal_split(cutoff)
    window_builder = SequenceWindowBuilder(sequence_config)
    train_sequences = window_builder.build(train_observations)
    validation_sequences = window_builder.build(validation_observations)
    if not train_sequences or not validation_sequences:
        raise ValueError("date range is too short for the configured sequence windows")

    model_config = ModelConfig(
        profile_bins=args.profile_bins,
        d_model=args.d_model,
        motif_dim=args.motif_dim,
        heads=args.heads,
        layers=args.layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        max_sequence_length=args.sequence_length,
    )
    trainer_config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_config.build()
    trainer = BNHMMTrainer(model, trainer_config, device=device)
    result = trainer.fit(train_sequences, validation_sequences)

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "model.pt"
    metadata = {
        "symbol": args.symbol,
        "source": str(Path(args.duckdb_path).resolve()),
        "start": args.start,
        "end": args.end,
        "cutoff": str(cutoff),
        "training_cutoff": str(cutoff),
        "profile_states": len(states),
        "state_timeframe": args.state_timeframe,
        "train_observations": len(train_observations),
        "validation_observations": len(validation_observations),
    }
    save_checkpoint(
        checkpoint_path,
        model,
        model_config,
        trainer_config,
        result,
        metadata,
    )

    posterior_batch = collate_profile_sequences([validation_sequences[-1]])
    posterior = model.predict_posterior(
        posterior_batch["profiles"].to(device),
        posterior_batch["timeframe_ids"].to(device),
        valid_mask=posterior_batch["valid_mask"].to(device),
        mc_samples=args.mc_samples,
    )
    last = len(validation_sequences[-1]) - 1
    summary = {
        **metadata,
        "device": str(device),
        "bars": len(bars),
        "train_sequences": len(train_sequences),
        "validation_sequences": len(validation_sequences),
        "class_counts": np.bincount(observations.targets, minlength=5).tolist(),
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "history": list(result.history),
        "latest_signal_time": str(validation_sequences[-1].signal_times[last]),
        "latest_probability_mean": posterior.probability_mean[0, last]
        .cpu()
        .tolist(),
        "latest_aleatoric_variance": posterior.aleatoric_variance[0, last]
        .cpu()
        .tolist(),
        "latest_epistemic_variance": posterior.epistemic_variance[0, last]
        .cpu()
        .tolist(),
        "checkpoint": str(checkpoint_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _walk_forward(args: argparse.Namespace) -> int:
    import torch

    from .evaluation import (
        DecisionPolicy,
        DecisionPolicyConfig,
        evaluate_predictions,
        save_evaluation_report,
    )
    from .training import ModelConfig, TrainerConfig
    from .walk_forward import WalkForwardConfig, run_walk_forward

    source = DuckDBBarSource(args.duckdb_path)
    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    if not symbols:
        symbols = source.list_symbols(args.symbol_limit)
    observations = []
    data_summary: dict[str, dict[str, int]] = {}
    profile_config = ProfileConfig(
        bins=args.profile_bins,
        state_timeframes=_state_timeframes(args.state_timeframe),
    )
    label_config = LabelConfig(forward_sessions=args.forward_sessions)
    for symbol in symbols:
        bars = source.fetch(symbol, "1m", args.start, args.end)
        states = PriceVolumeProfileBuilder(profile_config).build(bars)
        if not states:
            continue
        labeled = ProfileLabelBuilder(label_config).build(states)
        observations.append(labeled)
        data_summary[symbol] = {
            "bars": len(bars),
            "profile_states": len(states),
            "observations": len(labeled),
        }
    if not observations:
        raise ValueError("none of the requested symbols produced observations")

    sequence_config = SequenceConfig(
        length=args.sequence_length,
        stride=args.sequence_stride,
        minimum_length=args.minimum_sequence_length,
    )
    model_config = ModelConfig(
        profile_bins=args.profile_bins,
        d_model=args.d_model,
        motif_dim=args.motif_dim,
        heads=args.heads,
        layers=args.layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        max_sequence_length=args.sequence_length,
    )
    trainer_config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    walk_config = WalkForwardConfig(
        minimum_train_sessions=args.minimum_train_sessions,
        tuning_sessions=args.tuning_sessions,
        test_sessions=args.test_sessions,
        step_sessions=args.step_sessions,
        max_folds=args.max_folds,
    )
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_walk_forward(
        observations,
        model_config,
        trainer_config,
        sequence_config,
        walk_config,
        device=device,
        output_dir=args.output_dir,
        mc_samples=args.mc_samples,
    )
    policy = DecisionPolicy(
        DecisionPolicyConfig(
            allow_short=args.allow_short,
            uncertainty_aversion=args.uncertainty_aversion,
            transaction_cost_bps=args.transaction_cost_bps,
        )
    )
    report = evaluate_predictions(
        result.predictions,
        np.asarray(result.regime_return_means),
        np.asarray(result.regime_return_variances),
        policy=policy,
    )
    save_evaluation_report(args.output_dir, report)
    summary = {
        "symbols": list(data_summary),
        "state_timeframe": args.state_timeframe,
        "data": data_summary,
        "folds": list(result.folds),
        "oos_predictions": len(result.predictions),
        "accuracy": report["accuracy"],
        "brier_score": report["brier_score"],
        "log_loss": report["log_loss"],
        "return_residual": report["return_residual"],
        "standardized_residual": report["standardized_residual"],
        "backtest": {
            key: report["backtest"][key]
            for key in (
                "total_return",
                "benchmark_return",
                "max_drawdown",
                "average_abs_position",
                "active_position_ratio",
                "average_turnover",
            )
        },
        "output_dir": args.output_dir,
    }
    Path(args.output_dir, "run-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bnhmm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("profile-smoke")
    smoke.add_argument("--broker-url", required=True)
    smoke.add_argument("--symbol", required=True)
    smoke.add_argument("--start", required=True)
    smoke.add_argument("--end", required=True)
    smoke.add_argument("--timeout", type=float, default=180.0)
    smoke.set_defaults(handler=_profile_smoke)

    migrate = subparsers.add_parser("migrate")
    migrate.add_argument("--dsn", required=True)
    migrate.set_defaults(handler=_migrate)

    train = subparsers.add_parser("train")
    train.add_argument("--duckdb-path", required=True)
    train.add_argument("--symbol", required=True)
    train.add_argument("--start", required=True)
    train.add_argument("--end", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--train-fraction", type=float, default=0.8)
    train.add_argument("--forward-sessions", type=int, default=1)
    train.add_argument("--profile-bins", type=int, default=48)
    train.add_argument(
        "--state-timeframe",
        choices=("daily", "mixed"),
        default="daily",
        help="daily builds one state from each full 1m session; mixed also builds 60m states",
    )
    train.add_argument("--sequence-length", type=int, default=128)
    train.add_argument("--sequence-stride", type=int, default=64)
    train.add_argument("--minimum-sequence-length", type=int, default=32)
    train.add_argument("--d-model", type=int, default=96)
    train.add_argument("--motif-dim", type=int, default=32)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--layers", type=int, default=3)
    train.add_argument("--feedforward-dim", type=int, default=192)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--mc-samples", type=int, default=8)
    train.add_argument("--seed", type=int, default=7)
    train.set_defaults(handler=_train)

    walk = subparsers.add_parser("walk-forward")
    walk.add_argument("--duckdb-path", required=True)
    walk.add_argument("--symbols", default="")
    walk.add_argument("--symbol-limit", type=int, default=10)
    walk.add_argument("--start", required=True)
    walk.add_argument("--end", required=True)
    walk.add_argument("--output-dir", required=True)
    walk.add_argument("--device", default="auto")
    walk.add_argument("--epochs", type=int, default=10)
    walk.add_argument("--batch-size", type=int, default=8)
    walk.add_argument("--learning-rate", type=float, default=3e-4)
    walk.add_argument("--forward-sessions", type=int, default=1)
    walk.add_argument("--minimum-train-sessions", type=int, default=252)
    walk.add_argument("--tuning-sessions", type=int, default=60)
    walk.add_argument("--test-sessions", type=int, default=60)
    walk.add_argument("--step-sessions", type=int, default=60)
    walk.add_argument("--max-folds", type=int)
    walk.add_argument("--profile-bins", type=int, default=48)
    walk.add_argument(
        "--state-timeframe",
        choices=("daily", "mixed"),
        default="daily",
        help="daily builds one state from each full 1m session; mixed also builds 60m states",
    )
    walk.add_argument("--sequence-length", type=int, default=128)
    walk.add_argument("--sequence-stride", type=int, default=64)
    walk.add_argument("--minimum-sequence-length", type=int, default=32)
    walk.add_argument("--d-model", type=int, default=96)
    walk.add_argument("--motif-dim", type=int, default=32)
    walk.add_argument("--heads", type=int, default=4)
    walk.add_argument("--layers", type=int, default=3)
    walk.add_argument("--feedforward-dim", type=int, default=192)
    walk.add_argument("--dropout", type=float, default=0.1)
    walk.add_argument("--mc-samples", type=int, default=16)
    walk.add_argument("--uncertainty-aversion", type=float, default=2.0)
    walk.add_argument("--transaction-cost-bps", type=float, default=10.0)
    walk.add_argument("--allow-short", action="store_true")
    walk.add_argument("--seed", type=int, default=7)
    walk.set_defaults(handler=_walk_forward)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

