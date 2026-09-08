"""Operational commands for data smoke checks and storage migration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import (
    REGIME_COUNT,
    REGIME_NAMES,
    REGIME_PROTOCOL,
    PROFILE_PROTOCOL,
    SEQUENCE_PROTOCOL,
    LabelConfig,
    ProfileConfig,
    SequenceConfig,
)
from .data.dataset import ProfileLabelBuilder, SequenceWindowBuilder
from .data.profile import PriceVolumeProfileBuilder
from .data.source import BrokerBarSource, DuckDBBarSource
from .storage import apply_migrations


def _state_timeframes(mode: str) -> tuple[str, ...]:
    return ("1d",) if mode == "daily" else ("60m", "1d")


def _regime_metadata(labels: LabelConfig) -> dict[str, object]:
    return {
        "version": REGIME_PROTOCOL,
        "states": list(REGIME_NAMES),
        "thresholds": list(labels.thresholds),
        "threshold_unit": "rolling_sigma",
        "standardization_lookback_sessions": labels.standardization_lookback_sessions,
        "standardization_min_observations": labels.standardization_min_observations,
        "minimum_return_scale": labels.minimum_return_scale,
    }


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
        "minimum_profile_bins": min((len(state.distribution) for state in states), default=0),
        "maximum_profile_bins": max((len(state.distribution) for state in states), default=0),
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




def _boundary_flow_walk_forward(args: argparse.Namespace) -> int:
    from .boundary_fk.experiment import run_boundary_flow_experiment

    run_boundary_flow_experiment(args)
    return 0


def _boundary_data_build(args: argparse.Namespace) -> int:
    from .boundary_fk.precompute import run_boundary_data_build

    run_boundary_data_build(args)
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
        bin_width_bps=args.bin_width_bps,
        max_profile_bins=args.max_profile_bins,
        state_timeframes=_state_timeframes(args.state_timeframe),
        relative_volume_lookback_sessions=args.normalization_lookback_sessions,
        relative_volume_min_observations=args.normalization_lookback_sessions,
        velocity_bins=args.velocity_bins,
        velocity_aggregation_minutes=args.velocity_aggregation_minutes,
        velocity_clip=args.velocity_clip,
        velocity_smoothing=args.velocity_smoothing,
    )
    sequence_config = SequenceConfig(
        length=args.sequence_length,
        stride=args.sequence_stride,
        minimum_length=args.minimum_sequence_length,
    )
    source = DuckDBBarSource(args.duckdb_path)
    bars = source.fetch(args.symbol, "1m", args.start, args.end)
    states = PriceVolumeProfileBuilder(profile_config).build(bars)
    label_config = LabelConfig(
        forward_sessions=args.forward_sessions,
        standardization_lookback_sessions=args.return_standardization_lookback,
        standardization_min_observations=args.return_standardization_min_observations,
        minimum_return_scale=args.minimum_return_scale_percent / 100.0,
        strong_threshold_sigma=args.strong_threshold_sigma,
    )
    observations = ProfileLabelBuilder(label_config).build(states)
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
        max_profile_bins=args.max_profile_bins,
        profile_channels=profile_config.profile_channels,
        latent_states=args.latent_states,
        transition_rank=args.transition_rank,
        strong_bear_threshold=label_config.strong_bear_threshold,
        direction_threshold=label_config.direction_threshold,
        strong_bull_threshold=label_config.strong_bull_threshold,
        feature_mode=args.feature_mode,
        auxiliary_gate_floor=args.auxiliary_gate_floor,
        auxiliary_gate_initial=args.auxiliary_gate_initial,
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
        kl_weight=args.kl_weight,
        ordinal_weight=args.ordinal_weight,
        symmetry_weight=args.symmetry_weight,
        pretrain_epochs=args.pretrain_epochs,
        auxiliary_warmup_epochs=args.auxiliary_warmup_epochs,
        pretrain_mask_probability=args.pretrain_mask_probability,
        sample_during_training=not args.deterministic_training,
        seed=args.seed,
    )
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_config.build(seed=args.seed)
    trainer = BNHMMTrainer(model, trainer_config, device=device)
    result = trainer.fit(train_sequences, validation_sequences)

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "model.pt"
    metadata = {
        "symbol": args.symbol,
        "model_scope": "single-symbol",
        "source": str(Path(args.duckdb_path).resolve()),
        "start": args.start,
        "end": args.end,
        "cutoff": str(cutoff),
        "training_cutoff": str(cutoff),
        "profile_states": len(states),
        "profile_protocol": PROFILE_PROTOCOL,
        "profile_channels": profile_config.profile_channels,
        "output_head": "training-prior-plus-bayesian-evidence-and-return",
        "normalization_lookback_sessions": args.normalization_lookback_sessions,
        "velocity_bins": args.velocity_bins,
        "velocity_aggregation_minutes": args.velocity_aggregation_minutes,
        "velocity_clip": args.velocity_clip,
        "velocity_smoothing": args.velocity_smoothing,
        "feature_mode": args.feature_mode,
        "regime_protocol": _regime_metadata(label_config),
        "sequence_protocol": SEQUENCE_PROTOCOL,
        "bin_width_bps": args.bin_width_bps,
        "maximum_profile_bins": max((len(state.distribution) for state in states), default=0),
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

    inference_seed = args.seed + 100_000
    torch.manual_seed(inference_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(inference_seed)
    posterior_batch = collate_profile_sequences([validation_sequences[-1]])
    posterior = model.predict_posterior(
        posterior_batch["profiles"].to(device),
        posterior_batch["timeframe_ids"].to(device),
        posterior_batch["state_features"].to(device),
        posterior_batch["bin_coordinates"].to(device),
        posterior_batch["profile_mask"].to(device),
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
        "class_counts": np.bincount(
            observations.targets, minlength=REGIME_COUNT
        ).tolist(),
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "history": list(result.history),
        "pretraining_history": list(result.pretraining_history),
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
        "latest_expected_standardized_return": float(
            posterior.expected_return[0, last].cpu()
        ),
        "latest_expected_return": float(
            validation_sequences[-1].return_scales[last]
            * float(posterior.expected_return[0, last].cpu())
        ),
        "latest_return_variance": float(
            posterior.return_variance[0, last].cpu()
            * validation_sequences[-1].return_scales[last] ** 2
        ),
        "training_class_prior": model.class_prior.detach().cpu().tolist(),
        "checkpoint": str(checkpoint_path),
        "auxiliary_gates": dict(
            zip(
                ("relative_volume", "joint_price_volume_velocity"),
                model.tokenizer.effective_auxiliary_gates().detach().cpu().tolist(),
                strict=True,
            )
        ),
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
    skipped_symbols: dict[str, str] = {}
    profile_config = ProfileConfig(
        bin_width_bps=args.bin_width_bps,
        max_profile_bins=args.max_profile_bins,
        state_timeframes=_state_timeframes(args.state_timeframe),
        relative_volume_lookback_sessions=args.normalization_lookback_sessions,
        relative_volume_min_observations=args.normalization_lookback_sessions,
        velocity_bins=args.velocity_bins,
        velocity_aggregation_minutes=args.velocity_aggregation_minutes,
        velocity_clip=args.velocity_clip,
        velocity_smoothing=args.velocity_smoothing,
    )
    label_config = LabelConfig(
        forward_sessions=args.forward_sessions,
        standardization_lookback_sessions=args.return_standardization_lookback,
        standardization_min_observations=args.return_standardization_min_observations,
        minimum_return_scale=args.minimum_return_scale_percent / 100.0,
        strong_threshold_sigma=args.strong_threshold_sigma,
    )
    for symbol in symbols:
        bars = source.fetch(symbol, "1m", args.start, args.end)
        states = PriceVolumeProfileBuilder(profile_config).build(bars)
        if not states:
            skipped_symbols[symbol] = "no valid profile states"
            continue
        try:
            labeled = ProfileLabelBuilder(label_config).build(states)
        except ValueError as error:
            if str(error) != "profile states do not contain enough future sessions":
                raise
            skipped_symbols[symbol] = str(error)
            continue
        observations.append(labeled)
        data_summary[symbol] = {
            "bars": len(bars),
            "profile_states": len(states),
            "observations": len(labeled),
            "minimum_profile_bins": min(len(state.bin_coordinates) for state in states),
            "maximum_profile_bins": max(len(state.bin_coordinates) for state in states),
        }
    if not observations:
        raise ValueError("none of the requested symbols produced observations")

    sequence_config = SequenceConfig(
        length=args.sequence_length,
        stride=args.sequence_stride,
        minimum_length=args.minimum_sequence_length,
    )
    model_config = ModelConfig(
        max_profile_bins=args.max_profile_bins,
        profile_channels=profile_config.profile_channels,
        latent_states=args.latent_states,
        transition_rank=args.transition_rank,
        strong_bear_threshold=label_config.strong_bear_threshold,
        direction_threshold=label_config.direction_threshold,
        strong_bull_threshold=label_config.strong_bull_threshold,
        feature_mode=args.feature_mode,
        auxiliary_gate_floor=args.auxiliary_gate_floor,
        auxiliary_gate_initial=args.auxiliary_gate_initial,
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
        kl_weight=args.kl_weight,
        ordinal_weight=args.ordinal_weight,
        symmetry_weight=args.symmetry_weight,
        pretrain_epochs=args.pretrain_epochs,
        auxiliary_warmup_epochs=args.auxiliary_warmup_epochs,
        pretrain_mask_probability=args.pretrain_mask_probability,
        sample_during_training=not args.deterministic_training,
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
        training_scope=args.training_scope,
        dataset_id=args.dataset_id,
    )
    policy = DecisionPolicy(
        DecisionPolicyConfig(
            uncertainty_aversion=args.uncertainty_aversion,
            transaction_cost_bps=args.transaction_cost_bps,
        )
    )
    report = evaluate_predictions(
        result.predictions,
        result.regime_return_means,
        result.regime_return_variances,
        policy=policy,
    )
    save_evaluation_report(args.output_dir, report)
    per_symbol: list[dict[str, object]] = []
    for symbol in data_summary:
        symbol_predictions = [
            item for item in result.predictions if item.symbol == symbol
        ]
        if not symbol_predictions:
            continue
        symbol_report = evaluate_predictions(
            symbol_predictions,
            result.regime_return_means,
            result.regime_return_variances,
            policy=policy,
        )
        save_evaluation_report(
            Path(args.output_dir) / "symbols" / symbol, symbol_report
        )
        per_symbol.append(
            {
                "symbol": symbol,
                "oos_predictions": len(symbol_predictions),
                "accuracy": symbol_report["accuracy"],
                "majority_baseline": symbol_report["majority_baseline"],
                "balanced_accuracy": symbol_report["balanced_accuracy"],
                "brier_score": symbol_report["brier_score"],
                "log_loss": symbol_report["log_loss"],
                "log_information_gain": symbol_report["baselines"]["log_information_gain"],
                "ranked_probability_score": symbol_report[
                    "ranked_probability_score"
                ],
                "backtest": {
                    key: symbol_report["backtest"][key]
                    for key in ("total_return", "max_drawdown")
                },
            }
        )
    summary = {
        "symbols": list(data_summary),
        "model_scope": args.training_scope,
        "dataset_id": args.dataset_id,
        "state_timeframe": args.state_timeframe,
        "profile_protocol": {
            "version": PROFILE_PROTOCOL,
            "bin_width_bps": args.bin_width_bps,
            "max_profile_bins": args.max_profile_bins,
            "channels": profile_config.profile_channels,
            "normalization_lookback_sessions": args.normalization_lookback_sessions,
            "relative_volume_baseline": "prior-mean-daily-volume/240",
            "velocity_bins": args.velocity_bins,
            "velocity_aggregation_minutes": args.velocity_aggregation_minutes,
            "velocity_clip": args.velocity_clip,
            "velocity_smoothing": args.velocity_smoothing,
            "velocity_representation": "z2-hydrodynamic-v1",
            "hydrodynamic_fields": [
                "density",
                "price_current",
                "volume_current",
                "price_temperature",
                "volume_temperature",
                "covariance_flux",
                "z2_odd_amplitude",
                "local_laminar_order",
                "global_signed_coherence",
                "current_divergence",
            ],
            "price_coordinate": "log-price-relative-to-previous-close",
        },
        "return_standardization": {
            "lookback_sessions": label_config.standardization_lookback_sessions,
            "minimum_observations": label_config.standardization_min_observations,
            "minimum_scale": label_config.minimum_return_scale,
            "definition": "next_open_to_close_return / known_rolling_std",
            "direction_zero": "actual_zero_return",
            "rolling_mean_usage": "diagnostic_baseline_only",
        },
        "regime_protocol": _regime_metadata(label_config),
        "model_config": asdict(model_config),
        "trainer_config": asdict(trainer_config),
        "sequence_config": asdict(sequence_config),
        "sequence_protocol": SEQUENCE_PROTOCOL,
        "walk_forward_config": asdict(walk_config),
        "data": data_summary,
        "skipped_symbols": skipped_symbols,
        "per_symbol": per_symbol,
        "folds": list(result.folds),
        "oos_predictions": len(result.predictions),
        "accuracy": report["accuracy"],
        "majority_baseline": report["majority_baseline"],
        "balanced_accuracy": report["balanced_accuracy"],
        "macro_f1": report["macro_f1"],
        "class_support": report["class_support"],
        "predicted_class_counts": report["predicted_class_counts"],
        "confusion_matrix": report["confusion_matrix"],
        "brier_score": report["brier_score"],
        "log_loss": report["log_loss"],
        "ranked_probability_score": report["ranked_probability_score"],
        "baselines": report["baselines"],
        "validation_gate": report["validation_gate"],
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








def _bnhmm_ablation(args: argparse.Namespace) -> int:
    import torch

    from .ablation import BNHMMAblationConfig, run_bnhmm_ablation
    from .training import ModelConfig, TrainerConfig
    from .walk_forward import WalkForwardConfig

    source = DuckDBBarSource(args.duckdb_path)
    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    if not symbols:
        symbols = source.list_symbols(args.symbol_limit)
    profile_config = ProfileConfig(
        bin_width_bps=args.bin_width_bps,
        max_profile_bins=args.max_profile_bins,
        state_timeframes=("1d",),
        relative_volume_lookback_sessions=args.normalization_lookback_sessions,
        relative_volume_min_observations=args.normalization_lookback_sessions,
        velocity_bins=args.velocity_bins,
        velocity_aggregation_minutes=args.velocity_aggregation_minutes,
        velocity_clip=args.velocity_clip,
        velocity_smoothing=args.velocity_smoothing,
    )
    label_config = LabelConfig(
        forward_sessions=1,
        standardization_lookback_sessions=args.return_standardization_lookback,
        standardization_min_observations=args.return_standardization_min_observations,
        minimum_return_scale=args.minimum_return_scale_percent / 100.0,
        strong_threshold_sigma=args.strong_threshold_sigma,
    )
    observations = []
    data_summary: dict[str, dict[str, int]] = {}
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

    sequence_config = SequenceConfig(length=args.sequence_length)
    model_config = ModelConfig(
        max_profile_bins=args.max_profile_bins,
        profile_channels=profile_config.profile_channels,
        latent_states=args.latent_states,
        transition_rank=args.transition_rank,
        strong_bear_threshold=label_config.strong_bear_threshold,
        direction_threshold=label_config.direction_threshold,
        strong_bull_threshold=label_config.strong_bull_threshold,
        feature_mode=args.feature_mode,
        auxiliary_gate_floor=args.auxiliary_gate_floor,
        auxiliary_gate_initial=args.auxiliary_gate_initial,
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
        kl_weight=args.kl_weight,
        brier_weight=0.25,
        ordinal_weight=args.ordinal_weight,
        consistency_weight=0.25,
        symmetry_weight=args.symmetry_weight,
        pretrain_epochs=args.pretrain_epochs,
        auxiliary_warmup_epochs=args.auxiliary_warmup_epochs,
        seed=args.seed,
    )
    walk_config = WalkForwardConfig(
        minimum_train_sessions=args.minimum_train_sessions,
        tuning_sessions=args.tuning_sessions,
        test_sessions=args.test_sessions,
        step_sessions=args.test_sessions,
        max_folds=1,
    )
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = tuple(
        int(value.strip())
        for value in args.confirmation_seeds.split(",")
        if value.strip()
    )

    def progress(payload: dict[str, object]) -> None:
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    result = run_bnhmm_ablation(
        observations,
        model_config,
        trainer_config,
        sequence_config,
        walk_config,
        device=device,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        mc_samples=args.mc_samples,
        ablation_config=BNHMMAblationConfig(
            screening_seed=args.seed,
            confirmation_seeds=seeds,
            log_loss_tolerance=args.ablation_log_loss_tolerance,
            brier_tolerance=args.ablation_brier_tolerance,
        ),
        progress=progress,
    )
    summary = {
        "strategy": "bnhmm-ablation",
        "dataset_id": args.dataset_id,
        "symbols_requested": len(symbols),
        "symbols_loaded": len(observations),
        "selected_candidate": result.selected_candidate.name,
        "selected_model_config": asdict(result.selected_candidate.model),
        "selected_trainer_config": asdict(result.selected_candidate.trainer),
        "selected_active_objectives": list(result.selected_candidate.active_objectives),
        "outer_metrics": result.outer_metrics,
        "fold": result.fold,
        "data": data_summary,
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

    boundary_data = subparsers.add_parser("boundary-data-build")
    boundary_data.add_argument("--duckdb-path", required=True)
    boundary_data.add_argument("--symbols", required=True)
    boundary_data.add_argument("--dataset-id", required=True)
    boundary_data.add_argument("--start", required=True)
    boundary_data.add_argument("--end", required=True)
    boundary_data.add_argument(
        "--cache-root",
        default="/home/wujianfeng/private/bnhmm-storage/path-efficiency-data",
    )
    boundary_data.set_defaults(handler=_boundary_data_build)

    boundary_flow = subparsers.add_parser("boundary-flow-fk-walk-forward")
    boundary_flow.add_argument("--duckdb-path", required=True)
    boundary_flow.add_argument("--symbols", required=True)
    boundary_flow.add_argument("--dataset-id", required=True)
    boundary_flow.add_argument("--start", required=True)
    boundary_flow.add_argument("--end", required=True)
    boundary_flow.add_argument("--output-dir", required=True)
    boundary_flow.add_argument(
        "--cache-root",
        default="/home/wujianfeng/private/bnhmm-storage/path-efficiency-data",
    )
    boundary_flow.add_argument("--minimum-train-sessions", type=int, default=120)
    boundary_flow.add_argument("--neutral-sigma", type=float, default=0.35)
    boundary_flow.add_argument("--fk-paths", type=int, default=10_000)
    boundary_flow.add_argument("--drift-half-life-sessions", type=float, default=60.0)
    boundary_flow.add_argument("--minimum-direction-probability", type=float, default=0.51)
    boundary_flow.add_argument("--minimum-signal-to-noise", type=float, default=0.10)
    boundary_flow.add_argument("--minimum-survival-probability", type=float, default=0.35)
    boundary_flow.add_argument(
        "--relative-strength-lookback-sessions", type=int, default=20
    )
    boundary_flow.add_argument("--transaction-cost-bps", type=float, default=5.0)
    boundary_flow.add_argument(
        "--maximum-drawdown", type=float, default=0.05
    )
    boundary_flow.add_argument("--seed", type=int, default=7)
    boundary_flow.add_argument("--max-folds", type=int)
    boundary_flow.set_defaults(handler=_boundary_flow_walk_forward)

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
    train.add_argument("--kl-weight", type=float, default=1e-3)
    train.add_argument("--ordinal-weight", type=float, default=0.0)
    train.add_argument("--symmetry-weight", type=float, default=0.2)
    train.add_argument("--pretrain-epochs", type=int, default=10)
    train.add_argument("--pretrain-mask-probability", type=float, default=0.15)
    train.add_argument("--deterministic-training", action="store_true")
    train.add_argument("--train-fraction", type=float, default=0.8)
    train.add_argument("--forward-sessions", type=int, default=1)
    train.add_argument("--strong-threshold-sigma", type=float, default=0.75)
    train.add_argument("--return-standardization-lookback", type=int, default=20)
    train.add_argument("--return-standardization-min-observations", type=int, default=10)
    train.add_argument("--minimum-return-scale-percent", type=float, default=0.2)
    train.add_argument("--bin-width-bps", type=float, default=10.0)
    train.add_argument("--max-profile-bins", type=int, default=512)
    train.add_argument("--normalization-lookback-sessions", type=int, default=10)
    train.add_argument("--velocity-bins", type=int, default=3)
    train.add_argument("--velocity-aggregation-minutes", type=int, default=10)
    train.add_argument("--velocity-clip", type=float, default=4.0)
    train.add_argument("--velocity-smoothing", type=float, default=0.25)
    train.add_argument(
        "--feature-mode",
        choices=("gated-full", "price-only", "price-volume", "price-velocity"),
        default="price-volume",
    )
    train.add_argument("--auxiliary-gate-floor", type=float, default=0.25)
    train.add_argument("--auxiliary-gate-initial", type=float, default=0.75)
    train.add_argument("--auxiliary-warmup-epochs", type=int, default=3)
    train.add_argument(
        "--state-timeframe",
        choices=("daily", "mixed"),
        default="daily",
        help="daily builds one state from each full 1m session; mixed also builds 60m states",
    )
    train.add_argument("--sequence-length", type=int, default=5)
    train.add_argument("--sequence-stride", type=int, default=1)
    train.add_argument("--minimum-sequence-length", type=int, default=2)
    train.add_argument("--d-model", type=int, default=48)
    train.add_argument("--latent-states", type=int, default=20)
    train.add_argument("--transition-rank", type=int, default=4)
    train.add_argument("--motif-dim", type=int, default=16)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--feedforward-dim", type=int, default=96)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--mc-samples", type=int, default=8)
    train.add_argument("--seed", type=int, default=7)
    train.set_defaults(handler=_train)

    walk = subparsers.add_parser("walk-forward")
    walk.add_argument("--duckdb-path", required=True)
    walk.add_argument("--symbols", default="")
    walk.add_argument("--symbol-limit", type=int, default=10)
    walk.add_argument(
        "--training-scope",
        choices=("per-symbol", "sector-pooled"),
        default="per-symbol",
    )
    walk.add_argument("--dataset-id")
    walk.add_argument("--start", required=True)
    walk.add_argument("--end", required=True)
    walk.add_argument("--output-dir", required=True)
    walk.add_argument("--device", default="auto")
    walk.add_argument("--epochs", type=int, default=10)
    walk.add_argument("--batch-size", type=int, default=8)
    walk.add_argument("--learning-rate", type=float, default=3e-4)
    walk.add_argument("--kl-weight", type=float, default=1e-3)
    walk.add_argument("--ordinal-weight", type=float, default=0.0)
    walk.add_argument("--symmetry-weight", type=float, default=0.2)
    walk.add_argument("--pretrain-epochs", type=int, default=10)
    walk.add_argument("--pretrain-mask-probability", type=float, default=0.15)
    walk.add_argument("--deterministic-training", action="store_true")
    walk.add_argument("--forward-sessions", type=int, default=1)
    walk.add_argument("--strong-threshold-sigma", type=float, default=0.75)
    walk.add_argument("--return-standardization-lookback", type=int, default=20)
    walk.add_argument("--return-standardization-min-observations", type=int, default=10)
    walk.add_argument("--minimum-return-scale-percent", type=float, default=0.2)
    walk.add_argument("--minimum-train-sessions", type=int, default=252)
    walk.add_argument("--tuning-sessions", type=int, default=60)
    walk.add_argument("--test-sessions", type=int, default=60)
    walk.add_argument("--step-sessions", type=int, default=60)
    walk.add_argument("--max-folds", type=int)
    walk.add_argument("--bin-width-bps", type=float, default=10.0)
    walk.add_argument("--max-profile-bins", type=int, default=512)
    walk.add_argument("--normalization-lookback-sessions", type=int, default=10)
    walk.add_argument("--velocity-bins", type=int, default=3)
    walk.add_argument("--velocity-aggregation-minutes", type=int, default=10)
    walk.add_argument("--velocity-clip", type=float, default=4.0)
    walk.add_argument("--velocity-smoothing", type=float, default=0.25)
    walk.add_argument(
        "--feature-mode",
        choices=("gated-full", "price-only", "price-volume", "price-velocity"),
        default="price-volume",
    )
    walk.add_argument("--auxiliary-gate-floor", type=float, default=0.25)
    walk.add_argument("--auxiliary-gate-initial", type=float, default=0.75)
    walk.add_argument("--auxiliary-warmup-epochs", type=int, default=3)
    walk.add_argument(
        "--state-timeframe",
        choices=("daily", "mixed"),
        default="daily",
        help="daily builds one state from each full 1m session; mixed also builds 60m states",
    )
    walk.add_argument("--sequence-length", type=int, default=5)
    walk.add_argument("--sequence-stride", type=int, default=1)
    walk.add_argument("--minimum-sequence-length", type=int, default=2)
    walk.add_argument("--d-model", type=int, default=48)
    walk.add_argument("--latent-states", type=int, default=20)
    walk.add_argument("--transition-rank", type=int, default=4)
    walk.add_argument("--motif-dim", type=int, default=16)
    walk.add_argument("--heads", type=int, default=4)
    walk.add_argument("--layers", type=int, default=2)
    walk.add_argument("--feedforward-dim", type=int, default=96)
    walk.add_argument("--dropout", type=float, default=0.1)
    walk.add_argument("--mc-samples", type=int, default=16)
    walk.add_argument("--uncertainty-aversion", type=float, default=2.0)
    walk.add_argument("--transaction-cost-bps", type=float, default=10.0)
    walk.add_argument("--seed", type=int, default=7)
    walk.set_defaults(handler=_walk_forward)

    bnhmm_ablation = subparsers.add_parser("bnhmm-ablation")
    bnhmm_ablation.add_argument("--duckdb-path", required=True)
    bnhmm_ablation.add_argument("--symbols", default="")
    bnhmm_ablation.add_argument("--symbol-limit", type=int, default=10)
    bnhmm_ablation.add_argument("--dataset-id", required=True)
    bnhmm_ablation.add_argument("--start", required=True)
    bnhmm_ablation.add_argument("--end", required=True)
    bnhmm_ablation.add_argument("--output-dir", required=True)
    bnhmm_ablation.add_argument("--device", default="auto")
    bnhmm_ablation.add_argument("--epochs", type=int, default=10)
    bnhmm_ablation.add_argument("--batch-size", type=int, default=8)
    bnhmm_ablation.add_argument("--learning-rate", type=float, default=3e-4)
    bnhmm_ablation.add_argument("--kl-weight", type=float, default=1e-3)
    bnhmm_ablation.add_argument("--ordinal-weight", type=float, default=0.25)
    bnhmm_ablation.add_argument("--symmetry-weight", type=float, default=0.2)
    bnhmm_ablation.add_argument("--pretrain-epochs", type=int, default=10)
    bnhmm_ablation.add_argument("--auxiliary-warmup-epochs", type=int, default=3)
    bnhmm_ablation.add_argument("--strong-threshold-sigma", type=float, default=0.75)
    bnhmm_ablation.add_argument("--return-standardization-lookback", type=int, default=20)
    bnhmm_ablation.add_argument("--return-standardization-min-observations", type=int, default=10)
    bnhmm_ablation.add_argument("--minimum-return-scale-percent", type=float, default=0.2)
    bnhmm_ablation.add_argument("--minimum-train-sessions", type=int, default=252)
    bnhmm_ablation.add_argument("--tuning-sessions", type=int, default=60)
    bnhmm_ablation.add_argument("--test-sessions", type=int, default=60)
    bnhmm_ablation.add_argument("--bin-width-bps", type=float, default=10.0)
    bnhmm_ablation.add_argument("--max-profile-bins", type=int, default=512)
    bnhmm_ablation.add_argument("--normalization-lookback-sessions", type=int, default=10)
    bnhmm_ablation.add_argument("--velocity-bins", type=int, default=3)
    bnhmm_ablation.add_argument("--velocity-aggregation-minutes", type=int, default=10)
    bnhmm_ablation.add_argument("--velocity-clip", type=float, default=4.0)
    bnhmm_ablation.add_argument("--velocity-smoothing", type=float, default=0.25)
    bnhmm_ablation.add_argument(
        "--feature-mode",
        choices=("gated-full", "price-only", "price-volume", "price-velocity"),
        default="price-volume",
    )
    bnhmm_ablation.add_argument("--auxiliary-gate-floor", type=float, default=0.25)
    bnhmm_ablation.add_argument("--auxiliary-gate-initial", type=float, default=0.75)
    bnhmm_ablation.add_argument("--sequence-length", type=int, default=5)
    bnhmm_ablation.add_argument("--d-model", type=int, default=48)
    bnhmm_ablation.add_argument("--latent-states", type=int, default=20)
    bnhmm_ablation.add_argument("--transition-rank", type=int, default=4)
    bnhmm_ablation.add_argument("--motif-dim", type=int, default=16)
    bnhmm_ablation.add_argument("--heads", type=int, default=4)
    bnhmm_ablation.add_argument("--layers", type=int, default=2)
    bnhmm_ablation.add_argument("--feedforward-dim", type=int, default=96)
    bnhmm_ablation.add_argument("--dropout", type=float, default=0.1)
    bnhmm_ablation.add_argument("--mc-samples", type=int, default=8)
    bnhmm_ablation.add_argument("--seed", type=int, default=7)
    bnhmm_ablation.add_argument("--confirmation-seeds", default="7,17,29")
    bnhmm_ablation.add_argument("--ablation-log-loss-tolerance", type=float, default=1e-3)
    bnhmm_ablation.add_argument("--ablation-brier-tolerance", type=float, default=5e-4)
    bnhmm_ablation.set_defaults(handler=_bnhmm_ablation)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
