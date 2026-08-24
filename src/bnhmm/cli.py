"""Operational commands for data smoke checks and storage migration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import (
    HOLOGRAPHIC_FIELD_NAMES,
    HOLOGRAPHIC_LITE_FIELD_NAMES,
    HOLOGRAPHIC_LITE_PROFILE_PROTOCOL,
    HOLOGRAPHIC_PROFILE_PROTOCOL,
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
from .data.holographic import (
    HolographicPhaseProfileBuilder,
    HolographicProfileBuilder,
)
from .data.profile import PriceVolumeProfileBuilder
from .data.source import BrokerBarSource, DuckDBBarSource
from .repair import RepairVariantCatalog
from .storage import apply_migrations


def _state_timeframes(mode: str) -> tuple[str, ...]:
    return ("1d",) if mode == "daily" else ("60m", "1d")


def _holographic_fields(value: str) -> tuple[str, ...]:
    fields = tuple(item.strip() for item in value.split(",") if item.strip())
    if not fields:
        raise argparse.ArgumentTypeError("at least one holographic field is required")
    return fields


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


def _sector_flow_walk_forward(args: argparse.Namespace) -> int:
    from .sector_flow.pipeline import SectorFlowConfig, SectorFlowPipeline

    source = DuckDBBarSource(args.duckdb_path)
    symbols = tuple(
        item.strip() for item in args.symbols.split(",") if item.strip()
    )
    if not symbols:
        symbols = tuple(source.list_symbols(args.symbol_limit))
    path_windows = tuple(
        int(item.strip())
        for item in args.path_windows.split(",")
        if item.strip()
    )
    config = SectorFlowConfig(
        symbols=symbols,
        start=args.start,
        end=args.end,
        protocol=args.protocol,
        dataset_id=args.dataset_id,
        sector_index_symbol=args.sector_index_symbol,
        topology_lookback_sessions=args.topology_lookback_sessions,
        normalization_lookback_sessions=args.normalization_lookback_sessions,
        normalization_minimum_observations=args.normalization_minimum_observations,
        minimum_train_sessions=args.minimum_train_sessions,
        maximum_lag_minutes=args.maximum_lag_minutes,
        minimum_attribution_rows=args.minimum_attribution_rows,
        factor_process_variance=args.factor_process_variance,
        path_windows=path_windows,
        ranking_window=args.ranking_window,
        evaluation_horizon_minutes=args.evaluation_horizon_minutes,
        top_k=args.top_k,
        transaction_cost_bps=args.transaction_cost_bps,
        monte_carlo_samples=args.monte_carlo_samples,
        random_seed=args.seed,
        max_folds=args.max_folds,
        regime_context_sessions=args.regime_context_sessions,
        regime_hidden_channels=args.regime_hidden_channels,
        regime_latent_states=args.regime_latent_states,
        regime_neutral_sigma=args.regime_neutral_sigma,
        regime_volatility_lookback_sessions=(
            args.regime_volatility_lookback_sessions
        ),
        regime_volatility_minimum_observations=(
            args.regime_volatility_minimum_observations
        ),
        regime_confidence_threshold=args.regime_confidence_threshold,
        regime_maximum_bull_variance=args.regime_maximum_bull_variance,
        regime_epochs=args.regime_epochs,
        regime_batch_size=args.regime_batch_size,
        regime_learning_rate=args.regime_learning_rate,
        regime_kl_weight=args.regime_kl_weight,
        regime_state_loss_weight=args.regime_state_loss_weight,
        regime_mc_samples=args.regime_mc_samples,
        device=args.device,
    )
    result = SectorFlowPipeline(config).run(source, args.output_dir)
    summary = {
        "model_version": (
            "sector-regime-bnhmm-v1"
            if args.protocol == "river-regime-v3"
            else "sector-flow-river-v2"
            if args.protocol == "river-topology-v2"
            else "sector-flow-bayes-v1"
        ),
        "protocol": args.protocol,
        "dataset_id": args.dataset_id,
        "symbols": len(symbols),
        "folds": len(result.folds),
        "predictions": len(
            result.regime_predictions
            or result.graph_predictions
            or result.predictions
        ),
        "metrics": result.metrics,
        "output_dir": args.output_dir,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
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

    repair_variant = (
        RepairVariantCatalog.resolve(args.repair_variant)
        if args.repair_variant
        else None
    )
    if repair_variant is not None and repair_variant.stage > 6:
        raise ValueError(
            f"{repair_variant.name} is not implemented in the current repair stage"
        )
    source = DuckDBBarSource(args.duckdb_path)
    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    if not symbols:
        symbols = source.list_symbols(args.symbol_limit)
    observations = []
    data_summary: dict[str, dict[str, int]] = {}
    skipped_symbols: dict[str, str] = {}
    holographic = args.profile_kind == "holographic"
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
        holographic_volume_bins=args.holographic_volume_bins,
        holographic_smoothing=args.holographic_smoothing,
        holographic_fields=args.holographic_fields,
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
        if holographic and repair_variant is not None and repair_variant.stage >= 4:
            builder = HolographicPhaseProfileBuilder(profile_config)
        elif holographic:
            builder = HolographicProfileBuilder(profile_config)
        else:
            builder = PriceVolumeProfileBuilder(profile_config)
        states = builder.build(bars)
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
        if holographic:
            data_summary[symbol]["flow_samples"] = sum(
                state.flow_samples for state in states
            )
    if not observations:
        raise ValueError("none of the requested symbols produced observations")

    sequence_config = SequenceConfig(
        length=args.sequence_length,
        stride=args.sequence_stride,
        minimum_length=args.minimum_sequence_length,
    )
    model_config = ModelConfig(
        max_profile_bins=args.max_profile_bins,
        profile_channels=(
            profile_config.holographic_profile_channels
            if holographic
            else profile_config.profile_channels
        ),
        latent_states=args.latent_states,
        tokenizer_kind="holographic-2d" if holographic else "profile-motif",
        holographic_volume_bins=profile_config.holographic_volume_bins,
        holographic_field_count=len(profile_config.holographic_fields),
        holographic_field_names=profile_config.holographic_fields,
        holographic_conv_layers=args.holographic_conv_layers,
        daily_encoder=(
            repair_variant.daily_encoder if repair_variant else "shallow-cnn"
        ),
        temporal_encoder=(
            repair_variant.temporal_encoder if repair_variant else "transformer"
        ),
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
        repair_variant=repair_variant.name if repair_variant else None,
        control_mode=repair_variant.control_mode if repair_variant else "full",
        include_metadata=(
            repair_variant is None
            or repair_variant.control_mode not in {"image-only", "prior-only"}
        ),
        direct_autoregression=(
            repair_variant is None
            or repair_variant.control_mode not in {"image-only", "prior-only"}
        ),
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
        "repair_variant": repair_variant.name if repair_variant else None,
        "state_timeframe": args.state_timeframe,
        "profile_protocol": {
            "version": (
                (
                    HOLOGRAPHIC_LITE_PROFILE_PROTOCOL
                    if profile_config.holographic_fields
                    == HOLOGRAPHIC_LITE_FIELD_NAMES
                    else HOLOGRAPHIC_PROFILE_PROTOCOL
                )
                if holographic
                else PROFILE_PROTOCOL
            ),
            "bin_width_bps": args.bin_width_bps,
            "max_profile_bins": args.max_profile_bins,
            "channels": (
                profile_config.holographic_profile_channels
                if holographic
                else profile_config.profile_channels
            ),
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
            **(
                {
                    "domain": (
                        ["relative_log_price", "standardized_10m_price_velocity"]
                        if repair_variant is not None and repair_variant.stage >= 4
                        else ["relative_log_price", "relative_volume_level"]
                    ),
                    "volume_bins": profile_config.holographic_volume_bins,
                    "field_smoothing": profile_config.holographic_smoothing,
                    "fields": list(profile_config.holographic_fields),
                    "flow": (
                        ["volume_weighted_phase_mass"]
                        if repair_variant is not None and repair_variant.stage >= 4
                        else ["price_velocity", "volume_level_velocity"]
                    ),
                    "geometry": (
                        ["price_density_gradient", "velocity_density_gradient"]
                        if repair_variant is not None and repair_variant.stage >= 4
                        else ["divergence", "vorticity"]
                    ),
                }
                if holographic
                else {}
            ),
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
    if repair_variant is not None:
        from .repair import RepairExperimentPipeline

        RepairExperimentPipeline.write_stage_report(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _stock_transformer_walk_forward(args: argparse.Namespace) -> int:
    import torch

    from .evaluation import (
        DecisionPolicy,
        DecisionPolicyConfig,
        evaluate_predictions,
        save_evaluation_report,
    )
    from .stock_transformer import (
        ConservationConfig,
        StockPipelineConfig,
        StockTrainerConfig,
        StockTransformerConfig,
        TokenizationConfig,
        run_stock_transformer_walk_forward,
    )
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
    conservation_config = ConservationConfig(
        grid_points=args.conservation_grid_points,
        ridge=args.conservation_ridge,
    )
    tokenization_config = TokenizationConfig(
        minimum_token_bins=args.minimum_token_bins,
        maximum_tokens=args.maximum_tokens,
        boundary_threshold=args.boundary_threshold,
    )
    model_config = StockTransformerConfig(
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        maximum_days=args.sequence_length,
        maximum_tokens=args.maximum_tokens,
    )
    trainer_config = StockTrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        conservation_weight=args.conservation_weight,
        seed=args.seed,
    )
    pipeline_config = StockPipelineConfig(
        sequence_length=args.sequence_length,
        mc_samples=args.mc_samples,
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
    result = run_stock_transformer_walk_forward(
        observations,
        conservation_config,
        tokenization_config,
        model_config,
        trainer_config,
        pipeline_config,
        walk_config,
        device=device,
        output_dir=args.output_dir,
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
    per_symbol = []
    for symbol in data_summary:
        predictions = [item for item in result.predictions if item.symbol == symbol]
        if not predictions:
            continue
        symbol_report = evaluate_predictions(
            predictions,
            result.regime_return_means,
            result.regime_return_variances,
            policy=policy,
        )
        per_symbol.append(
            {
                "symbol": symbol,
                "oos_predictions": len(predictions),
                "accuracy": symbol_report["accuracy"],
                "majority_baseline": symbol_report["majority_baseline"],
                "balanced_accuracy": symbol_report["balanced_accuracy"],
                "brier_score": symbol_report["brier_score"],
                "log_loss": symbol_report["log_loss"],
                "backtest": {
                    key: symbol_report["backtest"][key]
                    for key in ("total_return", "max_drawdown")
                },
            }
        )
    summary = {
        "strategy": "stock-transformer",
        "symbols": list(data_summary),
        "model_scope": "sector-pooled",
        "dataset_id": args.dataset_id,
        "three_layer_protocol": {
            "conservation": "stochastic-continuity-v1",
            "tokenization": "gradient-current-sentence-v1",
            "model": "stock-transformer-v1",
        },
        "conservation_config": asdict(conservation_config),
        "tokenization_config": asdict(tokenization_config),
        "model_config": asdict(model_config),
        "trainer_config": asdict(trainer_config),
        "pipeline_config": asdict(pipeline_config),
        "walk_forward_config": asdict(walk_config),
        "data": data_summary,
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


def _fk_stock_walk_forward(args: argparse.Namespace) -> int:
    from .evaluation import (
        DecisionPolicy,
        DecisionPolicyConfig,
        evaluate_predictions,
        save_evaluation_report,
    )
    from .fk_stock import (
        FKGeneratorConfig,
        FKStockPipelineConfig,
        FeynmanKacConfig,
        run_fk_stock_walk_forward,
    )
    from .stock_transformer import ConservationConfig
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
    conservation_config = ConservationConfig()
    generator_config = FKGeneratorConfig()
    solver_config = FeynmanKacConfig(
        strong_threshold_sigma=args.strong_threshold_sigma,
    )
    pipeline_config = FKStockPipelineConfig()
    walk_config = WalkForwardConfig(
        minimum_train_sessions=args.minimum_train_sessions,
        tuning_sessions=args.tuning_sessions,
        test_sessions=args.test_sessions,
        step_sessions=args.step_sessions,
        max_folds=args.max_folds,
    )
    result = run_fk_stock_walk_forward(
        observations,
        conservation_config,
        generator_config,
        solver_config,
        pipeline_config,
        walk_config,
        output_dir=args.output_dir,
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
    per_symbol = []
    for symbol in data_summary:
        predictions = [item for item in result.predictions if item.symbol == symbol]
        if not predictions:
            continue
        symbol_report = evaluate_predictions(
            predictions,
            result.regime_return_means,
            result.regime_return_variances,
            policy=policy,
        )
        per_symbol.append(
            {
                "symbol": symbol,
                "oos_predictions": len(predictions),
                "accuracy": symbol_report["accuracy"],
                "majority_baseline": symbol_report["majority_baseline"],
                "balanced_accuracy": symbol_report["balanced_accuracy"],
                "brier_score": symbol_report["brier_score"],
                "log_loss": symbol_report["log_loss"],
                "backtest": {
                    key: symbol_report["backtest"][key]
                    for key in ("total_return", "max_drawdown")
                },
            }
        )
    summary = {
        "strategy": "fk-stock",
        "symbols": list(data_summary),
        "model_scope": "sector-pooled",
        "dataset_id": args.dataset_id,
        "three_layer_protocol": {
            "input": "rolling-standardized-return-diffusion-v2",
            "generator": "single-feature-bayesian-ridge-v2",
            "solver": "zero-potential-posterior-v2",
        },
        "generator_config": asdict(generator_config),
        "solver_config": asdict(solver_config),
        "pipeline_config": asdict(pipeline_config),
        "walk_forward_config": asdict(walk_config),
        "data": data_summary,
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


def _fk_stock_ablation(args: argparse.Namespace) -> int:
    from .fk_stock import (
        FKAblationConfig,
        FKGeneratorConfig,
        FeynmanKacConfig,
        FeynmanKacPotentialConfig,
        run_fk_stock_ablation,
    )
    from .stock_transformer import ConservationConfig
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
    )
    label_config = LabelConfig(
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
    conservation_config = ConservationConfig(
        grid_points=args.conservation_grid_points,
        ridge=args.conservation_ridge,
    )
    generator_config = FKGeneratorConfig(
        drift_lookback_sessions=args.fk_drift_lookback_sessions,
        ridge=args.fk_ridge,
        minimum_return_history=args.fk_minimum_return_history,
        minimum_fit_rows=args.fk_minimum_fit_rows,
        minimum_noise_variance=args.fk_minimum_noise_variance,
    )
    solver_config = FeynmanKacConfig(
        strong_threshold_sigma=args.strong_threshold_sigma,
    )
    potential_config = FeynmanKacPotentialConfig(
        potential_floor=args.fk_potential_floor,
        noise_potential_weight=args.fk_noise_potential_weight,
        uncertainty_potential_weight=args.fk_uncertainty_potential_weight,
        incoherence_potential_weight=args.fk_incoherence_potential_weight,
        maximum_potential=args.fk_maximum_potential,
        mc_samples=args.mc_samples,
        seed=args.seed,
    )
    walk_config = WalkForwardConfig(
        minimum_train_sessions=args.minimum_train_sessions,
        tuning_sessions=args.tuning_sessions,
        test_sessions=args.test_sessions,
        step_sessions=args.test_sessions,
        max_folds=1,
    )
    result = run_fk_stock_ablation(
        observations,
        conservation_config,
        generator_config,
        solver_config,
        potential_config,
        walk_config,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        ablation_config=FKAblationConfig(
            log_loss_tolerance=args.ablation_log_loss_tolerance,
            brier_tolerance=args.ablation_brier_tolerance,
        ),
    )
    selected_outer = next(
        item
        for item in result.outer_potential_results
        if item["name"] == result.selected_potential
    )
    summary = {
        "strategy": "fk-stock-ablation",
        "dataset_id": args.dataset_id,
        "symbols_requested": len(symbols),
        "symbols_loaded": len(observations),
        "selected_features": list(result.selected_features),
        "selected_feature_count": len(result.selected_features),
        "selected_potential": result.selected_potential,
        "selected_outer_metrics": selected_outer,
        "fold": result.fold,
        "data": data_summary,
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

    sector_flow = subparsers.add_parser("sector-flow-walk-forward")
    sector_flow.add_argument("--duckdb-path", required=True)
    sector_flow.add_argument("--symbols", default="")
    sector_flow.add_argument("--symbol-limit", type=int, default=100)
    sector_flow.add_argument("--dataset-id")
    sector_flow.add_argument("--start", required=True)
    sector_flow.add_argument("--end", required=True)
    sector_flow.add_argument("--output-dir", required=True)
    sector_flow.add_argument(
        "--protocol",
        choices=("legacy-v1", "river-topology-v2", "river-regime-v3"),
        default="river-topology-v2",
    )
    sector_flow.add_argument(
        "--topology-lookback-sessions", type=int, default=60
    )
    sector_flow.add_argument(
        "--normalization-lookback-sessions", type=int, default=20
    )
    sector_flow.add_argument(
        "--normalization-minimum-observations", type=int, default=10
    )
    sector_flow.add_argument("--minimum-train-sessions", type=int, default=60)
    sector_flow.add_argument("--maximum-lag-minutes", type=int, default=30)
    sector_flow.add_argument("--minimum-attribution-rows", type=int, default=120)
    sector_flow.add_argument("--factor-process-variance", type=float)
    sector_flow.add_argument("--path-windows", default="5,15,30,60")
    sector_flow.add_argument("--ranking-window", type=int, default=60)
    sector_flow.add_argument(
        "--evaluation-horizon-minutes", type=int, default=30
    )
    sector_flow.add_argument("--top-k", type=int, default=10)
    sector_flow.add_argument("--transaction-cost-bps", type=float, default=10.0)
    sector_flow.add_argument("--monte-carlo-samples", type=int, default=1000)
    sector_flow.add_argument("--max-folds", type=int)
    sector_flow.add_argument("--seed", type=int, default=20260820)
    sector_flow.add_argument("--sector-index-symbol", default="881121")
    sector_flow.add_argument("--regime-context-sessions", type=int, default=5)
    sector_flow.add_argument("--regime-hidden-channels", type=int, default=16)
    sector_flow.add_argument("--regime-latent-states", type=int, default=12)
    sector_flow.add_argument("--regime-neutral-sigma", type=float, default=0.35)
    sector_flow.add_argument(
        "--regime-volatility-lookback-sessions", type=int, default=20
    )
    sector_flow.add_argument(
        "--regime-volatility-minimum-observations", type=int, default=10
    )
    sector_flow.add_argument(
        "--regime-confidence-threshold", type=float, default=0.6
    )
    sector_flow.add_argument(
        "--regime-maximum-bull-variance", type=float, default=0.02
    )
    sector_flow.add_argument("--regime-epochs", type=int, default=20)
    sector_flow.add_argument("--regime-batch-size", type=int, default=8)
    sector_flow.add_argument("--regime-learning-rate", type=float, default=3e-4)
    sector_flow.add_argument("--regime-kl-weight", type=float, default=1e-4)
    sector_flow.add_argument(
        "--regime-state-loss-weight", type=float, default=0.25
    )
    sector_flow.add_argument("--regime-mc-samples", type=int, default=16)
    sector_flow.add_argument("--device", default="cpu")
    sector_flow.set_defaults(handler=_sector_flow_walk_forward)

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
        "--profile-kind",
        choices=("phase-space", "holographic"),
        default="phase-space",
    )
    walk.add_argument("--holographic-volume-bins", type=int, default=5)
    walk.add_argument("--holographic-smoothing", type=float, default=0.75)
    walk.add_argument(
        "--holographic-fields",
        type=_holographic_fields,
        default=HOLOGRAPHIC_FIELD_NAMES,
        metavar="FIELD[,FIELD...]",
    )
    walk.add_argument("--holographic-conv-layers", type=int, choices=(1, 2), default=2)
    walk.add_argument("--repair-variant", choices=RepairVariantCatalog.names())
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

    stock = subparsers.add_parser("stock-transformer-walk-forward")
    stock.add_argument("--duckdb-path", required=True)
    stock.add_argument("--symbols", default="")
    stock.add_argument("--symbol-limit", type=int, default=10)
    stock.add_argument("--dataset-id")
    stock.add_argument("--start", required=True)
    stock.add_argument("--end", required=True)
    stock.add_argument("--output-dir", required=True)
    stock.add_argument("--device", default="auto")
    stock.add_argument("--epochs", type=int, default=10)
    stock.add_argument("--batch-size", type=int, default=16)
    stock.add_argument("--learning-rate", type=float, default=3e-4)
    stock.add_argument("--conservation-weight", type=float, default=0.5)
    stock.add_argument("--strong-threshold-sigma", type=float, default=0.75)
    stock.add_argument("--return-standardization-lookback", type=int, default=20)
    stock.add_argument("--return-standardization-min-observations", type=int, default=10)
    stock.add_argument("--minimum-return-scale-percent", type=float, default=0.2)
    stock.add_argument("--minimum-train-sessions", type=int, default=252)
    stock.add_argument("--tuning-sessions", type=int, default=60)
    stock.add_argument("--test-sessions", type=int, default=60)
    stock.add_argument("--step-sessions", type=int, default=60)
    stock.add_argument("--max-folds", type=int)
    stock.add_argument("--bin-width-bps", type=float, default=10.0)
    stock.add_argument("--max-profile-bins", type=int, default=512)
    stock.add_argument("--normalization-lookback-sessions", type=int, default=10)
    stock.add_argument("--conservation-grid-points", type=int, default=64)
    stock.add_argument("--conservation-ridge", type=float, default=1e-3)
    stock.add_argument("--minimum-token-bins", type=int, default=2)
    stock.add_argument("--maximum-tokens", type=int, default=16)
    stock.add_argument("--boundary-threshold", type=float, default=1.5)
    stock.add_argument("--sequence-length", type=int, default=5)
    stock.add_argument("--d-model", type=int, default=64)
    stock.add_argument("--heads", type=int, default=4)
    stock.add_argument("--layers", type=int, default=3)
    stock.add_argument("--feedforward-dim", type=int, default=128)
    stock.add_argument("--dropout", type=float, default=0.1)
    stock.add_argument("--mc-samples", type=int, default=16)
    stock.add_argument("--uncertainty-aversion", type=float, default=2.0)
    stock.add_argument("--transaction-cost-bps", type=float, default=10.0)
    stock.add_argument("--seed", type=int, default=7)
    stock.set_defaults(handler=_stock_transformer_walk_forward)

    fk = subparsers.add_parser("fk-stock-walk-forward")
    fk.add_argument("--duckdb-path", required=True)
    fk.add_argument("--symbols", default="")
    fk.add_argument("--symbol-limit", type=int, default=10)
    fk.add_argument("--dataset-id", required=True)
    fk.add_argument("--start", required=True)
    fk.add_argument("--end", required=True)
    fk.add_argument("--output-dir", required=True)
    fk.add_argument("--strong-threshold-sigma", type=float, default=0.75)
    fk.add_argument("--return-standardization-lookback", type=int, default=20)
    fk.add_argument("--return-standardization-min-observations", type=int, default=10)
    fk.add_argument("--minimum-return-scale-percent", type=float, default=0.2)
    fk.add_argument("--minimum-train-sessions", type=int, default=252)
    fk.add_argument("--tuning-sessions", type=int, default=60)
    fk.add_argument("--test-sessions", type=int, default=60)
    fk.add_argument("--step-sessions", type=int, default=60)
    fk.add_argument("--max-folds", type=int)
    fk.add_argument("--bin-width-bps", type=float, default=10.0)
    fk.add_argument("--max-profile-bins", type=int, default=512)
    fk.add_argument("--normalization-lookback-sessions", type=int, default=10)
    fk.add_argument("--uncertainty-aversion", type=float, default=2.0)
    fk.add_argument("--transaction-cost-bps", type=float, default=10.0)
    fk.set_defaults(handler=_fk_stock_walk_forward)

    ablation = subparsers.add_parser("fk-stock-ablation")
    ablation.add_argument("--duckdb-path", required=True)
    ablation.add_argument("--symbols", default="")
    ablation.add_argument("--symbol-limit", type=int, default=10)
    ablation.add_argument("--dataset-id", required=True)
    ablation.add_argument("--start", required=True)
    ablation.add_argument("--end", required=True)
    ablation.add_argument("--output-dir", required=True)
    ablation.add_argument("--strong-threshold-sigma", type=float, default=0.75)
    ablation.add_argument("--return-standardization-lookback", type=int, default=20)
    ablation.add_argument("--return-standardization-min-observations", type=int, default=10)
    ablation.add_argument("--minimum-return-scale-percent", type=float, default=0.2)
    ablation.add_argument("--minimum-train-sessions", type=int, default=252)
    ablation.add_argument("--tuning-sessions", type=int, default=60)
    ablation.add_argument("--test-sessions", type=int, default=30)
    ablation.add_argument("--bin-width-bps", type=float, default=10.0)
    ablation.add_argument("--max-profile-bins", type=int, default=512)
    ablation.add_argument("--normalization-lookback-sessions", type=int, default=10)
    ablation.add_argument("--conservation-grid-points", type=int, default=64)
    ablation.add_argument("--conservation-ridge", type=float, default=1e-3)
    ablation.add_argument("--fk-drift-lookback-sessions", type=int, default=20)
    ablation.add_argument("--fk-ridge", type=float, default=1.0)
    ablation.add_argument("--fk-minimum-return-history", type=int, default=5)
    ablation.add_argument("--fk-minimum-fit-rows", type=int, default=20)
    ablation.add_argument("--fk-minimum-noise-variance", type=float, default=1e-4)
    ablation.add_argument("--fk-potential-floor", type=float, default=0.0)
    ablation.add_argument("--fk-noise-potential-weight", type=float, default=0.75)
    ablation.add_argument("--fk-uncertainty-potential-weight", type=float, default=0.75)
    ablation.add_argument("--fk-incoherence-potential-weight", type=float, default=0.25)
    ablation.add_argument("--fk-maximum-potential", type=float, default=4.0)
    ablation.add_argument("--mc-samples", type=int, default=16)
    ablation.add_argument("--seed", type=int, default=7)
    ablation.add_argument("--ablation-log-loss-tolerance", type=float, default=1e-3)
    ablation.add_argument("--ablation-brier-tolerance", type=float, default=5e-4)
    ablation.set_defaults(handler=_fk_stock_ablation)

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

