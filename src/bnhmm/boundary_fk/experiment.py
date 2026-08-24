"""Real-data construction and CLI execution for BoundaryFlowFK experiments."""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ..data.source import DuckDBBarSource
from ..sector_flow.panel import SectorMinutePanelBuilder
from ..sector_flow.topology import TopologyConfig
from .bayesian_state import BoundaryStateLabels
from .daily_state import BoundaryDailyStateBuilder, BoundaryTopologyBuilder
from .moments import BoundaryFieldConfig, BoundaryFieldExtractor
from .normalization import CausalBoundaryNormalizer
from .pipeline import (
    BoundaryFlowPipeline,
    BoundaryFlowPipelineConfig,
    BoundaryWalkForwardData,
)
from .reporting import write_json
from .synthetic_index import SyntheticSectorIndexBuilder
from .types import BoundaryDailyStates, NormalizedBoundaryPanel, SyntheticSectorIndex


def run_boundary_flow_experiment(args: Any) -> None:
    started = time.perf_counter()
    symbols = tuple(item.strip() for item in args.symbols.split(",") if item.strip())
    if len(symbols) < 3:
        raise ValueError("BoundaryFlowFK requires at least three symbols")
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end <= start:
        raise ValueError("BoundaryFlowFK end must follow start")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = DuckDBBarSource(args.duckdb_path)
    data, context = build_boundary_experiment_data(
        source,
        symbols,
        start,
        end,
        branch_count=args.branch_count,
        topology_lookback_sessions=args.topology_lookback_sessions,
        viscosity_window=args.viscosity_window,
    )
    _write_upstream_artifacts(output, context)
    result = BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(
            minimum_train_sessions=args.minimum_train_sessions,
            max_folds=args.max_folds,
            fk_paths=args.fk_paths,
            seed=args.seed,
            neutral_sigma=args.neutral_sigma,
            transaction_cost_bps=args.transaction_cost_bps,
            minimum_raw_bullish_probability=args.minimum_raw_bullish_probability,
            minimum_survival_bullish_probability=args.minimum_survival_bullish_probability,
            minimum_trend_survival_probability=args.minimum_trend_survival_probability,
            maximum_bullish_probability_std=args.maximum_bullish_probability_std,
            maximum_thickness_risk_quantile=args.maximum_thickness_risk_quantile,
        )
    ).run(data, output)
    summary = {
        "strategy": "boundary-flow-fk",
        "dataset_id": args.dataset_id,
        "symbols": len(symbols),
        "elapsed_seconds": time.perf_counter() - started,
        "model_names": list(result.model_names),
        "stages": [
            {
                "model_name": stage.model_name,
                "status": stage.status,
                "promotion_eligible": stage.promotion_eligible,
                "failure_reason": stage.failure_reason,
            }
            for stage in result.stages
        ],
    }
    write_json(output / "run-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_boundary_experiment_data(
    source: DuckDBBarSource,
    symbols: tuple[str, ...],
    start: date,
    end: date,
    *,
    branch_count: int,
    topology_lookback_sessions: int,
    viscosity_window: int,
) -> tuple[BoundaryWalkForwardData, dict[str, Any]]:
    warmup_start = start - timedelta(days=max(450, topology_lookback_sessions * 3))
    bars = {
        symbol: source.fetch(symbol, "1m", warmup_start, end)
        for symbol in symbols
    }
    panel = SectorMinutePanelBuilder().build(bars)
    market_caps = source.fetch_market_caps(
        symbols, warmup_start - timedelta(days=370), end
    )
    index = SyntheticSectorIndexBuilder().build(panel, market_caps)
    normalized = CausalBoundaryNormalizer().transform(panel)
    pre_start = index.valid & (index.timestamps < np.datetime64(start))
    if int(pre_start.sum()) < 1:
        raise ValueError("BoundaryFlowFK has no valid pre-start index history")
    index_scale = float(np.sqrt(np.mean(np.square(index.velocity[pre_start]))))
    stabilizer = _thickness_stabilizer(normalized, index, np.datetime64(start))
    moments = BoundaryFieldExtractor(
        BoundaryFieldConfig(
            viscosity_window=viscosity_window,
            minimum_viscosity_transitions=viscosity_window,
        )
    ).extract(
        normalized,
        index,
        index_velocity_scale=max(index_scale, 1e-8),
        thickness_stabilizer=stabilizer,
    )
    months = tuple(
        np.arange(
            np.datetime64(start, "M"),
            np.datetime64(end, "M") + np.timedelta64(1, "M"),
            dtype="datetime64[M]",
        )
    )
    topology_builder = BoundaryTopologyBuilder(
        TopologyConfig(lookback_sessions=topology_lookback_sessions)
    )
    topologies = topology_builder.build(normalized, months)
    prior_moments = moments.valid & (moments.timestamps < np.datetime64(start))
    if int(prior_moments.sum()) < 1:
        raise ValueError("BoundaryFlowFK has no valid pre-start boundary moments")
    turbulence_threshold = float(
        np.quantile(moments.turbulence[prior_moments], 0.9)
    )
    states = BoundaryDailyStateBuilder(branch_count=branch_count).build(
        normalized,
        moments,
        index,
        topologies,
        high_turbulence_threshold=turbulence_threshold,
    )
    data = _build_labels_and_side_data(
        states,
        index.daily_sessions,
        index.daily_open_to_close_return,
        index.daily_valid,
    )
    return data, {
        "index": index,
        "moments": moments,
        "topologies": topologies,
        "index_scale": index_scale,
        "thickness_stabilizer": stabilizer,
        "turbulence_threshold": turbulence_threshold,
    }


def _build_labels_and_side_data(
    states: BoundaryDailyStates,
    daily_sessions: np.ndarray,
    daily_returns: np.ndarray,
    daily_valid: np.ndarray,
) -> BoundaryWalkForwardData:
    rows = len(states.sessions)
    next_return = np.full(rows, np.nan, dtype=np.float64)
    historical_volatility = np.full(rows, np.nan, dtype=np.float64)
    target_time = np.full(rows, np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_daily = daily_valid & np.isfinite(daily_returns)
    sessions = daily_sessions[valid_daily].astype("datetime64[D]")
    returns = daily_returns[valid_daily].astype(np.float64)
    session_ids = sessions.astype(np.int64)
    state_ids = states.sessions.astype("datetime64[D]").astype(np.int64)
    for row, session_id in enumerate(state_ids):
        position = int(np.searchsorted(session_ids, session_id))
        if position < len(session_ids) - 1 and session_ids[position] == session_id:
            target_time[row] = sessions[position + 1].astype("datetime64[ns]")
            next_return[row] = returns[position + 1]
        prior = returns[max(0, position - 20) : position]
        if len(prior) == 20:
            historical_volatility[row] = max(float(np.std(prior)), 1e-6)

    return_valid = (
        states.valid & np.isfinite(next_return) & np.isfinite(historical_volatility)
    )
    momentum = states.feature("momentum_last60")
    thickness = states.feature("thickness_last60")
    risk = np.full(rows, 0.5, dtype=np.float64)
    survival_valid = np.zeros(rows, dtype=np.bool_)
    survival = np.zeros(rows, dtype=np.float64)
    threshold = np.maximum(0.002, 0.35 * historical_volatility)
    for row in range(rows):
        prior_rows = np.flatnonzero(states.valid[:row])
        if len(prior_rows):
            prior_thickness = thickness[prior_rows]
            risk[row] = float(np.mean(prior_thickness <= thickness[row]))
        direction_rows = prior_rows[-60:]
        if return_valid[row] and len(direction_rows) >= 20:
            direction_scale = max(float(np.std(momentum[direction_rows])), 1e-12)
            survival_valid[row] = abs(momentum[row]) > 0.25 * direction_scale
            survival[row] = float(
                np.sign(next_return[row]) == np.sign(momentum[row])
                and abs(next_return[row]) > threshold[row]
            )
    labels = BoundaryStateLabels(
        signal_sessions=states.sessions.copy(),
        target_time=target_time,
        next_return=next_return,
        historical_volatility=historical_volatility,
        survival=survival,
        return_valid=return_valid,
        survival_valid=survival_valid,
    )
    return BoundaryWalkForwardData(
        states=states,
        labels=labels,
        thickness_risk_quantile=risk,
        gross_strategy_returns=np.where(np.isfinite(next_return), next_return, 0.0),
    )


def _thickness_stabilizer(
    panel: NormalizedBoundaryPanel,
    index: SyntheticSectorIndex,
    cutoff: np.datetime64,
) -> float:
    columns = {symbol: position for position, symbol in enumerate(panel.symbols)}
    weights = {item.month: item for item in index.monthly_weights}
    values: list[float] = []
    for row in np.flatnonzero(panel.timestamps < cutoff):
        frozen = weights.get(panel.timestamps[row].astype("datetime64[M]"))
        if frozen is None:
            continue
        selected = np.asarray([columns[symbol] for symbol in frozen.symbols])
        valid = panel.velocity_valid[row, selected] & panel.density_valid[row, selected]
        if not np.any(valid):
            continue
        mass = frozen.weights[valid] * panel.relative_volume[row, selected[valid]]
        total = float(mass.sum())
        if total <= 0.0:
            continue
        mean_velocity = float(
            np.dot(mass, panel.standardized_velocity[row, selected[valid]]) / total
        )
        values.append(abs(mean_velocity))
    if not values:
        raise ValueError("BoundaryFlowFK cannot estimate a pre-start thickness stabilizer")
    return max(float(np.quantile(values, 0.25)), 1e-3)


def _write_upstream_artifacts(output: Path, context: dict[str, Any]) -> None:
    index = context["index"]
    moments = context["moments"]
    topologies = context["topologies"]
    write_json(
        output / "synthetic-index.json",
        {
            "excluded_symbols": list(index.excluded_symbols),
            "valid_minutes": int(index.valid.sum()),
            "daily_sessions": [str(item) for item in index.daily_sessions],
            "daily_returns": [
                float(value) if valid and np.isfinite(value) else None
                for value, valid in zip(
                    index.daily_open_to_close_return,
                    index.daily_valid,
                    strict=True,
                )
            ],
            "monthly_weights": [
                {
                    "month": str(item.month),
                    "weight_date": str(item.weight_date),
                    "symbols": list(item.symbols),
                    "weights": item.weights.tolist(),
                }
                for item in index.monthly_weights
            ],
        },
    )
    np.savez_compressed(
        output / "boundary-minute-moments.npz",
        timestamps=moments.timestamps,
        valid=moments.valid,
        water=moments.water,
        momentum=moments.momentum,
        velocity=moments.velocity,
        coherence=moments.coherence,
        turbulence=moments.turbulence,
        decoupling=moments.decoupling,
        thickness=moments.thickness,
        viscosity=moments.viscosity,
    )
    write_json(
        output / "branch-topologies.json",
        [
            {
                "month": str(item.month),
                "training_cutoff": str(item.training_cutoff),
                "symbols": list(item.symbols),
                "assigned": item.assigned.tolist(),
                "band_ids": item.band_ids.tolist(),
                "row_order": item.row_order.tolist(),
            }
            for item in topologies
        ],
    )
