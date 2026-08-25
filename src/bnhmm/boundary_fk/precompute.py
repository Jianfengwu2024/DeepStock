"""One-pass construction of reusable PathEfficiencyFK datasets."""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np

from ..data.source import DuckDBBarSource
from ..sector_flow.panel import SectorMinutePanelBuilder
from ..sector_flow.topology import TopologyConfig
from .branch_state import TwoBranchDailyStateBuilder
from .cache import CACHE_LOOKBACKS, BoundaryDataCache, BoundaryDataCacheKey
from .daily_state import BoundaryTopologyBuilder
from .execution import build_member_execution_constraints
from .normalization import CausalBoundaryNormalizer


class MinuteSource(Protocol):
    def fetch(self, symbol: str, timeframe: str, start: date, end: date): ...


def build_boundary_cache_variants(
    source: MinuteSource,
    cache: BoundaryDataCache,
    *,
    dataset_id: str,
    source_path: str | Path,
    symbols: tuple[str, ...],
    start: str | date,
    end: str | date,
    lookbacks: tuple[int, ...] = CACHE_LOOKBACKS,
) -> dict[str, object]:
    started = time.perf_counter()
    start_date = date.fromisoformat(start) if isinstance(start, str) else start
    end_date = date.fromisoformat(end) if isinstance(end, str) else end
    if end_date <= start_date:
        raise ValueError("cache end must follow start")
    if tuple(sorted(set(lookbacks))) != tuple(lookbacks):
        raise ValueError("cache lookbacks must be unique and increasing")
    if any(item not in CACHE_LOOKBACKS for item in lookbacks):
        raise ValueError(f"supported lookbacks are {CACHE_LOOKBACKS}")

    warmup_start = start_date - timedelta(days=max(450, max(lookbacks) * 5))
    bars = {
        symbol: source.fetch(symbol, "1m", warmup_start, end_date)
        for symbol in symbols
    }
    panel = SectorMinutePanelBuilder().build(bars)
    normalized = CausalBoundaryNormalizer().transform(panel)
    execution = build_member_execution_constraints(panel)
    topology_start = start_date - timedelta(days=300)
    months = tuple(
        np.arange(
            np.datetime64(topology_start, "M"),
            np.datetime64(end_date, "M") + np.timedelta64(1, "M"),
            dtype="datetime64[M]",
        )
    )
    rows: list[dict[str, object]] = []
    for lookback in lookbacks:
        topologies = BoundaryTopologyBuilder(
            TopologyConfig(
                lookback_sessions=lookback,
                minimum_sessions=max(2, int(np.ceil(lookback * 2.0 / 3.0))),
                minimum_pair_sessions=max(2, int(np.ceil(lookback / 2.0))),
                maximum_bands=2,
                minimum_band_size=max(2, int(np.ceil(len(symbols) * 0.10))),
            )
        ).build(normalized, months)
        built = TwoBranchDailyStateBuilder().build(normalized, topologies)
        if not np.array_equal(built.states.sessions, execution.sessions):
            raise ValueError("execution constraints do not align with daily states")
        data = type(built)(
            states=built.states,
            labels=built.labels,
            memberships=built.memberships,
            symbols=built.symbols,
            member_next_returns=built.member_next_returns,
            member_next_valid=built.member_next_valid,
            member_next_buy_returns=execution.buy_returns,
            member_next_buy_executable=execution.buy_executable,
            member_next_sell_returns=execution.sell_returns,
            member_next_sell_executable=execution.sell_executable,
            convection_strength=built.convection_strength,
            convection_valid=built.convection_valid,
            convection_total_flow=built.convection_total_flow,
            convection_observations=built.convection_observations,
        )
        key = BoundaryDataCacheKey.from_source(
            dataset_id=dataset_id,
            source_path=source_path,
            symbols=symbols,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            lookback_sessions=lookback,
        )
        cache.save(key, data, topologies=topologies)
        rows.append(
            {
                "lookback_sessions": lookback,
                "rows": len(data.states.sessions),
                "memberships": len(data.memberships),
            }
        )
    return {
        "dataset_id": dataset_id,
        "lookbacks": list(lookbacks),
        "symbols": len(symbols),
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "variants": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_boundary_data_build(args) -> None:
    symbols = tuple(item.strip() for item in args.symbols.split(",") if item.strip())
    if len(symbols) < 4:
        raise ValueError("boundary data construction requires at least four symbols")
    lookbacks = tuple(int(item.strip()) for item in args.lookbacks.split(",") if item.strip())
    result = build_boundary_cache_variants(
        DuckDBBarSource(args.duckdb_path),
        BoundaryDataCache(args.cache_root),
        dataset_id=args.dataset_id,
        source_path=args.duckdb_path,
        symbols=symbols,
        start=args.start,
        end=args.end,
        lookbacks=lookbacks,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
