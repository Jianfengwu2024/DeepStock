"""One-pass construction of reusable whole-sector FK inputs."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np

from ..data.source import create_bar_source
from ..sector_flow.panel import SectorMinutePanelBuilder
from .cache import (
    ANCHOR_PROTOCOL,
    CACHE_CONSTRUCTION_VERSION,
    SECTOR_CACHE_SLOT,
    BoundaryDataCache,
    BoundaryDataCacheKey,
)
from .execution import build_member_execution_constraints
from .normalization import CausalBoundaryNormalizer
from .sector_state import SectorDailyStateBuilder


class MinuteSource(Protocol):
    def fetch(self, symbol: str, timeframe: str, start: date, end: date): ...


def build_sector_boundary_cache(
    source: MinuteSource,
    cache: BoundaryDataCache,
    *,
    dataset_id: str,
    source_path: str | Path,
    symbols: tuple[str, ...],
    start: str | date,
    end: str | date,
) -> dict[str, object]:
    started = time.perf_counter()
    start_date = date.fromisoformat(start) if isinstance(start, str) else start
    end_date = date.fromisoformat(end) if isinstance(end, str) else end
    if end_date <= start_date:
        raise ValueError("cache end must follow start")

    warmup_start = start_date - timedelta(days=450)
    bars = {
        symbol: source.fetch(symbol, "1m", warmup_start, end_date)
        for symbol in symbols
    }
    panel = SectorMinutePanelBuilder().build(bars)
    normalized = CausalBoundaryNormalizer().transform(panel)
    if not len(normalized.timestamps):
        raise ValueError("sector cache requires available minute history")
    execution = build_member_execution_constraints(panel)
    built = SectorDailyStateBuilder().build(normalized)
    if not np.array_equal(built.states.sessions, execution.sessions):
        raise ValueError("execution constraints do not align with daily states")
    data = replace(
        built,
        member_next_hold_returns=execution.hold_returns,
        member_next_buy_returns=execution.buy_returns,
        member_next_buy_executable=execution.buy_executable,
        member_next_sell_returns=execution.sell_returns,
        member_next_sell_executable=execution.sell_executable,
    )
    key = BoundaryDataCacheKey.from_source(
        dataset_id=dataset_id,
        source_path=source_path,
        symbols=symbols,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        lookback_sessions=SECTOR_CACHE_SLOT,
    )
    cache.save(key, data)
    return {
        "dataset_id": dataset_id,
        "symbols": len(symbols),
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "rows": len(data.states.sessions),
        "construction_version": CACHE_CONSTRUCTION_VERSION,
        "anchor_protocol": ANCHOR_PROTOCOL,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_boundary_data_build(args) -> None:
    symbols = tuple(item.strip() for item in args.symbols.split(",") if item.strip())
    if len(symbols) < 4:
        raise ValueError("sector data construction requires at least four symbols")
    result = build_sector_boundary_cache(
        create_bar_source(args.duckdb_path),
        BoundaryDataCache(args.cache_root),
        dataset_id=args.dataset_id,
        source_path=args.duckdb_path,
        symbols=symbols,
        start=args.start,
        end=args.end,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
