"""One-pass construction of reusable whole-sector FK inputs."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Protocol

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
from ..progress import start_progress


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
    progress: Callable[[str, dict[str, object]], None] | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    start_date = date.fromisoformat(start) if isinstance(start, str) else start
    end_date = date.fromisoformat(end) if isinstance(end, str) else end
    if end_date <= start_date:
        raise ValueError("cache end must follow start")

    warmup_start = start_date - timedelta(days=450)
    telemetry = start_progress(Path(cache.root) / dataset_id) if hasattr(cache, "root") else None
    def notify(phase: str, **details: object) -> None:
        if telemetry is not None:
            names = {"loading_member_bars": "读取已保存的本地分钟数据", "building_minute_panel": "构建成分股分钟矩阵",
                     "normalizing_minute_panel": "因果标准化", "building_execution_constraints": "生成成交约束",
                     "building_sector_daily_states": "计算60日滚动状态与标签", "saving_fk_cache": "持久化策略输入"}
            counts = {"completed_units": details["members_loaded"], "total_units": len(symbols), "unit": "股票"} if "members_loaded" in details else {}
            telemetry.update(phase, message=names.get(phase, phase), **counts, **details)
        if progress is not None:
            progress(phase, details)

    bars = {}
    for index, symbol in enumerate(symbols):
        bars[symbol] = source.fetch(symbol, "1m", warmup_start, end_date)
        if (index + 1) % 10 == 0 or index + 1 == len(symbols):
            notify("loading_member_bars", members_loaded=index+1, total_members=len(symbols))
    notify("building_minute_panel")
    panel = SectorMinutePanelBuilder().build(bars)
    notify("normalizing_minute_panel")
    normalized = CausalBoundaryNormalizer().transform(panel)
    if not len(normalized.timestamps):
        raise ValueError("sector cache requires available minute history")
    notify("building_execution_constraints")
    execution = build_member_execution_constraints(panel)
    notify("building_sector_daily_states")
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
    notify("saving_fk_cache", session_rows=len(data.states.sessions))
    cache.save(key, data)
    if telemetry is not None:
        telemetry.update("completed", message="策略数据集已保存，下次运行直接复用", completed_units=len(symbols), total_units=len(symbols))
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
