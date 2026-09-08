"""Versioned, pickle-free cache for precomputed PathEfficiencyFK inputs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .types import (
    TwoBranchDailyStates,
    TwoBranchDataset,
    TwoBranchLabels,
)


CACHE_FORMAT_VERSION = 3
CACHE_CONSTRUCTION_VERSION = "sector-rolling-anchor-60d-v2"
ANCHOR_PROTOCOL = "rolling-60-session-previous-close"
SECTOR_CACHE_SLOT = 60


class BoundaryDataCacheMiss(FileNotFoundError):
    """Raised when a requested precomputed dataset is absent or stale."""


@dataclass(frozen=True, slots=True)
class BoundaryDataCacheKey:
    dataset_id: str
    source_path: Path
    source_fingerprint: str
    symbols: tuple[str, ...]
    start: str
    end: str
    lookback_sessions: int
    construction_version: str = CACHE_CONSTRUCTION_VERSION

    def __post_init__(self) -> None:
        if self.lookback_sessions != SECTOR_CACHE_SLOT:
            raise ValueError(
                f"supported lookback is the {SECTOR_CACHE_SLOT}-session sector cache slot"
            )
        if not self.dataset_id or not self.symbols:
            raise ValueError("cache key requires a dataset and symbols")

    @classmethod
    def from_source(
        cls,
        *,
        dataset_id: str,
        source_path: str | Path,
        symbols: tuple[str, ...],
        start: str,
        end: str,
        lookback_sessions: int,
    ) -> "BoundaryDataCacheKey":
        path = Path(source_path).resolve()
        stat = path.stat()
        fingerprint = hashlib.sha256(
            f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()
        return cls(
            dataset_id=dataset_id,
            source_path=path,
            source_fingerprint=fingerprint,
            symbols=tuple(symbols),
            start=start,
            end=end,
            lookback_sessions=lookback_sessions,
        )


class BoundaryDataCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def save(
        self,
        key: BoundaryDataCacheKey,
        data: TwoBranchDataset,
    ) -> Path:
        if data.memberships:
            raise ValueError("whole-sector FK data must not contain branch memberships")
        if data.states.values.shape[1] != 1:
            raise ValueError("whole-sector FK data must contain exactly one state stream")
        dataset_root = self.root / key.dataset_id
        dataset_root.mkdir(parents=True, exist_ok=True)
        destination = dataset_root / f"lookback-{key.lookback_sessions}.npz"
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                states_sessions=data.states.sessions,
                states_topology_months=data.states.topology_months,
                states_feature_names=np.asarray(data.states.feature_names, dtype=np.str_),
                states_values=data.states.values,
                states_valid=data.states.valid,
                labels_signal_sessions=data.labels.signal_sessions,
                labels_target_time=data.labels.target_time,
                labels_next_return=data.labels.next_return,
                labels_historical_volatility=data.labels.historical_volatility,
                labels_valid=data.labels.valid,
                symbols=np.asarray(data.symbols, dtype=np.str_),
                member_next_returns=data.member_next_returns,
                member_next_valid=data.member_next_valid,
                member_next_hold_returns=data.member_next_hold_returns,
                member_next_buy_returns=data.member_next_buy_returns,
                member_next_buy_executable=data.member_next_buy_executable,
                member_next_sell_returns=data.member_next_sell_returns,
                member_next_sell_executable=data.member_next_sell_executable,
                convection_lookbacks=np.asarray(
                    data.convection_lookbacks, dtype=np.int64
                ),
                convection_strength=data.convection_strength,
                convection_valid=data.convection_valid,
                convection_total_flow=data.convection_total_flow,
                convection_observations=data.convection_observations,
            )
        os.replace(temporary, destination)
        self._update_manifest(key, destination, data)
        return destination

    def load(
        self,
        key: BoundaryDataCacheKey,
        *,
        evaluation_start: str | np.datetime64 | None = None,
        evaluation_end: str | np.datetime64 | None = None,
    ) -> TwoBranchDataset:
        entry = self._entry(key)
        if entry.get("source_fingerprint") != key.source_fingerprint:
            raise BoundaryDataCacheMiss("cached source fingerprint is stale")
        if entry.get("construction_version") != key.construction_version:
            raise BoundaryDataCacheMiss("cached construction version is stale")
        if tuple(entry.get("symbols", ())) != key.symbols:
            raise BoundaryDataCacheMiss("cached symbol universe does not match")
        if entry.get("start", "9999-12-31") > key.start or entry.get("end", "0001-01-01") < key.end:
            raise BoundaryDataCacheMiss("cached date range does not cover the request")
        path = self.root / key.dataset_id / str(entry["file"])
        if not path.is_file():
            raise BoundaryDataCacheMiss(f"cached data file is missing: {path}")
        with np.load(path, allow_pickle=False) as payload:
            states = TwoBranchDailyStates(
                sessions=payload["states_sessions"].copy(),
                topology_months=payload["states_topology_months"].copy(),
                feature_names=tuple(str(item) for item in payload["states_feature_names"]),
                values=payload["states_values"].copy(),
                valid=payload["states_valid"].copy(),
            )
            labels = TwoBranchLabels(
                signal_sessions=payload["labels_signal_sessions"].copy(),
                target_time=payload["labels_target_time"].copy(),
                next_return=payload["labels_next_return"].copy(),
                historical_volatility=payload["labels_historical_volatility"].copy(),
                valid=payload["labels_valid"].copy(),
            )
            return TwoBranchDataset(
                states=states,
                labels=labels,
                memberships=(),
                symbols=tuple(str(item) for item in payload["symbols"]),
                member_next_returns=payload["member_next_returns"].copy(),
                member_next_valid=payload["member_next_valid"].copy(),
                member_next_hold_returns=payload["member_next_hold_returns"].copy(),
                member_next_buy_returns=payload["member_next_buy_returns"].copy(),
                member_next_buy_executable=payload["member_next_buy_executable"].copy(),
                member_next_sell_returns=payload["member_next_sell_returns"].copy(),
                member_next_sell_executable=payload["member_next_sell_executable"].copy(),
                convection_lookbacks=tuple(
                    int(item) for item in payload["convection_lookbacks"]
                ),
                convection_strength=payload["convection_strength"].copy(),
                convection_valid=payload["convection_valid"].copy(),
                convection_total_flow=payload["convection_total_flow"].copy(),
                convection_observations=payload["convection_observations"].copy(),
                evaluation_start=(
                    None if evaluation_start is None else np.datetime64(evaluation_start, "D")
                ),
                evaluation_end=(
                    None if evaluation_end is None else np.datetime64(evaluation_end, "D")
                ),
            )

    def status(self, dataset_id: str) -> dict[str, Any]:
        path = self.root / dataset_id / "manifest.json"
        if not path.is_file():
            return {"dataset_id": dataset_id, "variants": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def _entry(self, key: BoundaryDataCacheKey) -> dict[str, Any]:
        manifest = self.status(key.dataset_id)
        if manifest.get("format_version") != CACHE_FORMAT_VERSION:
            raise BoundaryDataCacheMiss("cached format version is stale")
        if manifest.get("anchor_protocol") != ANCHOR_PROTOCOL:
            raise BoundaryDataCacheMiss("cached anchor protocol is stale")
        for entry in manifest.get("variants", []):
            if entry.get("lookback_sessions") == key.lookback_sessions:
                return entry
        raise BoundaryDataCacheMiss(
            f"precomputed {key.lookback_sessions}-session data is not ready"
        )

    def _update_manifest(
        self,
        key: BoundaryDataCacheKey,
        destination: Path,
        data: TwoBranchDataset,
    ) -> None:
        dataset_root = destination.parent
        path = dataset_root / "manifest.json"
        current = self.status(key.dataset_id)
        variants = [
            item
            for item in current.get("variants", [])
            if item.get("construction_version") == key.construction_version
            and item.get("lookback_sessions") != key.lookback_sessions
        ]
        variants.append(
            {
                "lookback_sessions": key.lookback_sessions,
                "status": "ready",
                "file": destination.name,
                "rows": len(data.states.sessions),
                "source_path": str(key.source_path),
                "source_fingerprint": key.source_fingerprint,
                "construction_version": key.construction_version,
                "symbols": list(key.symbols),
                "start": key.start,
                "end": key.end,
                "training_cutoff": str(data.states.sessions[-1]),
                "available_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "anchor_protocol": ANCHOR_PROTOCOL,
            "dataset_id": key.dataset_id,
            "variants": sorted(variants, key=lambda item: item["lookback_sessions"]),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
