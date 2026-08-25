"""Versioned, pickle-free cache for precomputed PathEfficiencyFK inputs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .types import (
    MonthlyBranchMembership,
    TwoBranchDailyStates,
    TwoBranchDataset,
    TwoBranchLabels,
)

if TYPE_CHECKING:
    from ..sector_flow.types import MonthlyRiverTopology


CACHE_FORMAT_VERSION = 1
CACHE_CONSTRUCTION_VERSION = "path-efficiency-two-branch-v2-convection"
CACHE_LOOKBACKS = (10, 20, 30, 40, 60, 90, 120)


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
        if self.lookback_sessions not in CACHE_LOOKBACKS:
            raise ValueError(f"supported lookback sessions are {CACHE_LOOKBACKS}")
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
        *,
        topologies: tuple["MonthlyRiverTopology", ...] = (),
    ) -> Path:
        dataset_root = self.root / key.dataset_id
        dataset_root.mkdir(parents=True, exist_ok=True)
        destination = dataset_root / f"lookback-{key.lookback_sessions}.npz"
        temporary = destination.with_suffix(".npz.tmp")
        membership_payload = json.dumps(
            [
                {
                    "month": str(item.month),
                    "training_cutoff": str(item.training_cutoff),
                    "branches": [list(item.branches[0]), list(item.branches[1])],
                }
                for item in data.memberships
            ],
            ensure_ascii=True,
            separators=(",", ":"),
        )
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
                convection_strength=data.convection_strength,
                convection_valid=data.convection_valid,
                convection_total_flow=data.convection_total_flow,
                convection_observations=data.convection_observations,
                memberships_json=np.asarray(membership_payload, dtype=np.str_),
                topology_months=np.asarray(
                    [item.month for item in topologies], dtype="datetime64[M]"
                ),
                topology_training_cutoffs=np.asarray(
                    [item.training_cutoff for item in topologies], dtype="datetime64[ns]"
                ),
                topology_assigned=self._stack_topology(topologies, "assigned", np.bool_),
                topology_band_ids=self._stack_topology(topologies, "band_ids", np.int64),
                topology_row_order=self._stack_topology(topologies, "row_order", np.int64),
                topology_common_similarity=self._stack_topology(
                    topologies, "common_similarity", np.float64, matrix=True
                ),
                topology_residual_similarity=self._stack_topology(
                    topologies, "residual_similarity", np.float64, matrix=True
                ),
                topology_fused_similarity=self._stack_topology(
                    topologies, "fused_similarity", np.float64, matrix=True
                ),
                topology_graph=self._stack_topology(
                    topologies, "graph", np.float64, matrix=True
                ),
                topology_cohesion=self._stack_topology(topologies, "cohesion", np.float64),
                topology_centrality=self._stack_topology(topologies, "centrality", np.float64),
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
            memberships = tuple(
                MonthlyBranchMembership(
                    month=np.datetime64(item["month"], "M"),
                    training_cutoff=np.datetime64(item["training_cutoff"], "ns"),
                    branches=(tuple(item["branches"][0]), tuple(item["branches"][1])),
                )
                for item in json.loads(str(payload["memberships_json"]))
            )
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
                memberships=memberships,
                symbols=tuple(str(item) for item in payload["symbols"]),
                member_next_returns=payload["member_next_returns"].copy(),
                member_next_valid=payload["member_next_valid"].copy(),
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
            if item.get("lookback_sessions") != key.lookback_sessions
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
            "dataset_id": key.dataset_id,
            "variants": sorted(variants, key=lambda item: item["lookback_sessions"]),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)

    @staticmethod
    def _stack_topology(
        topologies: tuple["MonthlyRiverTopology", ...],
        field: str,
        dtype: type[np.generic],
        *,
        matrix: bool = False,
    ) -> np.ndarray:
        if topologies:
            return np.stack([getattr(item, field) for item in topologies]).astype(
                dtype, copy=False
            )
        rank = 3 if matrix else 2
        return np.empty((0,) * rank, dtype=dtype)
