"""Background construction service for reusable PathEfficiencyFK data."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..boundary_fk.cache import BoundaryDataCache, BoundaryDataCacheKey, BoundaryDataCacheMiss, SECTOR_CACHE_SLOT


def build_boundary_data_command(
    *,
    python: str,
    database_path: str,
    symbols: tuple[str, ...],
    dataset_id: str,
    start: str,
    end: str,
    cache_root: str | Path,
) -> list[str]:
    return [
        python,
        "-m",
        "bnhmm.cli",
        "boundary-data-build",
        "--duckdb-path",
        database_path,
        "--symbols",
        ",".join(symbols),
        "--dataset-id",
        dataset_id,
        "--start",
        start,
        "--end",
        end,
        "--cache-root",
        str(cache_root),
    ]


class BoundaryDataBuildManager:
    def __init__(self, root: str | Path, project_root: str | Path, python: str, *, resource_lock: Any = None) -> None:
        self.root = Path(root).resolve()
        self.project_root = Path(project_root).resolve()
        self.python = python
        self.cache = BoundaryDataCache(self.root)
        self._lock = threading.RLock()
        self._resource_lock = resource_lock if resource_lock is not None else threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        for path in self.root.glob("*/build-status.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and payload.get("dataset_id"):
                    if payload.get("status") == "running":
                        payload.update(status="interrupted", error="数据准备服务在完成前重启，请重新准备")
                    self._records[str(payload["dataset_id"])] = payload
            except (OSError, json.JSONDecodeError):
                continue

    @staticmethod
    def _validate_id(dataset_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", dataset_id):
            raise ValueError("无效的策略数据集标识")

    def _persist(self, record: dict[str, Any]) -> None:
        self._validate_id(record["dataset_id"])
        path = self.root / record["dataset_id"] / "build-status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def describe(self, dataset: dict[str, Any], *, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        """Describe inputs from disk; never read minute bars or construct data."""
        dataset_id = dataset["dataset_id"]
        self._validate_id(dataset_id)
        try:
            manifest = self.cache.status(dataset_id)
            entry = next((v for v in manifest.get("variants", []) if v.get("lookback_sessions") == SECTOR_CACHE_SLOT), None)
        except BoundaryDataCacheMiss:
            entry = None
        result = {
            "strategy": "boundary-flow-fk", "strategy_name": "FK-贝叶斯",
            "dataset_id": dataset_id, "market_dataset_id": dataset_id,
            "name": f"FK-贝叶斯 · {dataset['name']}", "lookback_sessions": SECTOR_CACHE_SLOT,
            "symbols": len(dataset["symbols"]), "rows": entry.get("rows", 0) if entry else 0,
            "start": entry.get("start") if entry else None, "end": entry.get("end") if entry else None,
            "available_at": entry.get("available_at") if entry else None,
            "construction_version": entry.get("construction_version") if entry else None,
            "source_fingerprint": entry.get("source_fingerprint") if entry else None,
            "source_path": dataset["database_path"], "cache_root": str(self.root),
            "dataset_path": str(self.root / dataset_id / entry["file"]) if entry else None,
            "status": "not_ready", "reason": "尚未准备策略数据集",
            "persistent": True, "run_mode": "reuse_only",
        }
        with self._lock:
            build = dict(self._records.get(dataset_id, {}))
        result["build"] = build or None
        progress_path = self.root / dataset_id / "progress.json"
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            result["progress"] = progress if isinstance(progress, dict) else None
        except (OSError, json.JSONDecodeError):
            result["progress"] = None
        result["history"] = manifest.get("history", []) if entry else []
        if not entry:
            if build.get("status") in {"running", "failed", "interrupted"}:
                result.update(status=build["status"], reason=build.get("error") or "策略数据准备中")
            return result
        try:
            key = BoundaryDataCacheKey.from_source(
                dataset_id=dataset_id, source_path=dataset["database_path"], symbols=tuple(dataset["symbols"]),
                start=start or entry["start"], end=end or entry["end"], lookback_sessions=SECTOR_CACHE_SLOT,
            )
            self.cache.inspect(key)
            result.update(status="ready", reason="已保存，可直接复用")
        except (OSError, ValueError, KeyError) as exc:
            result.update(status="stale", reason=str(exc))
        if build.get("status") == "running":
            result.update(status="running", reason="正在准备新版本，旧版本文件保留")
        return result

    def require_ready(self, dataset: dict[str, Any], *, start: str, end: str) -> dict[str, Any]:
        result = self.describe(dataset, start=start, end=end)
        if result["status"] != "ready":
            raise ValueError("FK 策略数据集未就绪：" + result["reason"] + "。请在数据服务页先准备并保存，运行中不会临时构建。")
        return result

    def submit(
        self,
        *,
        database_path: str,
        symbols: tuple[str, ...],
        dataset_id: str,
        start: str,
        end: str,
    ) -> dict[str, Any]:
        self._validate_id(dataset_id)
        if end <= start:
            raise ValueError("策略数据覆盖结束必须晚于开始")
        with self._lock:
            if self._records.get(dataset_id, {}).get("status") == "running":
                raise RuntimeError("该板块的策略数据正在构造")
        try:
            key = BoundaryDataCacheKey.from_source(
                dataset_id=dataset_id,
                source_path=database_path,
                symbols=symbols,
                start=start,
                end=end,
                lookback_sessions=SECTOR_CACHE_SLOT,
            )
            self.cache.inspect(key)
            record = {
                "dataset_id": dataset_id,
                "status": "ready",
                "reused": True,
                "started_at": None,
                "finished_at": None,
                "error": None,
            }
            with self._lock:
                self._records[dataset_id] = record
                self._persist(record)
            return self.status(dataset_id)
        except BoundaryDataCacheMiss:
            pass
        with self._lock:
            current = self._records.get(dataset_id)
            if current and current["status"] == "running":
                raise RuntimeError("该板块的策略数据正在构造")
            record = {
                "dataset_id": dataset_id,
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "error": None,
                "source_path": database_path,
                "start": start,
                "end": end,
                "reused": False,
            }
            self._records[dataset_id] = record
            self._persist(record)
        from ..progress import ProgressReporter
        ProgressReporter(self.root / dataset_id).update("queued", message="等待本机训练任务释放计算资源")
        command = build_boundary_data_command(
            python=self.python,
            database_path=database_path,
            symbols=symbols,
            dataset_id=dataset_id,
            start=start,
            end=end,
            cache_root=self.root,
        )
        threading.Thread(
            target=self._run,
            args=(dataset_id, command),
            name=f"boundary-data-{dataset_id}",
            daemon=True,
        ).start()
        return self.status(dataset_id)

    def status(self, dataset_id: str) -> dict[str, Any]:
        self._validate_id(dataset_id)
        with self._lock:
            record = dict(
                self._records.get(
                    dataset_id,
                    {
                        "dataset_id": dataset_id,
                        "status": "idle",
                        "started_at": None,
                        "finished_at": None,
                        "error": None,
                    },
                )
            )
        record["cache"] = self.cache.status(dataset_id)
        progress_path = self.root / dataset_id / "progress.json"
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            record["progress"] = progress if isinstance(progress, dict) else None
        except (OSError, json.JSONDecodeError):
            record["progress"] = None
        if record.get("status") == "idle":
            variants = record["cache"].get("variants", [])
            if any(item.get("lookback_sessions") == SECTOR_CACHE_SLOT and item.get("status") == "ready" for item in variants):
                record["status"] = "ready"
                record["reused"] = True
        return record

    def delete(self, dataset_id: str) -> None:
        self._validate_id(dataset_id)
        with self._lock:
            if self._records.get(dataset_id, {}).get("status") == "running":
                raise RuntimeError("正在构建的 FK 数据不能清理")
            target = (self.root / dataset_id).resolve()
            if not target.is_relative_to(self.root):
                raise ValueError("FK 数据目录超出管理范围")
            if target.exists():
                shutil.rmtree(target, ignore_errors=False)
            self._records.pop(dataset_id, None)

    def _run(self, dataset_id: str, command: list[str]) -> None:
        with self._resource_lock:
            self._run_locked(dataset_id, command)

    def _run_locked(self, dataset_id: str, command: list[str]) -> None:
        dataset_root = self.root / dataset_id
        dataset_root.mkdir(parents=True, exist_ok=True)
        log_path = dataset_root / "build.log"
        try:
            from ..progress import ProgressReporter
            ProgressReporter(dataset_root).update("starting", message="开始准备独立策略数据集")
            with log_path.open("w", encoding="utf-8") as log:
                log.write("command: " + " ".join(command) + "\n")
                result = subprocess.run(
                    command,
                    cwd=self.project_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
            status = "completed" if result.returncode == 0 else "failed"
            error = None if result.returncode == 0 else f"构造进程退出码 {result.returncode}"
        except Exception as exc:  # pragma: no cover - operating-system failure
            status, error = "failed", str(exc)
        with self._lock:
            self._records[dataset_id].update(
                status=status,
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=error,
            )
            self._persist(self._records[dataset_id])

