"""Background construction service for reusable PathEfficiencyFK data."""

from __future__ import annotations

import json
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..boundary_fk.cache import BoundaryDataCache


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
    def __init__(self, root: str | Path, project_root: str | Path, python: str) -> None:
        self.root = Path(root).resolve()
        self.project_root = Path(project_root).resolve()
        self.python = python
        self.cache = BoundaryDataCache(self.root)
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}

    def submit(
        self,
        *,
        database_path: str,
        symbols: tuple[str, ...],
        dataset_id: str,
        start: str,
        end: str,
    ) -> dict[str, Any]:
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
            }
            self._records[dataset_id] = record
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
        return record

    def _run(self, dataset_id: str, command: list[str]) -> None:
        dataset_root = self.root / dataset_id
        dataset_root.mkdir(parents=True, exist_ok=True)
        log_path = dataset_root / "build.log"
        try:
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
            (dataset_root / "build-status.json").write_text(
                json.dumps(self._records[dataset_id], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

