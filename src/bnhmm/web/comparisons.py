"""Persisted orchestration for PathEfficiencyFK window comparisons."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .jobs import JobRecord, WalkForwardJobRequest


COMPARISON_WINDOWS = (90, 60, 40, 30, 20, 10)
TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled", "interrupted"}


class ComparisonJobs(Protocol):
    def submit(self, request: WalkForwardJobRequest) -> JobRecord: ...

    def get(self, identifier: str) -> JobRecord | None: ...

    def summary(self, identifier: str) -> dict[str, Any] | None: ...

    def artifact_path(self, identifier: str, name: str) -> Path | None: ...


class WindowComparisonRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    request: WalkForwardJobRequest


class ComparisonChild(BaseModel):
    window_sessions: int
    job_id: str


class ComparisonRecord(BaseModel):
    id: str
    name: str
    created_at: datetime
    children: list[ComparisonChild]


class DuplicateComparisonNameError(RuntimeError):
    pass


class ComparisonManager:
    def __init__(self, root: str | Path, jobs: ComparisonJobs) -> None:
        self.root = Path(root).resolve() / "_comparisons"
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = jobs
        self._lock = threading.RLock()
        self._records: dict[str, ComparisonRecord] = {}
        self._load_existing()

    def create(self, payload: WindowComparisonRequest) -> ComparisonRecord:
        if payload.request.strategy != "boundary-flow-fk":
            raise ValueError("window comparison only supports PathEfficiencyFK")
        with self._lock:
            normalized = payload.name.strip().casefold()
            if any(item.name.casefold() == normalized for item in self._records.values()):
                raise DuplicateComparisonNameError(
                    f"comparison name already exists: {payload.name}"
                )
            children = []
            for window in COMPARISON_WINDOWS:
                request = payload.request.model_copy(
                    update={
                        "name": f"{payload.name.strip()}-{window}d",
                        "topology_lookback_sessions": window,
                    }
                )
                job = self.jobs.submit(request)
                children.append(
                    ComparisonChild(window_sessions=window, job_id=job.id)
                )
            record = ComparisonRecord(
                id=f"comparison-{uuid.uuid4().hex[:12]}",
                name=payload.name.strip(),
                created_at=datetime.now(timezone.utc),
                children=children,
            )
            self._records[record.id] = record
            self._persist(record)
            return record.model_copy(deep=True)

    def list(self) -> list[ComparisonRecord]:
        with self._lock:
            return [
                item.model_copy(deep=True)
                for item in sorted(
                    self._records.values(),
                    key=lambda row: row.created_at,
                    reverse=True,
                )
            ]

    def get(self, identifier: str) -> ComparisonRecord | None:
        with self._lock:
            record = self._records.get(identifier)
            return record.model_copy(deep=True) if record else None

    def delete(self, identifier: str) -> bool:
        with self._lock:
            if identifier not in self._records:
                return False
            path = self.root / f"{identifier}.json"
            if path.is_file():
                path.unlink()
            del self._records[identifier]
            return True

    def payload(self, identifier: str) -> dict[str, Any]:
        record = self.get(identifier)
        if record is None:
            raise KeyError(identifier)
        windows = []
        statuses = []
        for child in record.children:
            job = self.jobs.get(child.job_id)
            status = job.status if job else "failed"
            statuses.append(status)
            item = {
                "window_sessions": child.window_sessions,
                "job_id": child.job_id,
                "status": status,
                "error": job.error if job else "child job not found",
                "metrics": {},
                "equity_curve": [],
            }
            if job is not None and status == "completed":
                item.update(self._completed_result(job))
            windows.append(item)
        if any(status == "running" for status in statuses):
            status = "running"
        elif any(status == "queued" for status in statuses):
            status = "queued"
        elif any(status == "completed" for status in statuses):
            status = "completed"
        else:
            status = "failed"
        return {
            **record.model_dump(mode="json"),
            "status": status,
            "completed_count": statuses.count("completed"),
            "finished_count": sum(
                item in TERMINAL_JOB_STATUSES for item in statuses
            ),
            "windows": windows,
        }

    def _completed_result(self, job: JobRecord) -> dict[str, Any]:
        evaluation_path = self.jobs.artifact_path(
            job.id, "boundary-policy-evaluation.json"
        )
        predictions_path = self.jobs.artifact_path(
            job.id, "feynman-kac-predictions.json"
        )
        evaluation = self._read_json(evaluation_path, [])
        predictions = self._read_json(predictions_path, [])
        stage = evaluation[0] if isinstance(evaluation, list) and evaluation else {}
        metrics = stage.get("metrics", {}) if isinstance(stage, dict) else {}
        equity = 1.0
        cost_rate = float(job.request.transaction_cost_bps) / 10_000.0
        curve = []
        if isinstance(predictions, list):
            for prediction in predictions:
                if not isinstance(prediction, dict):
                    continue
                equity *= (
                    1.0
                    + float(prediction.get("gross_strategy_return", 0.0) or 0.0)
                    - cost_rate * float(prediction.get("turnover", 0.0) or 0.0)
                )
                curve.append(
                    {
                        "signal_time": prediction.get("signal_time"),
                        "value": equity,
                    }
                )
        return {"metrics": metrics, "equity_curve": curve}

    @staticmethod
    def _read_json(path: Path | None, fallback: Any) -> Any:
        if path is None:
            return fallback
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return fallback

    def _persist(self, record: ComparisonRecord) -> None:
        path = self.root / f"{record.id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _load_existing(self) -> None:
        for path in self.root.glob("comparison-*.json"):
            try:
                record = ComparisonRecord.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            self._records[record.id] = record
