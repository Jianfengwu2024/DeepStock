"""Local, atomic progress telemetry; estimates use completed work only."""
from __future__ import annotations

import json
import os
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_current: ContextVar[ProgressReporter | None] = ContextVar("research_progress", default=None)


class ProgressReporter:
    def __init__(self, output: str | Path) -> None:
        self.path = Path(output) / "progress.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.phase_started = self.started
        self.last_write = 0.0
        self.values: dict[str, Any] = {}
        self._lock = threading.RLock()

    def update(self, phase: str | None = None, *, force: bool = True, **values: Any) -> None:
        with self._lock:
            now = time.monotonic()
            if phase and phase != self.values.get("phase"):
                self.phase_started = now
                self.values.update(phase=phase, completed_units=0, total_units=None, eta_seconds=None)
            self.values.update(values)
            elapsed = now - self.started
            done, total = self.values.get("completed_units"), self.values.get("total_units")
            fraction = min(1.0, max(0.0, done / total)) if isinstance(done, (int, float)) and isinstance(total, (int, float)) and total > 0 else None
            stage_elapsed = now - self.phase_started
            eta = stage_elapsed * (total - done) / done if fraction is not None and done > 0 else None
            if self.values.get("phase") == "completed":
                fraction, eta = 1.0, 0.0
            payload = {**self.values, "fraction": fraction, "elapsed_seconds": elapsed,
                       "stage_elapsed_seconds": stage_elapsed, "eta_seconds": max(0.0, eta) if eta is not None else None,
                       "eta_basis": "current_stage_measured_rate" if eta is not None else "warming_up",
                       "updated_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}
            if self.values.get("phase") == "walk_forward":
                payload.update(backtest_fraction=fraction, backtest_completed_units=done,
                               backtest_total_units=total, backtest_unit=self.values.get("unit"))
            self.values = payload
            if not force and now - self.last_write < 0.5:
                return
            temp = self.path.with_name(f".progress-{os.getpid()}.tmp")
            try:
                temp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")
                os.replace(temp, self.path)
            except OSError:
                # Observability must not interrupt training on transient file locks.
                return
            self.last_write = now


def start_progress(output: str | Path) -> ProgressReporter:
    reporter = ProgressReporter(output)
    _current.set(reporter)
    return reporter


def report_progress(phase: str | None = None, **values: Any) -> None:
    reporter = _current.get()
    if reporter is not None:
        reporter.update(phase, **values)
