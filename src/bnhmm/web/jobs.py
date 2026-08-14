"""Persistent single-worker queue for isolated walk-forward processes."""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


JobStatus = Literal["queued", "running", "completed", "failed", "cancelled", "interrupted"]
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ActiveJobDeletionError(RuntimeError):
    pass


class WalkForwardJobRequest(BaseModel):
    name: str = "experiment"
    database_path: str = "/home/share/data/market.duckdb"
    symbols: list[str] = Field(default_factory=lambda: ["000090", "000099", "000488"])
    training_scope: Literal["per-symbol", "shared-legacy"] = "per-symbol"
    start: date = date(2024, 1, 1)
    end: date = date(2026, 8, 5)
    device: str = "cuda:0"
    epochs: int = Field(10, ge=1, le=1000)
    batch_size: int = Field(8, ge=1, le=512)
    learning_rate: float = Field(3e-4, gt=0, le=1.0)
    kl_weight: float = Field(1e-3, ge=0.0, le=1.0)
    ordinal_weight: float = Field(0.0, ge=0.0, le=10.0)
    pretrain_epochs: int = Field(0, ge=0, le=1000)
    pretrain_mask_probability: float = Field(0.15, gt=0.0, lt=1.0)
    deterministic_training: bool = False
    forward_sessions: int = Field(1, ge=1, le=60)
    minimum_train_sessions: int = Field(252, ge=10)
    tuning_sessions: int = Field(60, ge=2)
    test_sessions: int = Field(60, ge=1)
    step_sessions: int = Field(60, ge=1)
    max_folds: int | None = Field(None, ge=1, le=100)
    bin_width_bps: float = Field(10.0, gt=0.0, le=100.0)
    max_profile_bins: int = Field(512, ge=8, le=2048)
    normalization_lookback_sessions: int = Field(10, ge=2, le=120)
    velocity_bins: int = Field(3, ge=3, le=31)
    velocity_clip: float = Field(4.0, gt=0.0, le=20.0)
    velocity_smoothing: float = Field(0.25, ge=0.0, le=10.0)
    profile_protocol: Literal[
        "phase-space-joint-v5",
        "phase-space-gated-v4",
        "phase-space-log-v3",
        "variable-log-bins-v2",
        "legacy-fixed-bins",
    ] = (
        "phase-space-joint-v5"
    )
    state_timeframe: Literal["daily", "mixed"] = "daily"
    sequence_length: int = Field(128, ge=8, le=2048)
    sequence_stride: int = Field(64, ge=1, le=2048)
    minimum_sequence_length: int = Field(32, ge=2, le=2048)
    d_model: int = Field(96, ge=8, le=2048)
    latent_states: int = Field(50, ge=1, le=512)
    feature_mode: Literal[
        "gated-full",
        "price-only",
        "price-volume",
        "price-velocity",
        "legacy-entangled",
    ] = "gated-full"
    motif_dim: int = Field(32, ge=2, le=512)
    heads: int = Field(4, ge=1, le=64)
    layers: int = Field(3, ge=1, le=48)
    feedforward_dim: int = Field(192, ge=8, le=8192)
    dropout: float = Field(0.1, ge=0.0, lt=1.0)
    mc_samples: int = Field(16, ge=2, le=256)
    uncertainty_aversion: float = Field(2.0, ge=0.0, le=100.0)
    transaction_cost_bps: float = Field(10.0, ge=0.0, le=1000.0)
    seed: int = Field(7, ge=0, le=2**31 - 1)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not _SAFE_NAME.fullmatch(value):
            raise ValueError("name may contain letters, numbers, dot, dash, and underscore")
        return value

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, value: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        if not normalized:
            raise ValueError("at least one symbol is required")
        if len(normalized) > 200:
            raise ValueError("at most 200 symbols are allowed")
        if any(not re.fullmatch(r"[A-Za-z0-9._-]{1,24}", item) for item in normalized):
            raise ValueError("a symbol contains unsupported characters")
        return normalized

    @model_validator(mode="after")
    def validate_relationships(self, info: ValidationInfo) -> "WalkForwardJobRequest":
        if self.end <= self.start:
            raise ValueError("end must be later than start")
        if self.minimum_train_sessions <= self.tuning_sessions:
            raise ValueError("minimum_train_sessions must exceed tuning_sessions")
        if self.minimum_sequence_length > self.sequence_length:
            raise ValueError("minimum_sequence_length cannot exceed sequence_length")
        if self.sequence_stride > self.sequence_length:
            raise ValueError("sequence_stride cannot exceed sequence_length")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if self.velocity_bins % 2 == 0:
            raise ValueError("velocity_bins must be odd")
        profile_channels = 4 + self.velocity_bins**2
        legacy = bool((info.context or {}).get("legacy_job"))
        if self.feature_mode == "legacy-entangled" and not legacy:
            raise ValueError("legacy-entangled is reserved for historical jobs")
        if not legacy and self.latent_states < profile_channels:
            raise ValueError("latent_states cannot be fewer than profile channels")
        return self


class JobRecord(BaseModel):
    id: str
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    return_code: int | None = None
    process_id: int | None = None
    current_gpu_memory_mib: int = 0
    peak_gpu_memory_mib: int = 0
    error: str | None = None
    output_dir: str
    request: WalkForwardJobRequest


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def gpu_memory_for_pid(process_id: int) -> int:
    """Return aggregate dedicated GPU memory for one host process."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    total = 0
    for line in result.stdout.splitlines():
        columns = [item.strip() for item in line.split(",")]
        if len(columns) != 2:
            continue
        try:
            if int(columns[0]) == process_id:
                total += int(columns[1])
        except ValueError:
            continue
    return total


def build_command(request: WalkForwardJobRequest, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "bnhmm.cli",
        "walk-forward",
        "--duckdb-path",
        request.database_path,
        "--symbols",
        ",".join(request.symbols),
        "--start",
        request.start.isoformat(),
        "--end",
        request.end.isoformat(),
        "--output-dir",
        str(output_dir),
        "--device",
        request.device,
        "--epochs",
        str(request.epochs),
        "--batch-size",
        str(request.batch_size),
        "--learning-rate",
        str(request.learning_rate),
        "--kl-weight",
        str(request.kl_weight),
        "--ordinal-weight",
        str(request.ordinal_weight),
        "--pretrain-epochs",
        str(request.pretrain_epochs),
        "--pretrain-mask-probability",
        str(request.pretrain_mask_probability),
        "--forward-sessions",
        str(request.forward_sessions),
        "--minimum-train-sessions",
        str(request.minimum_train_sessions),
        "--tuning-sessions",
        str(request.tuning_sessions),
        "--test-sessions",
        str(request.test_sessions),
        "--step-sessions",
        str(request.step_sessions),
        "--bin-width-bps",
        str(request.bin_width_bps),
        "--max-profile-bins",
        str(request.max_profile_bins),
        "--normalization-lookback-sessions",
        str(request.normalization_lookback_sessions),
        "--velocity-bins",
        str(request.velocity_bins),
        "--velocity-clip",
        str(request.velocity_clip),
        "--velocity-smoothing",
        str(request.velocity_smoothing),
        "--feature-mode",
        request.feature_mode,
        "--state-timeframe",
        request.state_timeframe,
        "--sequence-length",
        str(request.sequence_length),
        "--sequence-stride",
        str(request.sequence_stride),
        "--minimum-sequence-length",
        str(request.minimum_sequence_length),
        "--d-model",
        str(request.d_model),
        "--latent-states",
        str(request.latent_states),
        "--motif-dim",
        str(request.motif_dim),
        "--heads",
        str(request.heads),
        "--layers",
        str(request.layers),
        "--feedforward-dim",
        str(request.feedforward_dim),
        "--dropout",
        str(request.dropout),
        "--mc-samples",
        str(request.mc_samples),
        "--uncertainty-aversion",
        str(request.uncertainty_aversion),
        "--transaction-cost-bps",
        str(request.transaction_cost_bps),
        "--seed",
        str(request.seed),
    ]
    if request.max_folds is not None:
        command.extend(("--max-folds", str(request.max_folds)))
    if request.deterministic_training:
        command.append("--deterministic-training")
    return command


class JobManager:
    def __init__(self, root: str | Path, project_root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.project_root = Path(project_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobRecord] = {}
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.RLock()
        self._current_id: str | None = None
        self._current_process: subprocess.Popen[str] | None = None
        self._load_existing()
        self._worker = threading.Thread(target=self._work, name="bnhmm-web-worker", daemon=True)
        self._worker.start()

    def submit(self, request: WalkForwardJobRequest) -> JobRecord:
        identifier = f"{utcnow():%Y%m%d-%H%M%S}-{request.name}-{uuid.uuid4().hex[:6]}"
        output_dir = self.root / identifier
        output_dir.mkdir(parents=True, exist_ok=False)
        record = JobRecord(
            id=identifier,
            status="queued",
            created_at=utcnow(),
            output_dir=str(output_dir),
            request=request,
        )
        with self._lock:
            self._jobs[identifier] = record
            self._persist(record)
            self._queue.put(identifier)
        return record.model_copy(deep=True)

    def list(self) -> list[JobRecord]:
        with self._lock:
            return [
                item.model_copy(deep=True)
                for item in sorted(
                    self._jobs.values(), key=lambda row: row.created_at, reverse=True
                )
            ]

    def get(self, identifier: str) -> JobRecord | None:
        with self._lock:
            record = self._jobs.get(identifier)
            return record.model_copy(deep=True) if record else None

    def cancel(self, identifier: str) -> JobRecord | None:
        with self._lock:
            record = self._jobs.get(identifier)
            if record is None or record.status not in {"queued", "running"}:
                return record.model_copy(deep=True) if record else None
            record.status = "cancelled"
            record.finished_at = utcnow()
            self._persist(record)
            if self._current_id == identifier and self._current_process is not None:
                self._current_process.send_signal(signal.SIGTERM)
            return record.model_copy(deep=True)

    def delete(self, identifier: str) -> bool:
        with self._lock:
            record = self._jobs.get(identifier)
            if record is None:
                return False
            if record.status in {"queued", "running"}:
                raise ActiveJobDeletionError("active jobs cannot be deleted")
            output_dir = Path(record.output_dir).resolve()
            if output_dir.parent != self.root or output_dir == self.root:
                raise ValueError("job output directory is outside the storage root")
            if output_dir.exists():
                shutil.rmtree(output_dir)
            del self._jobs[identifier]
            return True

    def log_tail(self, identifier: str, lines: int = 120) -> list[str]:
        record = self.get(identifier)
        if record is None:
            return []
        path = Path(record.output_dir) / "job.log"
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]

    def summary(self, identifier: str) -> dict[str, Any] | None:
        record = self.get(identifier)
        if record is None:
            return None
        path = Path(record.output_dir) / "run-summary.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def artifact_path(self, identifier: str, name: str) -> Path | None:
        record = self.get(identifier)
        if record is None or "/" in name or "\\" in name:
            return None
        candidate = (Path(record.output_dir) / name).resolve()
        if candidate.parent != Path(record.output_dir).resolve() or not candidate.is_file():
            return None
        return candidate

    def symbol_artifact_path(
        self, identifier: str, symbol: str, name: str
    ) -> Path | None:
        record = self.get(identifier)
        if (
            record is None
            or not symbol
            or "/" in symbol
            or "\\" in symbol
            or "/" in name
            or "\\" in name
        ):
            return None
        output_root = Path(record.output_dir).resolve()
        symbols_root = (output_root / "symbols").resolve()
        symbol_root = (symbols_root / symbol).resolve()
        candidate = (symbol_root / name).resolve()
        if symbol_root.parent != symbols_root or candidate.parent != symbol_root:
            return None
        return candidate if candidate.is_file() else None

    def close(self) -> None:
        self._queue.put(None)

    def _work(self) -> None:
        while True:
            identifier = self._queue.get()
            if identifier is None:
                return
            with self._lock:
                record = self._jobs.get(identifier)
                if record is None or record.status != "queued":
                    continue
                record.status = "running"
                record.started_at = utcnow()
                self._current_id = identifier
                self._persist(record)
            output_dir = Path(record.output_dir)
            command = build_command(record.request, output_dir)
            log_path = output_dir / "job.log"
            return_code = -1
            error: str | None = None
            monitor_stop: threading.Event | None = None
            monitor: threading.Thread | None = None
            try:
                environment = os.environ.copy()
                process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                with self._lock:
                    self._current_process = process
                    current = self._jobs[identifier]
                    current.process_id = process.pid
                    current.current_gpu_memory_mib = 0
                    self._persist(current)
                monitor_stop = threading.Event()
                monitor = threading.Thread(
                    target=self._monitor_gpu,
                    args=(identifier, process.pid, monitor_stop),
                    name=f"bnhmm-gpu-{identifier}",
                    daemon=True,
                )
                monitor.start()
                assert process.stdout is not None
                with log_path.open("a", encoding="utf-8") as log:
                    log.write("command: " + " ".join(command) + "\n")
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                return_code = process.wait()
            except Exception as exc:  # pragma: no cover - operating-system failure
                error = str(exc)
            finally:
                if monitor_stop is not None:
                    monitor_stop.set()
                if monitor is not None:
                    monitor.join(timeout=2)
            with self._lock:
                current = self._jobs[identifier]
                if current.status != "cancelled":
                    current.return_code = return_code
                    current.status = (
                        "completed"
                        if return_code == 0 and (output_dir / "run-summary.json").exists()
                        else "failed"
                    )
                    current.error = error or (
                        None if current.status == "completed" else f"process exited {return_code}"
                    )
                    current.finished_at = utcnow()
                current.current_gpu_memory_mib = 0
                self._current_id = None
                self._current_process = None
                self._persist(current)

    def _monitor_gpu(
        self, identifier: str, process_id: int, stop: threading.Event
    ) -> None:
        while not stop.wait(0.5):
            memory = gpu_memory_for_pid(process_id)
            with self._lock:
                record = self._jobs.get(identifier)
                if record is None or record.status != "running":
                    return
                changed = memory != record.current_gpu_memory_mib
                record.current_gpu_memory_mib = memory
                if memory > record.peak_gpu_memory_mib:
                    record.peak_gpu_memory_mib = memory
                    changed = True
                if changed:
                    self._persist(record)

    def _persist(self, record: JobRecord) -> None:
        path = Path(record.output_dir) / "job.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)

    def _load_existing(self) -> None:
        for path in self.root.glob("*/job.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                request_payload = payload.get("request")
                if (
                    isinstance(request_payload, dict)
                    and "state_timeframe" not in request_payload
                ):
                    # Jobs created before daily-only mode always used mixed states.
                    request_payload["state_timeframe"] = "mixed"
                if (
                    isinstance(request_payload, dict)
                    and "profile_protocol" not in request_payload
                ):
                    request_payload["profile_protocol"] = "legacy-fixed-bins"
                if (
                    isinstance(request_payload, dict)
                    and "training_scope" not in request_payload
                ):
                    request_payload["training_scope"] = "shared-legacy"
                if (
                    isinstance(request_payload, dict)
                    and "latent_states" not in request_payload
                ):
                    request_payload["latent_states"] = 10
                if (
                    isinstance(request_payload, dict)
                    and "feature_mode" not in request_payload
                ):
                    request_payload["feature_mode"] = "legacy-entangled"
                record = JobRecord.model_validate(
                    payload, context={"legacy_job": True}
                )
            except Exception:
                continue
            if record.status in {"running", "queued"}:
                record.status = "interrupted"
                record.finished_at = utcnow()
                record.error = "web service restarted before the job completed"
                record.current_gpu_memory_mib = 0
                self._persist(record)
            self._jobs[record.id] = record
