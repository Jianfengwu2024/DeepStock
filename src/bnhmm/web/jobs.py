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

from ..config import HOLOGRAPHIC_FIELD_NAMES, HOLOGRAPHIC_LITE_FIELD_NAMES


JobStatus = Literal["queued", "running", "completed", "failed", "cancelled", "interrupted"]
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ActiveJobDeletionError(RuntimeError):
    pass


class DuplicateJobNameError(RuntimeError):
    pass


class WalkForwardJobRequest(BaseModel):
    strategy: Literal[
        "bnhmm",
        "holo-bnhmm",
        "holo-bnhmm-lite",
        "stock-transformer",
        "fk-stock",
    ] = "bnhmm"
    name: str = "experiment"
    database_path: str = "/home/share/data/market.duckdb"
    symbols: list[str] = Field(default_factory=lambda: ["000090", "000099", "000488"])
    dataset_id: str | None = None
    training_scope: Literal[
        "per-symbol", "sector-pooled", "shared-legacy"
    ] = "per-symbol"
    start: date = date(2024, 1, 1)
    end: date = date(2026, 8, 5)
    device: str = "cuda:0"
    epochs: int = Field(10, ge=1, le=1000)
    batch_size: int = Field(8, ge=1, le=512)
    learning_rate: float = Field(3e-4, gt=0, le=1.0)
    kl_weight: float = Field(1e-3, ge=0.0, le=1.0)
    ordinal_weight: float = Field(0.25, ge=0.0, le=10.0)
    symmetry_weight: float = Field(0.2, ge=0.0, le=10.0)
    conservation_weight: float = Field(0.5, ge=0.0, le=10.0)
    pretrain_epochs: int = Field(10, ge=0, le=1000)
    pretrain_mask_probability: float = Field(0.15, gt=0.0, lt=1.0)
    deterministic_training: bool = False
    forward_sessions: int = Field(1, ge=1, le=60)
    strong_threshold_sigma: float = Field(0.75, gt=0.0, le=5.0)
    return_standardization_lookback: int = Field(20, ge=2, le=252)
    return_standardization_min_observations: int = Field(10, ge=2, le=252)
    minimum_return_scale_percent: float = Field(0.2, gt=0.0, le=5.0)
    minimum_train_sessions: int = Field(252, ge=10)
    tuning_sessions: int = Field(60, ge=2)
    test_sessions: int = Field(60, ge=1)
    step_sessions: int = Field(60, ge=1)
    max_folds: int | None = Field(None, ge=1, le=100)
    bin_width_bps: float = Field(10.0, gt=0.0, le=100.0)
    max_profile_bins: int = Field(512, ge=8, le=2048)
    normalization_lookback_sessions: int = Field(10, ge=2, le=120)
    velocity_bins: int = Field(3, ge=3, le=31)
    velocity_aggregation_minutes: int = Field(10, ge=2, le=60)
    velocity_clip: float = Field(4.0, gt=0.0, le=20.0)
    velocity_smoothing: float = Field(0.25, ge=0.0, le=10.0)
    holographic_volume_bins: int = Field(5, ge=3, le=15)
    holographic_smoothing: float = Field(0.75, ge=0.0, le=4.0)
    holographic_fields: tuple[str, ...] = HOLOGRAPHIC_FIELD_NAMES
    holographic_conv_layers: int = Field(2, ge=1, le=2)
    conservation_grid_points: int = Field(64, ge=16, le=512)
    conservation_ridge: float = Field(1e-3, ge=0.0, le=10.0)
    minimum_token_bins: int = Field(2, ge=1, le=64)
    maximum_tokens: int = Field(16, ge=2, le=128)
    boundary_threshold: float = Field(1.5, gt=0.0, le=20.0)
    profile_protocol: Literal[
        "price-volume-holographic-flow-v1",
        "price-volume-holographic-lite-v1",
        "phase-space-10m-local-flow-v8",
        "phase-space-10m-joint-return-v7",
        "phase-space-joint-return-v6",
        "phase-space-joint-v5",
        "phase-space-gated-v4",
        "phase-space-log-v3",
        "variable-log-bins-v2",
        "legacy-fixed-bins",
    ] = "phase-space-10m-local-flow-v8"
    sequence_protocol: Literal[
        "fixed-recent-endpoint-v1", "legacy-overlapping-window-v1"
    ] = "fixed-recent-endpoint-v1"
    state_timeframe: Literal["daily", "mixed"] = "daily"
    sequence_length: int = Field(5, ge=2, le=2048)
    sequence_stride: int = Field(1, ge=1, le=2048)
    minimum_sequence_length: int = Field(2, ge=2, le=2048)
    d_model: int = Field(48, ge=8, le=2048)
    latent_states: int = Field(20, ge=1, le=512)
    transition_rank: int = Field(4, ge=1, le=512)
    feature_mode: Literal[
        "gated-full",
        "price-only",
        "price-volume",
        "price-velocity",
        "legacy-entangled",
    ] = "price-velocity"
    auxiliary_gate_floor: float = Field(0.25, ge=0.0, lt=1.0)
    auxiliary_gate_initial: float = Field(0.75, ge=0.0, lt=1.0)
    auxiliary_warmup_epochs: int = Field(3, ge=0, le=1000)
    motif_dim: int = Field(16, ge=2, le=512)
    heads: int = Field(4, ge=1, le=64)
    layers: int = Field(2, ge=1, le=48)
    feedforward_dim: int = Field(96, ge=8, le=8192)
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

    @field_validator("holographic_fields")
    @classmethod
    def validate_holographic_fields(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if (
            not value
            or value[0] != "density"
            or len(set(value)) != len(value)
            or set(value) - set(HOLOGRAPHIC_FIELD_NAMES)
        ):
            raise ValueError(
                "holographic fields must be unique known fields with density first"
            )
        return value

    @model_validator(mode="after")
    def validate_relationships(self, info: ValidationInfo) -> "WalkForwardJobRequest":
        legacy = bool((info.context or {}).get("legacy_job"))
        if self.end <= self.start:
            raise ValueError("end must be later than start")
        if self.minimum_train_sessions <= self.tuning_sessions:
            raise ValueError("minimum_train_sessions must exceed tuning_sessions")
        if self.minimum_sequence_length > self.sequence_length:
            raise ValueError("minimum_sequence_length cannot exceed sequence_length")
        if self.return_standardization_min_observations > self.return_standardization_lookback:
            raise ValueError("return standardization observations cannot exceed lookback")
        if self.sequence_stride > self.sequence_length:
            raise ValueError("sequence_stride cannot exceed sequence_length")
        if not legacy:
            if self.training_scope not in {"per-symbol", "sector-pooled"}:
                raise ValueError("unsupported current training scope")
            if self.training_scope == "sector-pooled" and not self.dataset_id:
                raise ValueError("sector-pooled jobs require a sector dataset")
            holographic = self.strategy in {"holo-bnhmm", "holo-bnhmm-lite"}
            if self.strategy in {"holo-bnhmm", "holo-bnhmm-lite", "stock-transformer", "fk-stock"} and self.training_scope != "sector-pooled":
                raise ValueError(f"{self.strategy} currently requires a sector dataset")
            expected_profile = (
                (
                    "price-volume-holographic-lite-v1"
                    if self.strategy == "holo-bnhmm-lite"
                    else "price-volume-holographic-flow-v1"
                )
                if holographic
                else "phase-space-10m-local-flow-v8"
            )
            if self.profile_protocol != expected_profile:
                raise ValueError(
                    f"{self.strategy} requires the {expected_profile} profile"
                )
            expected_fields = (
                HOLOGRAPHIC_LITE_FIELD_NAMES
                if self.strategy == "holo-bnhmm-lite"
                else HOLOGRAPHIC_FIELD_NAMES
            )
            if holographic and self.holographic_fields != expected_fields:
                raise ValueError(
                    f"{self.strategy} requires its fixed holographic field set"
                )
            if self.sequence_protocol != "fixed-recent-endpoint-v1":
                raise ValueError("current jobs require fixed recent endpoint windows")
            if self.state_timeframe != "daily":
                raise ValueError("current jobs require daily states built from one-minute bars")
            if self.forward_sessions != 1:
                raise ValueError("current jobs predict exactly the next trading session")
            if self.sequence_stride != 1:
                raise ValueError("current jobs require a rolling stride of one")
            if self.sequence_length > 20:
                raise ValueError("recent context cannot exceed twenty trading states")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if self.velocity_bins % 2 == 0:
            raise ValueError("velocity_bins must be odd")
        if self.holographic_volume_bins % 2 == 0:
            raise ValueError("holographic_volume_bins must be odd")
        if 60 % self.velocity_aggregation_minutes:
            raise ValueError("velocity_aggregation_minutes must divide 60")
        profile_channels = (
            self.holographic_volume_bins * len(self.holographic_fields)
            if self.strategy in {"holo-bnhmm", "holo-bnhmm-lite"}
            else 4 + self.velocity_bins**2
        )
        if self.feature_mode == "legacy-entangled" and not legacy:
            raise ValueError("legacy-entangled is reserved for historical jobs")
        if (
            not legacy
            and self.strategy in {"bnhmm", "holo-bnhmm", "holo-bnhmm-lite"}
            and self.latent_states < profile_channels
        ):
            raise ValueError("latent_states cannot be fewer than profile channels")
        if (
            not legacy
            and self.strategy in {"bnhmm", "holo-bnhmm", "holo-bnhmm-lite"}
            and self.transition_rank > self.latent_states
        ):
            raise ValueError("transition_rank cannot exceed latent_states")
        if self.auxiliary_gate_initial < self.auxiliary_gate_floor:
            raise ValueError("auxiliary gate initial cannot be below its floor")
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
    if request.strategy == "stock-transformer":
        return _build_stock_transformer_command(request, output_dir)
    if request.strategy == "fk-stock":
        return _build_fk_stock_command(request, output_dir)
    command = [
        sys.executable,
        "-m",
        "bnhmm.cli",
        "walk-forward",
        "--duckdb-path",
        request.database_path,
        "--symbols",
        ",".join(request.symbols),
        "--training-scope",
        request.training_scope,
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
        "--symmetry-weight",
        str(request.symmetry_weight),
        "--pretrain-epochs",
        str(request.pretrain_epochs),
        "--pretrain-mask-probability",
        str(request.pretrain_mask_probability),
        "--forward-sessions",
        str(request.forward_sessions),
        "--strong-threshold-sigma",
        str(request.strong_threshold_sigma),
        "--return-standardization-lookback",
        str(request.return_standardization_lookback),
        "--return-standardization-min-observations",
        str(request.return_standardization_min_observations),
        "--minimum-return-scale-percent",
        str(request.minimum_return_scale_percent),
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
        "--velocity-aggregation-minutes",
        str(request.velocity_aggregation_minutes),
        "--velocity-clip",
        str(request.velocity_clip),
        "--velocity-smoothing",
        str(request.velocity_smoothing),
        "--feature-mode",
        request.feature_mode,
        "--auxiliary-gate-floor",
        str(request.auxiliary_gate_floor),
        "--auxiliary-gate-initial",
        str(request.auxiliary_gate_initial),
        "--auxiliary-warmup-epochs",
        str(request.auxiliary_warmup_epochs),
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
        "--transition-rank",
        str(request.transition_rank),
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
    if request.strategy in {"holo-bnhmm", "holo-bnhmm-lite"}:
        command.extend(
            (
                "--profile-kind",
                "holographic",
                "--holographic-volume-bins",
                str(request.holographic_volume_bins),
                "--holographic-smoothing",
                str(request.holographic_smoothing),
                "--holographic-fields",
                ",".join(request.holographic_fields),
                "--holographic-conv-layers",
                str(request.holographic_conv_layers),
            )
        )
    if request.dataset_id:
        command.extend(("--dataset-id", request.dataset_id))
    if request.max_folds is not None:
        command.extend(("--max-folds", str(request.max_folds)))
    if request.deterministic_training:
        command.append("--deterministic-training")
    return command


def _build_stock_transformer_command(
    request: WalkForwardJobRequest, output_dir: Path
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "bnhmm.cli",
        "stock-transformer-walk-forward",
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
        "--conservation-weight",
        str(request.conservation_weight),
        "--strong-threshold-sigma",
        str(request.strong_threshold_sigma),
        "--return-standardization-lookback",
        str(request.return_standardization_lookback),
        "--return-standardization-min-observations",
        str(request.return_standardization_min_observations),
        "--minimum-return-scale-percent",
        str(request.minimum_return_scale_percent),
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
        "--conservation-grid-points",
        str(request.conservation_grid_points),
        "--conservation-ridge",
        str(request.conservation_ridge),
        "--minimum-token-bins",
        str(request.minimum_token_bins),
        "--maximum-tokens",
        str(request.maximum_tokens),
        "--boundary-threshold",
        str(request.boundary_threshold),
        "--sequence-length",
        str(request.sequence_length),
        "--d-model",
        str(request.d_model),
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
    if request.dataset_id:
        command.extend(("--dataset-id", request.dataset_id))
    if request.max_folds is not None:
        command.extend(("--max-folds", str(request.max_folds)))
    return command


def _build_fk_stock_command(
    request: WalkForwardJobRequest, output_dir: Path
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "bnhmm.cli",
        "fk-stock-walk-forward",
        "--duckdb-path",
        request.database_path,
        "--symbols",
        ",".join(request.symbols),
        "--dataset-id",
        str(request.dataset_id),
        "--start",
        request.start.isoformat(),
        "--end",
        request.end.isoformat(),
        "--output-dir",
        str(output_dir),
        "--strong-threshold-sigma",
        str(request.strong_threshold_sigma),
        "--return-standardization-lookback",
        str(request.return_standardization_lookback),
        "--return-standardization-min-observations",
        str(request.return_standardization_min_observations),
        "--minimum-return-scale-percent",
        str(request.minimum_return_scale_percent),
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
        "--uncertainty-aversion",
        str(request.uncertainty_aversion),
        "--transaction-cost-bps",
        str(request.transaction_cost_bps),
    ]
    if request.max_folds is not None:
        command.extend(("--max-folds", str(request.max_folds)))
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
        with self._lock:
            normalized_name = request.name.casefold()
            if any(
                item.request.name.casefold() == normalized_name
                for item in self._jobs.values()
            ):
                raise DuplicateJobNameError(
                    f"job name already exists: {request.name}"
                )
            identifier = (
                f"{utcnow():%Y%m%d-%H%M%S}-{request.name}-{uuid.uuid4().hex[:6]}"
            )
            output_dir = self.root / identifier
            output_dir.mkdir(parents=True, exist_ok=False)
            record = JobRecord(
                id=identifier,
                status="queued",
                created_at=utcnow(),
                output_dir=str(output_dir),
                request=request,
            )
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
                    and "sequence_protocol" not in request_payload
                ):
                    request_payload["sequence_protocol"] = (
                        "legacy-overlapping-window-v1"
                    )
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
