"""FastAPI application for the local BNHMM test console."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..data.sector_dataset import SectorDatasetCatalog
from ..data.source import create_bar_source
from ..visual_dataset import VisualDatasetIndex
from .jobs import (
    ActiveJobDeletionError,
    DuplicateJobNameError,
    JobManager,
    JobRecord,
    WalkForwardJobRequest,
)
from .boundary_data import BoundaryDataBuildManager
from .comparisons import (
    ComparisonManager,
    DuplicateComparisonNameError,
    WindowComparisonRequest,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[2]


def _load_environment_file(path: Path) -> None:
    """Load a small dotenv-compatible file without overriding explicit variables."""

    if not path.is_file():
        return

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            parts = shlex.split(raw_line, comments=True, posix=True)
        except ValueError as exc:
            raise RuntimeError(
                f"invalid environment file entry at {path}:{line_number}"
            ) from exc

        if not parts:
            continue
        if parts[0] == "export":
            parts = parts[1:]
        if len(parts) != 1 or "=" not in parts[0]:
            raise RuntimeError(
                f"invalid environment file entry at {path}:{line_number}"
            )

        key, value = parts[0].split("=", maxsplit=1)
        if (
            not key
            or not key.isascii()
            or key[0].isdigit()
            or not key.replace("_", "").isalnum()
        ):
            raise RuntimeError(
                f"invalid environment variable at {path}:{line_number}"
            )
        os.environ.setdefault(key, value)


_load_environment_file(
    Path(os.getenv("BNHMM_ENV_FILE", str(PROJECT_ROOT / ".env")))
)

STATIC_ROOT = PACKAGE_ROOT / "static"
DEFAULT_JOB_ROOT = "/home/wujianfeng/private/bnhmm-storage/ui-runs"
DEFAULT_SECTOR_DATASET_ROOT = "/home/share/data/sectors"
DEFAULT_VISUAL_DATASET_ROOT = (
    "/home/wujianfeng/private/bnhmm-storage/visual-datasets/semiconductor-2024-2025"
)
DEFAULT_BOUNDARY_CACHE_ROOT = (
    "/home/wujianfeng/private/bnhmm-storage/path-efficiency-data"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    job_root = os.getenv("BNHMM_WEB_JOB_ROOT", DEFAULT_JOB_ROOT)
    manager = JobManager(
        job_root,
        os.getenv("BNHMM_PROJECT_ROOT", str(PROJECT_ROOT)),
    )
    app.state.jobs = manager
    app.state.comparisons = ComparisonManager(job_root, manager)
    app.state.sector_datasets = SectorDatasetCatalog(
        os.getenv("BNHMM_SECTOR_DATASET_ROOT", DEFAULT_SECTOR_DATASET_ROOT)
    )
    app.state.visual_dataset = VisualDatasetIndex(
        os.getenv("BNHMM_VISUAL_DATASET_ROOT", DEFAULT_VISUAL_DATASET_ROOT)
    )
    app.state.boundary_data = BoundaryDataBuildManager(
        os.getenv("BNHMM_BOUNDARY_CACHE_ROOT", DEFAULT_BOUNDARY_CACHE_ROOT),
        os.getenv("BNHMM_PROJECT_ROOT", str(PROJECT_ROOT)),
        sys.executable,
    )
    yield
    manager.close()


app = FastAPI(title="DeepStock Lab", version=__version__, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


@app.middleware("http")
async def disable_ui_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def manager(request: Request) -> JobManager:
    return request.app.state.jobs


def comparison_manager(request: Request) -> ComparisonManager:
    return request.app.state.comparisons


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/visual-dataset", include_in_schema=False)
def visual_dataset_page() -> FileResponse:
    return FileResponse(STATIC_ROOT / "visual.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def config(request: Request) -> dict[str, Any]:
    jobs = manager(request).list()
    datasets = request.app.state.sector_datasets.list()
    defaults = WalkForwardJobRequest().model_dump(mode="json")
    if datasets:
        selected = next(
            (item for item in datasets if item.dataset_id == "semiconductor"),
            datasets[0],
        )
        defaults.update(
            {
                "database_path": selected.database_path,
                "symbols": list(selected.symbols),
                "dataset_id": selected.dataset_id,
            }
        )
    bnhmm_defaults = {
        **defaults,
        "strategy": "bnhmm",
        "training_scope": "per-symbol",
    }
    boundary_defaults = {
        **defaults,
        "strategy": "boundary-flow-fk",
        "training_scope": "sector-pooled",
        "device": "cpu",
        "minimum_train_sessions": 120,
        "regime_neutral_sigma": 0.35,
        "boundary_fk_paths": 10_000,
        "boundary_drift_half_life_sessions": 60.0,
        "boundary_minimum_direction_probability": 0.51,
        "boundary_minimum_signal_to_noise": 0.10,
        "boundary_minimum_survival_probability": 0.35,
        "boundary_relative_strength_lookback_sessions": 20,
        "boundary_maximum_drawdown": 0.05,
    }
    return {
        "devices": _devices(),
        "job_root": str(manager(request).root),
        "active_jobs": sum(item.status in {"queued", "running"} for item in jobs),
        "defaults": bnhmm_defaults,
        "strategy_defaults": {
            "bnhmm": bnhmm_defaults,
            "boundary-flow-fk": boundary_defaults,
        },
        "datasets": [item.as_json() for item in datasets],
    }


@app.get("/api/gpus")
def gpus() -> dict[str, Any]:
    return {"devices": _devices()}


@app.get("/api/symbols")
def symbols(
    database_path: str = Query(os.getenv("BNHMM_MARKET_SOURCE", "/home/share/data/market.duckdb")),
    limit: int = Query(200, ge=1, le=200),
) -> dict[str, list[str]]:
    try:
        values = create_bar_source(database_path).list_symbols(limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"symbols": values}


@app.get("/api/datasets")
def datasets(request: Request) -> dict[str, list[dict[str, object]]]:
    return {
        "datasets": [
            item.as_json() for item in request.app.state.sector_datasets.list()
        ]
    }


@app.get("/api/boundary-data/status/{dataset_id}")
def boundary_data_status(dataset_id: str, request: Request) -> dict[str, Any]:
    if request.app.state.sector_datasets.get(dataset_id) is None:
        raise HTTPException(status_code=404, detail="板块数据集不存在")
    return request.app.state.boundary_data.status(dataset_id)


@app.post("/api/boundary-data/build", status_code=202)
def build_boundary_data(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    dataset_id = str(payload.get("dataset_id", ""))
    dataset = request.app.state.sector_datasets.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="板块数据集不存在")
    try:
        start = date.fromisoformat(str(payload.get("start", "")))
        end = date.fromisoformat(str(payload.get("end", "")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="构造日期无效") from exc
    if end <= start:
        raise HTTPException(status_code=400, detail="结束日期必须晚于开始日期")
    try:
        return request.app.state.boundary_data.submit(
            database_path=dataset.database_path,
            symbols=dataset.symbols,
            dataset_id=dataset.dataset_id,
            start=start.isoformat(),
            end=end.isoformat(),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/visual-dataset/summary")
def visual_dataset_summary(request: Request) -> dict[str, Any]:
    return request.app.state.visual_dataset.summary()


@app.get("/api/visual-dataset/images")
def visual_dataset_images(
    request: Request,
    symbol: str | None = Query(None, min_length=1, max_length=24),
    start: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=12, le=60),
) -> dict[str, Any]:
    result = request.app.state.visual_dataset.query(
        symbol=symbol, start=start, end=end, page=page, page_size=page_size
    )
    for item in result["items"]:
        item["image_url"] = f"/api/visual-dataset/image/{item['symbol']}/{item['session']}"
        item.pop("image_path", None)
    return result


@app.get("/api/visual-dataset/image/{symbol}/{session}")
def visual_dataset_image(symbol: str, session: str, request: Request) -> FileResponse:
    if not symbol.replace(".", "").replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=404, detail="image not found")
    try:
        date.fromisoformat(session)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="image not found") from exc
    path = request.app.state.visual_dataset.image_path(symbol, session)
    if path is None:
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/jobs")
def list_jobs(request: Request) -> list[dict[str, Any]]:
    return [_job_payload(manager(request), record, detail=False) for record in manager(request).list()]


@app.get("/api/comparisons")
def list_comparisons(request: Request) -> list[dict[str, Any]]:
    comparisons = comparison_manager(request)
    return [comparisons.payload(record.id) for record in comparisons.list()]


@app.post("/api/comparisons", status_code=201)
def submit_comparison(
    payload: WindowComparisonRequest, request: Request
) -> dict[str, Any]:
    if payload.request.training_scope == "sector-pooled":
        dataset = request.app.state.sector_datasets.get(
            payload.request.dataset_id or ""
        )
        if dataset is None:
            raise HTTPException(status_code=400, detail="sector dataset not found")
        payload = payload.model_copy(
            update={
                "request": payload.request.model_copy(
                    update={
                        "database_path": dataset.database_path,
                        "symbols": list(dataset.symbols),
                        "dataset_id": dataset.dataset_id,
                    }
                )
            }
        )
    try:
        record = comparison_manager(request).create(payload)
    except (DuplicateComparisonNameError, DuplicateJobNameError) as exc:
        raise HTTPException(status_code=409, detail="对比名称已经存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return comparison_manager(request).payload(record.id)


@app.get("/api/comparisons/{identifier}")
def get_comparison(identifier: str, request: Request) -> dict[str, Any]:
    try:
        return comparison_manager(request).payload(identifier)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="comparison not found") from exc


@app.post("/api/comparisons/{identifier}/missing-windows")
def extend_comparison(identifier: str, request: Request) -> dict[str, Any]:
    comparisons = comparison_manager(request)
    try:
        record = comparisons.extend(identifier)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="comparison not found") from exc
    except (DuplicateJobNameError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return comparisons.payload(record.id)


@app.delete("/api/comparisons/{identifier}")
def delete_comparison(identifier: str, request: Request) -> dict[str, bool]:
    if not comparison_manager(request).delete(identifier):
        raise HTTPException(status_code=404, detail="comparison not found")
    return {"deleted": True}


@app.post("/api/jobs", status_code=201)
def submit_job(payload: WalkForwardJobRequest, request: Request) -> dict[str, Any]:
    if payload.training_scope == "sector-pooled":
        dataset = request.app.state.sector_datasets.get(payload.dataset_id or "")
        if dataset is None:
            raise HTTPException(status_code=400, detail="sector dataset not found")
        payload = payload.model_copy(
            update={
                "database_path": dataset.database_path,
                "symbols": list(dataset.symbols),
                "dataset_id": dataset.dataset_id,
            }
        )
    try:
        record = manager(request).submit(payload)
    except DuplicateJobNameError as exc:
        raise HTTPException(status_code=409, detail="实验名称已存在，请使用其他名称") from exc
    return _job_payload(manager(request), record, detail=True)


@app.get("/api/jobs/{identifier}")
def get_job(identifier: str, request: Request) -> dict[str, Any]:
    record = manager(request).get(identifier)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _job_payload(manager(request), record, detail=True)


@app.post("/api/jobs/{identifier}/cancel")
def cancel_job(identifier: str, request: Request) -> dict[str, Any]:
    record = manager(request).cancel(identifier)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _job_payload(manager(request), record, detail=True)


@app.delete("/api/jobs/{identifier}")
def delete_job(identifier: str, request: Request) -> dict[str, str]:
    try:
        deleted = manager(request).delete(identifier)
    except ActiveJobDeletionError as exc:
        raise HTTPException(status_code=409, detail="运行中的任务不能删除") from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="job not found")
    return {"deleted": identifier}


@app.get("/api/jobs/{identifier}/artifacts/{name}")
def artifact(identifier: str, name: str, request: Request) -> FileResponse:
    path = manager(request).artifact_path(identifier, name)
    if path is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(path)


@app.get("/api/jobs/{identifier}/symbols/{symbol}/artifacts/{name}")
def symbol_artifact(
    identifier: str, symbol: str, name: str, request: Request
) -> FileResponse:
    path = manager(request).symbol_artifact_path(identifier, symbol, name)
    if path is None:
        raise HTTPException(status_code=404, detail="symbol artifact not found")
    return FileResponse(path)






















def _job_payload(store: JobManager, record: JobRecord, *, detail: bool) -> dict[str, Any]:
    payload = record.model_dump(mode="json")
    payload.pop("process_id", None)
    payload["summary"] = store.summary(record.id)
    payload["progress"] = store.progress(record.id)
    payload["artifacts"] = {
        name: f"/api/jobs/{record.id}/artifacts/{name}"
        for name in (
            "equity_drawdown.png",
            "residuals.png",
            "calibration_acf.png",
            "evaluation.json",
            "predictions.json",
            "fk-diagnostics.json",
            "run-summary.json",
            "boundary-daily-states.json",
            "boundary-model-posterior.json",
            "feynman-kac-predictions.json",
            "boundary-policy-evaluation.json",
            "sector-direction-evaluation.json",
            "path-efficiency-fk-report.md",
            "path-efficiency-convection-potential-report.md",
            "convection-potential-evaluation.json",
            "boundary-residual-analysis.json",
            "predictions.csv",
        )
        if store.artifact_path(record.id, name) is not None
    }
    payload["symbol_artifacts"] = {
        symbol: {
            name: f"/api/jobs/{record.id}/symbols/{symbol}/artifacts/{name}"
            for name in (
                "equity_drawdown.png",
                "residuals.png",
                "calibration_acf.png",
                "evaluation.json",
                "predictions.json",
            )
            if store.symbol_artifact_path(record.id, symbol, name) is not None
        }
        for symbol in record.request.symbols
        if store.symbol_artifact_path(record.id, symbol, "evaluation.json") is not None
    }
    if detail:
        payload["log"] = store.log_tail(record.id)
    return payload


def _devices() -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = [{"id": "cpu", "name": "CPU"}]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            index, name, used, total, temperature = [item.strip() for item in line.split(",")]
            devices.append(
                {
                    "id": f"cuda:{index}",
                    "name": name,
                    "memory_used_mib": int(used),
                    "memory_total_mib": int(total),
                    "memory_free_mib": int(total) - int(used),
                    "memory_percent": round(int(used) / max(int(total), 1) * 100, 1),
                    "temperature_c": int(temperature),
                }
            )
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return devices


def run() -> None:
    import uvicorn

    uvicorn.run(
        "bnhmm.web.app:app",
        host=os.getenv("BNHMM_WEB_HOST", "127.0.0.1"),
        port=int(os.getenv("BNHMM_WEB_PORT", "8765")),
        reload=False,
    )


if __name__ == "__main__":
    run()
