"""FastAPI application for the local BNHMM test console."""

from __future__ import annotations

import os
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..data.source import DuckDBBarSource
from .jobs import (
    ActiveJobDeletionError,
    DuplicateJobNameError,
    JobManager,
    JobRecord,
    WalkForwardJobRequest,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[2]
STATIC_ROOT = PACKAGE_ROOT / "static"
DEFAULT_JOB_ROOT = "/home/wujianfeng/private/bnhmm-storage/ui-runs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = JobManager(
        os.getenv("BNHMM_WEB_JOB_ROOT", DEFAULT_JOB_ROOT),
        os.getenv("BNHMM_PROJECT_ROOT", str(PROJECT_ROOT)),
    )
    app.state.jobs = manager
    yield
    manager.close()


app = FastAPI(title="BNHMM Lab", version=__version__, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


@app.middleware("http")
async def disable_ui_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def manager(request: Request) -> JobManager:
    return request.app.state.jobs


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def config(request: Request) -> dict[str, Any]:
    jobs = manager(request).list()
    return {
        "devices": _devices(),
        "job_root": str(manager(request).root),
        "active_jobs": sum(item.status in {"queued", "running"} for item in jobs),
        "defaults": WalkForwardJobRequest().model_dump(mode="json"),
    }


@app.get("/api/gpus")
def gpus() -> dict[str, Any]:
    return {"devices": _devices()}


@app.get("/api/symbols")
def symbols(
    database_path: str = Query("/home/share/data/market.duckdb"),
    limit: int = Query(200, ge=1, le=200),
) -> dict[str, list[str]]:
    try:
        values = DuckDBBarSource(database_path).list_symbols(limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"symbols": values}


@app.get("/api/jobs")
def list_jobs(request: Request) -> list[dict[str, Any]]:
    return [_job_payload(manager(request), record, detail=False) for record in manager(request).list()]


@app.post("/api/jobs", status_code=201)
def submit_job(payload: WalkForwardJobRequest, request: Request) -> dict[str, Any]:
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
    payload["artifacts"] = {
        name: f"/api/jobs/{record.id}/artifacts/{name}"
        for name in (
            "equity_drawdown.png",
            "residuals.png",
            "calibration_acf.png",
            "evaluation.json",
            "predictions.json",
            "run-summary.json",
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
        host=os.getenv("BNHMM_WEB_HOST", "0.0.0.0"),
        port=int(os.getenv("BNHMM_WEB_PORT", "8765")),
        reload=False,
    )


if __name__ == "__main__":
    run()
