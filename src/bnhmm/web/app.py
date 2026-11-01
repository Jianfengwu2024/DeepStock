"""FastAPI application for the local BNHMM test console."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request as UrlRequest, urlopen
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..data.local_dataset import LocalMarketDatasetManager
from ..data.sector_dataset import SectorDatasetCatalog
from ..data.source import create_bar_source
from ..boundary_fk.cache import BoundaryDataCacheKey, BoundaryDataCacheMiss, SECTOR_CACHE_SLOT
from ..visual_dataset import VisualDatasetIndex
from .jobs import (
    ActiveJobDeletionError,
    DuplicateJobNameError,
    JobManager,
    JobRecord,
    WalkForwardJobRequest,
)
from .local_store import StockPoolStore
from .boundary_data import BoundaryDataBuildManager
from .cloud_strategy import CloudStrategyWorker
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
    resource_lock = threading.Lock()
    manager = JobManager(
        job_root,
        os.getenv("BNHMM_PROJECT_ROOT", str(PROJECT_ROOT)),
        resource_lock=resource_lock,
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
        resource_lock=resource_lock,
    )
    app.state.stock_pools = StockPoolStore(
        os.getenv("BNHMM_STOCK_POOL_PATH", str(Path(job_root).parent / "stock-pools.sqlite3"))
    )
    app.state.market_datasets = LocalMarketDatasetManager(
        os.getenv("BNHMM_MARKET_DATASET_ROOT", str(Path(job_root).parent / "market-datasets")),
        os.getenv("BNHMM_MARKET_SOURCE", "http://127.0.0.1:9000/api/v1/db"),
        os.getenv("BNHMM_MARKET_BROKER_TOKEN", os.getenv("QUANT_DB_BROKER_TOKEN", "")),
        resource_lock=resource_lock,
        expiration_cleanup=app.state.boundary_data.delete,
    )
    app.state.market_datasets.set_dataset_busy_probe(
        lambda dataset_id: any(
            job.status in {"queued", "running"}
            and (job.request.market_dataset_id or job.request.dataset_id) == dataset_id
            for job in app.state.jobs.list()
        )
    )
    app.state.cloud_fk = CloudStrategyWorker(
        server=os.getenv("BNHMM_QUANT_SERVER", "https://quant.hapgetaquant.com"),
        jobs=manager,
        boundary_data=app.state.boundary_data,
        market_datasets=app.state.market_datasets,
        stock_pools=app.state.stock_pools,
    )
    yield
    app.state.cloud_fk.disconnect()
    app.state.market_datasets.close()
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
    local_manager = getattr(request.app.state, "market_datasets", None)
    local_datasets = local_manager.store.list() if local_manager else []
    local_available = next(
        (item for item in local_datasets if item["status"] in {"ready", "partial"} and "1m" in item["intervals"]),
        None,
    )
    if local_available:
        data_start = date.fromisoformat(local_available["start_date"])
        data_end = date.fromisoformat(local_available["last_refresh_date"] or local_available["start_date"])
        common_defaults = {
            "database_path": local_available["database_path"],
            "symbols": local_available["symbols"],
            "market_dataset_id": local_available["dataset_id"],
            "end": (data_end + timedelta(days=1)).isoformat(),
        }
        bnhmm_defaults.update({**common_defaults, "start": data_start.isoformat()})
        # FK cache construction requires 450 calendar days before evaluation.
        boundary_start = max(data_start + timedelta(days=450), data_end - timedelta(days=365))
        if boundary_start <= data_end:
            boundary_defaults.update({**common_defaults, "start": boundary_start.isoformat()})
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
        "market_datasets": local_datasets,
    }


@app.get("/api/gpus")
def gpus() -> dict[str, Any]:
    return {"devices": _devices()}


@app.get("/api/data-service")
def data_service_status(request: Request) -> dict[str, Any]:
    source = os.getenv("BNHMM_MARKET_SOURCE", "")
    remote = source.lower().startswith(("http://", "https://"))
    result: dict[str, Any] = {
        "source": source,
        "source_type": "broker" if remote else "duckdb",
        "read_only": True,
        "status": "unknown",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if remote:
            payload = create_bar_source(source, timeout_seconds=8).query("SELECT 1 AS connected")
            result["status"] = "connected" if payload.get("rows") == [[1]] else "error"
        else:
            path = Path(source).expanduser()
            result["status"] = "connected" if path.is_file() else "missing"
            result["database_size_bytes"] = path.stat().st_size if path.is_file() else None
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
    result["devices"] = _devices()
    result["dataset_count"] = len(request.app.state.sector_datasets.list())
    result["local_market_dataset_count"] = len(request.app.state.market_datasets.store.list())
    return result


class StockPoolRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class StockPoolSymbolRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=24)


class CreateMarketDatasetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    stock_pool_id: str = Field(min_length=1, max_length=80)
    intervals: list[str] = Field(min_length=1, max_length=7)
    start_date: date


class AiModelsRequest(BaseModel):
    base_url: str = Field(min_length=8, max_length=500)
    api_key: str = Field(min_length=1, max_length=4096)


class QuantLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


@app.get("/api/stock-pools")
def list_stock_pools(request: Request) -> list[dict[str, Any]]:
    return request.app.state.stock_pools.list()


@app.post("/api/stock-pools", status_code=201)
def create_stock_pool(payload: StockPoolRequest, request: Request) -> dict[str, Any]:
    try:
        return request.app.state.stock_pools.create(payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.delete("/api/stock-pools/{pool_id}")
def delete_stock_pool(pool_id: str, request: Request) -> dict[str, bool]:
    if not request.app.state.stock_pools.delete(pool_id):
        raise HTTPException(status_code=404, detail="股票池不存在")
    return {"deleted": True}


@app.post("/api/stock-pools/{pool_id}/symbols")
def add_stock_to_pool(
    pool_id: str, payload: StockPoolSymbolRequest, request: Request
) -> dict[str, bool]:
    try:
        if not request.app.state.stock_pools.add_symbol(pool_id, payload.symbol):
            raise HTTPException(status_code=404, detail="股票池不存在")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"added": True}


@app.delete("/api/stock-pools/{pool_id}/symbols/{symbol}")
def remove_stock_from_pool(pool_id: str, symbol: str, request: Request) -> dict[str, bool]:
    if not request.app.state.stock_pools.remove_symbol(pool_id, symbol):
        raise HTTPException(status_code=404, detail="股票池不存在")
    return {"deleted": True}


@app.post("/api/ai/models")
def check_ai_models(payload: AiModelsRequest) -> dict[str, Any]:
    parsed = urlsplit(payload.base_url)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username or parsed.password or parsed.query or parsed.fragment
        or (parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"})
    ):
        raise HTTPException(status_code=422, detail="请使用 HTTPS 的 OpenAI 兼容 API 地址")
    base = payload.base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if base.endswith("/models"):
        base = base[: -len("/models")]
    request = UrlRequest(
        base + "/models",
        headers={"Authorization": "Bearer " + payload.api_key, "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=15) as response:
            data = json.loads(response.read(2_000_001))
    except HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"模型服务返回 HTTP {exc.code}") from None
    except (URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="无法连接模型服务或返回内容无效") from exc
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list) or len(models) > 5000:
        raise HTTPException(status_code=502, detail="模型服务未返回兼容的 models 列表")
    names = sorted({str(item["id"]) for item in models if isinstance(item, dict) and isinstance(item.get("id"), str)})
    return {"models": names, "count": len(names)}


@app.post("/api/verification/login")
def verify_quant_account(payload: QuantLoginRequest) -> dict[str, Any]:
    server = os.getenv("BNHMM_QUANT_SERVER", "https://quant.hapgetaquant.com").rstrip("/")
    parsed = urlsplit(server)
    if parsed.scheme != "https" or not parsed.hostname or parsed.path not in {"", "/"}:
        raise HTTPException(status_code=503, detail="Quant Platform 验证服务地址无效")
    body = json.dumps({"username": payload.username, "password": payload.password}).encode()
    login_request = UrlRequest(
        server + "/api/v1/auth/login", body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(login_request, timeout=20) as response:
            session = json.loads(response.read(1_000_001))
    except HTTPError as exc:
        raise HTTPException(status_code=exc.code if exc.code in {401, 403} else 502, detail="账号验证失败") from None
    except (URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="暂时无法连接 Quant Platform") from exc
    token = session.get("access_token") if isinstance(session, dict) else None
    user = session.get("user") if isinstance(session, dict) else None
    if not isinstance(token, str) or not isinstance(user, dict):
        raise HTTPException(status_code=502, detail="Quant Platform 返回的登录数据无效")
    authorized, permission_status, catalog_count = _deepstock_permission(server, token)
    return {
        "verified": True,
        "authorized": authorized,
        "permission_status": permission_status,
        "user": user,
        "catalog_count": catalog_count,
        "access_token": token,
    }


class QuantCatalogRequest(BaseModel):
    access_token: str = Field(min_length=16, max_length=4096)


@app.post("/api/cloud-fk/connect")
def connect_cloud_fk(payload: QuantCatalogRequest, request: Request) -> dict[str, Any]:
    try:
        return request.app.state.cloud_fk.connect(payload.access_token)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="无法连接 Quant Platform 本机策略执行服务") from exc


@app.get("/api/cloud-fk/status")
def cloud_fk_status(request: Request) -> dict[str, Any]:
    return request.app.state.cloud_fk.status()


@app.post("/api/cloud-fk/disconnect")
def disconnect_cloud_fk(request: Request) -> dict[str, Any]:
    return request.app.state.cloud_fk.disconnect()


class PublishCloudFkRequest(BaseModel):
    sector: str = Field(pattern=r"^(semiconductor|artificial_intelligence|new_energy)$")
    access_token: str | None = Field(default=None, min_length=16, max_length=4096)


@app.post("/api/cloud-fk/publish/{identifier}")
def publish_cloud_fk(identifier: str, payload: PublishCloudFkRequest, request: Request) -> dict[str, Any]:
    try:
        result = request.app.state.cloud_fk.publish_saved_run(identifier, payload.sector, token=payload.access_token)
        return {"job_id": result["job_id"], "status": result["status"], "source_run_id": identifier}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


class QuantStockPoolRequest(QuantCatalogRequest):
    pool_id: str | None = Field(default=None, min_length=1, max_length=128)


@app.post("/api/verification/session")
def verify_quant_session(payload: QuantCatalogRequest) -> dict[str, Any]:
    server = _quant_server_origin()
    request = UrlRequest(
        server + "/api/v1/auth/me",
        headers={"Authorization": "Bearer " + payload.access_token, "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=20) as response:
            data = json.loads(response.read(1_000_001))
    except HTTPError as exc:
        raise HTTPException(status_code=401 if exc.code in {401, 403} else 502, detail="登录已过期，请重新验证账号") from None
    except (URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="暂时无法连接 Quant Platform") from exc
    user = data.get("user") if isinstance(data, dict) else None
    if not isinstance(user, dict):
        raise HTTPException(status_code=502, detail="Quant Platform 返回的用户信息无效")
    authorized, permission_status, catalog_count = _deepstock_permission(server, payload.access_token)
    return {"user": user, "authorized": authorized, "permission_status": permission_status, "catalog_count": catalog_count}


def _quant_server_origin() -> str:
    server = os.getenv("BNHMM_QUANT_SERVER", "https://quant.hapgetaquant.com").rstrip("/")
    parsed = urlsplit(server)
    if parsed.scheme != "https" or not parsed.hostname or parsed.path not in {"", "/"}:
        raise HTTPException(status_code=503, detail="Quant Platform 验证服务地址无效")
    return server


def _deepstock_permission(server: str, access_token: str) -> tuple[bool, str, int]:
    request = UrlRequest(server + "/api/v1/deepstock/catalog", headers={
        "Authorization": "Bearer " + access_token,
        "Accept": "application/json",
    })
    try:
        with urlopen(request, timeout=20) as response:
            catalog = json.loads(response.read(2_000_001))
        return True, "granted", int(catalog.get("total", 0))
    except HTTPError as exc:
        if exc.code == 403:
            return False, "denied", 0
        return False, "unavailable", 0
    except (URLError, TimeoutError, ValueError):
        return False, "unavailable", 0


def _fetch_quant_stock_pools(access_token: str) -> list[dict[str, Any]]:
    server = _quant_server_origin()
    request = UrlRequest(
        server + "/api/v1/screener/pools",
        headers={"Authorization": "Bearer " + access_token, "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read(4_000_001))
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise HTTPException(status_code=exc.code, detail="线上 AI 股票池验证失败或权限不足") from None
        raise HTTPException(status_code=502, detail=f"Quant Platform 股票池服务返回 HTTP {exc.code}") from None
    except (URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="无法连接 Quant Platform AI 股票池") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    pools = data.get("pools") if isinstance(data, dict) else None
    if not isinstance(pools, list) or len(pools) > 5000:
        raise HTTPException(status_code=502, detail="Quant Platform 返回的股票池列表格式无效")
    result: list[dict[str, Any]] = []
    for pool in pools:
        if not isinstance(pool, dict):
            continue
        pool_id = pool.get("pool_id")
        name = pool.get("name")
        raw_symbols = pool.get("symbols")
        if not isinstance(pool_id, str) or not isinstance(name, str) or not isinstance(raw_symbols, list):
            continue
        symbols = list(dict.fromkeys(str(item).strip()[-6:] for item in raw_symbols if str(item).strip()))
        if not symbols or len(symbols) > 20_000 or any(not re.fullmatch(r"\d{6}", item) for item in symbols):
            continue
        result.append({
            "pool_id": pool_id,
            "name": name[:80],
            "symbols": symbols,
            "query": str(pool.get("query") or "")[:1000],
            "description": str(pool.get("description") or pool.get("query") or "")[:1000],
            "run_id": pool.get("run_id") if isinstance(pool.get("run_id"), str) else None,
            "source": pool.get("source") if isinstance(pool.get("source"), str) else "normal",
            "conditions": pool.get("conditions") if isinstance(pool.get("conditions"), list) else [],
            "created_at": pool.get("created_at") if isinstance(pool.get("created_at"), str) else None,
        })
    return result


def _fetch_quant_watchlist(access_token: str) -> list[dict[str, Any]]:
    request = UrlRequest(
        _quant_server_origin() + "/api/v1/watchlist",
        headers={"Authorization": "Bearer " + access_token, "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read(4_000_001))
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise HTTPException(status_code=exc.code, detail="线上自选股验证失败或权限不足") from None
        raise HTTPException(status_code=502, detail=f"Quant Platform 自选股服务返回 HTTP {exc.code}") from None
    except (URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="无法连接 Quant Platform 自选股") from exc
    if isinstance(payload, dict) and payload.get("success") is False:
        raise HTTPException(status_code=502, detail="Quant Platform 自选股读取失败")
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) > 20_000:
        raise HTTPException(status_code=502, detail="Quant Platform 返回的自选股列表格式无效")
    symbols = []
    notes = []
    added_at = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip()[-6:]
        if not re.fullmatch(r"\d{6}", symbol):
            continue
        symbols.append(symbol)
        if row.get("note"):
            notes.append(f"{symbol}: {str(row['note'])[:160]}")
        if row.get("added_at"):
            added_at.append(str(row["added_at"]))
    if not symbols:
        return []
    return [{
        "pool_id": "quant-watchlist-main",
        "name": "我的自选股",
        "symbols": list(dict.fromkeys(symbols)),
        "description": "Quant Platform 用户自选股。" + ("\n" + "\n".join(notes[:200]) if notes else ""),
        "query": "用户自选股",
        "run_id": None,
        "source": "quant-watchlist",
        "created_at": min(added_at) if added_at else "",
    }]


@app.post("/api/verification/catalog")
def read_verified_catalog(payload: QuantCatalogRequest) -> dict[str, Any]:
    server = _quant_server_origin()
    request = UrlRequest(server + "/api/v1/deepstock/catalog", headers={
        "Authorization": "Bearer " + payload.access_token,
        "Accept": "application/json",
    })
    try:
        with urlopen(request, timeout=20) as response:
            catalog = json.loads(response.read(2_000_001))
    except HTTPError as exc:
        raise HTTPException(status_code=exc.code if exc.code in {401, 403} else 502, detail="目录访问资格已失效或权限不足") from None
    except (URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="无法读取 DeepStock 目录") from exc
    return catalog


@app.post("/api/verification/stock-pools")
def list_verified_quant_stock_pools(payload: QuantCatalogRequest) -> dict[str, Any]:
    pools = _fetch_quant_stock_pools(payload.access_token)
    return {"pools": pools, "total": len(pools), "source": "quant-platform-ai-screener"}


@app.post("/api/verification/watchlist")
def list_verified_quant_watchlist(payload: QuantCatalogRequest) -> dict[str, Any]:
    items = _fetch_quant_watchlist(payload.access_token)
    return {"items": items, "total": len(items), "source": "quant-platform-watchlist"}


@app.post("/api/stock-pools/sync-quant-ai")
def sync_quant_ai_stock_pool(
    payload: QuantStockPoolRequest, request: Request
) -> dict[str, Any]:
    if payload.pool_id is None:
        raise HTTPException(status_code=422, detail="pool_id is required")
    pools = _fetch_quant_stock_pools(payload.access_token)
    pool = next((item for item in pools if item["pool_id"] == payload.pool_id), None)
    if pool is None:
        raise HTTPException(status_code=404, detail="线上 AI 股票池不存在或已删除")
    try:
        result = request.app.state.stock_pools.sync_cloud_pool(pool)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"pool": result, "source": "quant-platform-ai-screener"}


@app.post("/api/stock-pools/sync-quant-watchlist")
def sync_quant_watchlist(
    payload: QuantStockPoolRequest, request: Request
) -> dict[str, Any]:
    if payload.pool_id is None:
        raise HTTPException(status_code=422, detail="pool_id is required")
    items = _fetch_quant_watchlist(payload.access_token)
    item = next((row for row in items if row["pool_id"] == payload.pool_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="线上自选股不存在或已移除")
    try:
        result = request.app.state.stock_pools.sync_cloud_pool(item)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"pool": result, "source": "quant-platform-watchlist"}


@app.get("/api/symbols")
def symbols(
    database_path: str = Query(os.getenv("BNHMM_MARKET_SOURCE", "/home/share/data/market.duckdb")),
    limit: int = Query(200, ge=1, le=200),
    q: str | None = Query(None, min_length=1, max_length=24),
) -> dict[str, list[str]]:
    try:
        values = create_bar_source(database_path).list_symbols(limit, q=q)
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


@app.get("/api/market-datasets")
def list_market_datasets(request: Request) -> list[dict[str, Any]]:
    return request.app.state.market_datasets.store.list()


@app.get("/api/strategy-datasets")
def list_strategy_datasets(request: Request) -> list[dict[str, Any]]:
    """List durable precomputed strategy inputs kept by the local service."""
    result: list[dict[str, Any]] = []
    for dataset in request.app.state.market_datasets.store.list():
        if dataset["status"] not in {"ready", "partial"} or "1m" not in dataset["intervals"]:
            continue
        result.append(request.app.state.boundary_data.describe(dataset))
    return result


@app.get("/api/strategy-datasets/{dataset_id}")
def describe_strategy_dataset(
    dataset_id: str,
    request: Request,
    start: date | None = Query(None),
    end: date | None = Query(None),
) -> dict[str, Any]:
    dataset = request.app.state.market_datasets.store.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="本地行情数据集不存在")
    if dataset["status"] not in {"ready", "partial"} or "1m" not in dataset["intervals"]:
        raise HTTPException(status_code=409, detail="本地分钟行情尚未就绪")
    if (start is None) != (end is None):
        raise HTTPException(status_code=422, detail="开始与结束日期必须同时填写")
    if start is not None and end is not None and end <= start:
        raise HTTPException(status_code=422, detail="结束日期必须晚于开始日期")
    try:
        return request.app.state.boundary_data.describe(
            dataset, start=start.isoformat() if start else None,
            end=end.isoformat() if end else None,
        )
    except (OSError, ValueError, KeyError) as exc:
        return {
            "dataset_id": dataset_id, "market_dataset_id": dataset_id,
            "name": "FK-贝叶斯 · " + dataset["name"],
            "status": "stale", "reason": str(exc),
            "persistent": True, "run_mode": "reuse_only",
        }


@app.delete("/api/strategy-datasets/{dataset_id}")
def delete_strategy_dataset(dataset_id: str, request: Request) -> dict[str, bool]:
    try:
        request.app.state.boundary_data.delete(dataset_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deleted": True}


@app.post("/api/market-datasets", status_code=202)
def create_market_dataset(payload: CreateMarketDatasetRequest, request: Request) -> dict[str, Any]:
    pool = next(
        (item for item in request.app.state.stock_pools.list() if item["id"] == payload.stock_pool_id),
        None,
    )
    if pool is None:
        raise HTTPException(status_code=404, detail="股票池不存在")
    if not pool["symbols"]:
        raise HTTPException(status_code=422, detail="不能从空股票池创建数据集")
    try:
        return request.app.state.market_datasets.create(
            name=payload.name,
            symbols=pool["symbols"],
            intervals=payload.intervals,
            start_date=payload.start_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/market-datasets/{dataset_id}/refresh", status_code=202)
def refresh_market_dataset(dataset_id: str, request: Request) -> dict[str, Any]:
    try:
        return request.app.state.market_datasets.refresh(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="本地数据集不存在") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.delete("/api/market-datasets/{dataset_id}")
def delete_market_dataset(dataset_id: str, request: Request) -> dict[str, bool]:
    try:
        if not request.app.state.market_datasets.delete(dataset_id):
            raise HTTPException(status_code=404, detail="本地数据集不存在")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"deleted": True}


@app.get("/api/boundary-data/status/{dataset_id}")
def boundary_data_status(dataset_id: str, request: Request) -> dict[str, Any]:
    local_dataset = request.app.state.market_datasets.store.get(dataset_id)
    sector_dataset = request.app.state.sector_datasets.get(dataset_id)
    if local_dataset is None and sector_dataset is None:
        raise HTTPException(status_code=404, detail="板块数据集不存在")
    if local_dataset is not None:
        start_raw = request.query_params.get("start")
        end_raw = request.query_params.get("end")
        try:
            return request.app.state.boundary_data.describe(
                local_dataset,
                start=start_raw,
                end=date.fromisoformat(end_raw).isoformat() if end_raw else None,
            )
        except (OSError, ValueError, KeyError) as exc:
            return {
                "dataset_id": dataset_id, "status": "stale",
                "reason": str(exc), "persistent": True, "run_mode": "reuse_only",
                "cache": request.app.state.boundary_data.cache.status(dataset_id),
            }
    return request.app.state.boundary_data.status(dataset_id)


@app.post("/api/boundary-data/build", status_code=202)
def build_boundary_data(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    dataset_id = str(payload.get("dataset_id", ""))
    local_dataset = request.app.state.market_datasets.store.get(dataset_id)
    sector_dataset = request.app.state.sector_datasets.get(dataset_id)
    if local_dataset is None and sector_dataset is None:
        raise HTTPException(status_code=404, detail="数据集不存在")
    try:
        start = date.fromisoformat(str(payload.get("start", "")))
        end = date.fromisoformat(str(payload.get("end", "")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="构造日期无效") from exc
    if end <= start:
        raise HTTPException(status_code=400, detail="结束日期必须晚于开始日期")
    if local_dataset and (
        local_dataset["status"] not in {"ready", "partial"}
        or any(not local_dataset["coverage"].get("1m", {}).get(symbol, {}).get("rows") for symbol in local_dataset["symbols"])
    ):
        raise HTTPException(status_code=409, detail="所选标的的本地分钟数据尚未就绪")
    try:
        database_path = local_dataset["database_path"] if local_dataset else sector_dataset.database_path
        symbols = tuple(local_dataset["symbols"]) if local_dataset else sector_dataset.symbols
        result = request.app.state.boundary_data.submit(
            database_path=database_path,
            symbols=symbols,
            dataset_id=dataset_id,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        return {**result, "run_mode": "reuse_only"}
    except (RuntimeError, ValueError, OSError) as exc:
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
    if payload.strategy == "boundary-flow-fk" or payload.market_dataset_id:
        dataset = request.app.state.market_datasets.store.get(payload.market_dataset_id or "")
        if dataset is None:
            raise HTTPException(status_code=400, detail="请选择一个已创建的本地行情数据集")
        if dataset["status"] not in {"ready", "partial"} or "1m" not in dataset["intervals"]:
            raise HTTPException(status_code=409, detail="本地数据集尚未完成分钟线回补")
        if not set(payload.symbols).issubset(dataset["symbols"]):
            raise HTTPException(status_code=422, detail="测试标的必须来自所选本地数据集")
        if any(not dataset["coverage"].get("1m", {}).get(symbol, {}).get("rows") for symbol in payload.symbols):
            raise HTTPException(status_code=409, detail="测试标的缺少本地一分钟行情，请先完成数据回补")
        if payload.strategy == "boundary-flow-fk":
            if tuple(payload.symbols) != tuple(dataset["symbols"]):
                raise HTTPException(status_code=422, detail="FK 运行须使用已保存策略数据集的完整成分股；改变股票范围需先准备独立数据集")
            try:
                request.app.state.boundary_data.require_ready(
                    dataset, start=payload.start.isoformat(), end=payload.end.isoformat()
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        payload = payload.model_copy(
            update={
                "database_path": dataset["database_path"],
                "dataset_id": dataset["dataset_id"],
                "training_scope": "sector-pooled" if payload.strategy == "boundary-flow-fk" else "per-symbol",
            }
        )
    elif payload.training_scope == "sector-pooled":
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
