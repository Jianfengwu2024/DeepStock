from __future__ import annotations

import io
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from bnhmm.web.app import app
from bnhmm.types import BarBatch


class Response:
    def __init__(self, payload):
        self.body = io.BytesIO(json.dumps(payload).encode())

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.body.close()

    def read(self, size=-1):
        return self.body.read(size)


def test_stock_pool_api_persists_changes_in_local_state(tmp_path: Path):
    with patch.dict(os.environ, {
        "BNHMM_WEB_JOB_ROOT": str(tmp_path / "runs"),
        "BNHMM_STOCK_POOL_PATH": str(tmp_path / "pools.sqlite3"),
    }), TestClient(app) as client:
        created = client.post("/api/stock-pools", json={"name": "AI 供应链"})
        assert created.status_code == 201
        identifier = created.json()["id"]
        assert client.post(f"/api/stock-pools/{identifier}/symbols", json={"symbol": "000001"}).status_code == 200
        assert client.get("/api/stock-pools").json()[0]["symbols"] == ["000001"]
        assert client.delete(f"/api/stock-pools/{identifier}/symbols/000001").json() == {"deleted": True}
        assert client.delete(f"/api/stock-pools/{identifier}").json() == {"deleted": True}


def test_ai_configuration_accepts_provider_models_from_api(tmp_path: Path):
    calls = []

    def open_request(request, timeout):
        calls.append(request)
        return Response({"data": [{"id": "deepseek-flash"}, {"id": "future-model-v5"}]})

    with patch.dict(os.environ, {
        "BNHMM_WEB_JOB_ROOT": str(tmp_path / "runs"),
        "BNHMM_STOCK_POOL_PATH": str(tmp_path / "pools.sqlite3"),
    }), patch("bnhmm.web.app.urlopen", side_effect=open_request), TestClient(app) as client:
        result = client.post("/api/ai/models", json={
            "base_url": "https://api.example.com/v1",
            "api_key": "private-test-key",
        })

    assert result.json() == {"models": ["deepseek-flash", "future-model-v5"], "count": 2}
    assert calls[0].full_url == "https://api.example.com/v1/models"
    assert calls[0].get_header("Authorization") == "Bearer private-test-key"


def test_quant_user_verification_checks_the_existing_entitlement(tmp_path: Path):
    calls = []

    def open_request(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            return Response({"access_token": "temporary-access-token-value", "user": {"username": "researcher"}})
        return Response({"datasets": [], "total": 0})

    with patch.dict(os.environ, {
        "BNHMM_WEB_JOB_ROOT": str(tmp_path / "runs"),
        "BNHMM_STOCK_POOL_PATH": str(tmp_path / "pools.sqlite3"),
        "BNHMM_QUANT_SERVER": "https://quant.example",
    }), patch("bnhmm.web.app.urlopen", side_effect=open_request), TestClient(app) as client:
        result = client.post("/api/verification/login", json={"username": "researcher", "password": "ephemeral"})

    assert result.json()["verified"] is True
    assert result.json()["authorized"] is True
    assert result.json()["access_token"] == "temporary-access-token-value"
    assert calls[0].full_url == "https://quant.example/api/v1/auth/login"
    assert calls[1].get_header("Authorization") == "Bearer temporary-access-token-value"


def test_quant_ai_stock_pools_are_read_for_the_verified_user_and_synced_locally(tmp_path: Path):
    calls = []
    cloud_pools = [{
        "pool_id": "ai-pool-1", "name": "AI 供应链", "query": "人工智能供应链 市值大于200亿",
        "description": "AI 供应链市值筛选", "symbols": ["000001", "600000"],
        "source": "fast", "run_id": "screen-run-4", "created_at": "2026-10-22T12:00:00+08:00",
    }]

    def open_request(request, timeout):
        calls.append(request)
        return Response({"data": {"pools": cloud_pools, "total": len(cloud_pools)}})

    with patch.dict(os.environ, {
        "BNHMM_WEB_JOB_ROOT": str(tmp_path / "runs"),
        "BNHMM_STOCK_POOL_PATH": str(tmp_path / "pools.sqlite3"),
        "BNHMM_QUANT_SERVER": "https://quant.example",
    }), patch("bnhmm.web.app.urlopen", side_effect=open_request), TestClient(app) as client:
        listed = client.post("/api/verification/stock-pools", json={"access_token": "temporary-access-token-value"})
        synced = client.post("/api/stock-pools/sync-quant-ai", json={
            "access_token": "temporary-access-token-value", "pool_id": "ai-pool-1",
        })
        locals_ = client.get("/api/stock-pools").json()

    assert listed.status_code == 200
    assert listed.json()["pools"][0]["pool_id"] == "ai-pool-1"
    assert synced.status_code == 200
    assert synced.json()["pool"]["source"] == "quant-ai"
    assert synced.json()["pool"]["symbols"] == ["000001", "600000"]
    assert synced.json()["pool"]["run_id"] == "screen-run-4"
    assert len(locals_) == 1
    assert all(call.full_url == "https://quant.example/api/v1/screener/pools" for call in calls)
    assert all(call.get_header("Authorization") == "Bearer temporary-access-token-value" for call in calls)


def test_quant_watchlist_is_read_and_synced_as_one_replaceable_local_pool(tmp_path: Path):
    calls = []
    remote_items = [
        {"symbol": "000001", "name": "平安银行", "note": "银行", "source": "manual", "added_at": "2026-10-22T09:00:00"},
        {"symbol": "600000", "name": "浦发银行", "note": None, "source": "manual", "added_at": "2026-10-22T10:00:00"},
    ]

    def open_request(request, timeout):
        calls.append(request)
        return Response({"success": True, "data": remote_items})

    with patch.dict(os.environ, {
        "BNHMM_WEB_JOB_ROOT": str(tmp_path / "runs"),
        "BNHMM_STOCK_POOL_PATH": str(tmp_path / "pools.sqlite3"),
        "BNHMM_QUANT_SERVER": "https://quant.example",
    }), patch("bnhmm.web.app.urlopen", side_effect=open_request), TestClient(app) as client:
        listed = client.post("/api/verification/watchlist", json={"access_token": "temporary-access-token-value"})
        synced = client.post("/api/stock-pools/sync-quant-watchlist", json={
            "access_token": "temporary-access-token-value", "pool_id": "quant-watchlist-main",
        })
        remote_items[:] = [remote_items[1]]
        updated = client.post("/api/stock-pools/sync-quant-watchlist", json={
            "access_token": "temporary-access-token-value", "pool_id": "quant-watchlist-main",
        })
        pools = client.get("/api/stock-pools").json()

    assert listed.json()["total"] == 1
    assert listed.json()["items"][0]["symbols"] == ["000001", "600000"]
    assert synced.json()["pool"]["source"] == "quant-watchlist"
    assert updated.json()["pool"]["id"] == synced.json()["pool"]["id"]
    assert updated.json()["pool"]["symbols"] == ["600000"]
    assert len(pools) == 1
    assert all(call.full_url == "https://quant.example/api/v1/watchlist" for call in calls)
    assert all(call.get_header("Authorization") == "Bearer temporary-access-token-value" for call in calls)


def test_verified_session_restores_account_without_password(tmp_path: Path):
    calls = []

    def open_request(request, timeout):
        calls.append(request)
        return Response({"user": {"username": "researcher", "display_name": "Researcher"}})

    with patch.dict(os.environ, {
        "BNHMM_WEB_JOB_ROOT": str(tmp_path / "runs"),
        "BNHMM_STOCK_POOL_PATH": str(tmp_path / "pools.sqlite3"),
        "BNHMM_QUANT_SERVER": "https://quant.example",
    }), patch("bnhmm.web.app.urlopen", side_effect=open_request), TestClient(app) as client:
        result = client.post("/api/verification/session", json={"access_token": "temporary-access-token-value"})

    assert result.json()["user"]["username"] == "researcher"
    assert result.json()["authorized"] is True
    assert calls[0].full_url == "https://quant.example/api/v1/auth/me"
    assert calls[1].full_url == "https://quant.example/api/v1/deepstock/catalog"


def test_local_market_dataset_is_built_from_a_stock_pool_and_persisted(tmp_path: Path, monkeypatch):
    from bnhmm.data import local_dataset

    def create_source(*_args, **_kwargs):
        class Source:
            def fetch(self, symbol, interval, start, end):
                return BarBatch.from_records(symbol, [{
                    "timestamp": f"{start}T09:30:00",
                    "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100,
                }])
        return Source()

    monkeypatch.setattr(local_dataset, "create_bar_source", create_source)
    with patch.dict(os.environ, {
        "BNHMM_WEB_JOB_ROOT": str(tmp_path / "runs"),
        "BNHMM_STOCK_POOL_PATH": str(tmp_path / "pools.sqlite3"),
        "BNHMM_MARKET_DATASET_ROOT": str(tmp_path / "market-datasets"),
    }), TestClient(app) as client:
        pool = client.post("/api/stock-pools", json={"name": "local data pool"}).json()
        client.post(f"/api/stock-pools/{pool['id']}/symbols", json={"symbol": "000001"})
        created = client.post("/api/market-datasets", json={
            "name": "offline daily+minute",
            "stock_pool_id": pool["id"],
            "intervals": ["1m", "60m", "1d"],
            "start_date": (date.today() - timedelta(days=3)).isoformat(),
        })
        assert created.status_code == 202
        dataset_id = created.json()["dataset_id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            dataset = next(item for item in client.get("/api/market-datasets").json() if item["dataset_id"] == dataset_id)
            if dataset["status"] not in {"queued", "updating"}:
                break
            time.sleep(0.03)
        assert dataset["status"] == "ready", dataset["error"]
        assert dataset["intervals"] == ["1m", "60m", "1d"]
        assert all(dataset["row_counts"][interval] for interval in dataset["intervals"])
        import duckdb
        with duckdb.connect(dataset["database_path"], read_only=True) as db:
            assert db.execute("SELECT COUNT(*) FROM kline_1m").fetchone()[0] == 1
            assert db.execute("SELECT COUNT(*) FROM kline_60m").fetchone()[0] == 1
            assert db.execute("SELECT COUNT(*) FROM kline_1d").fetchone()[0] == 1

        from bnhmm.web.jobs import JobRecord, utcnow
        captured = []

        def capture_job(payload):
            captured.append(payload)
            return JobRecord(id="local-source-test", status="queued", created_at=utcnow(),
                             output_dir=str(tmp_path / "runs" / "local-source-test"), request=payload)

        with patch("bnhmm.web.jobs.JobManager.submit", side_effect=capture_job):
            response = client.post("/api/jobs", json={
                "name": "local-source-test", "market_dataset_id": dataset_id,
                "database_path": "http://incorrect-source", "symbols": ["000001"],
            })
            assert response.status_code == 201, response.text
            assert captured[0].database_path == dataset["database_path"]
            assert captured[0].market_dataset_id == dataset_id
            assert captured[0].training_scope == "per-symbol"
            rejected = client.post("/api/jobs", json={
                "name": "wrong-symbol", "market_dataset_id": dataset_id, "symbols": ["600000"],
            })
            assert rejected.status_code == 422


def test_materialized_fk_dataset_is_visible_and_submittable_without_refresh(tmp_path: Path):
    from bnhmm.data.local_dataset import LocalMarketDatasetStore
    from bnhmm.web.jobs import JobRecord, utcnow
    from bnhmm.boundary_fk.cache import BoundaryDataCache, BoundaryDataCacheKey
    from boundary_fk_fixtures import walk_forward_sector_data

    dataset_root = tmp_path / "market-datasets"
    database = tmp_path / "completed-semiconductor.duckdb"
    database.write_bytes(b"prebuilt local data")
    symbols = ["000001", "000002", "600000", "600001"]
    coverage = {"1m": {
        symbol: {"rows": 240, "start": "2026-10-26 09:31:00", "end": "2026-10-26 15:00:00"}
        for symbol in symbols
    }}
    LocalMarketDatasetStore(dataset_root).register_materialized(
        dataset_id="fk-semiconductor-local",
        name="FK-贝叶斯-半导体（本地 DuckDB）",
        symbols=symbols,
        intervals=["1m"],
        start_date=date(2024, 1, 1),
        last_refresh_date=date(2026, 10, 26),
        database_path=database,
        row_counts={"1m": 960},
        coverage=coverage,
    )
    captured = []
    cache_root = tmp_path / "strategy-inputs"
    BoundaryDataCache(cache_root).save(BoundaryDataCacheKey.from_source(
        dataset_id="fk-semiconductor-local", source_path=database, symbols=tuple(symbols),
        start="2025-10-01", end="2026-10-27", lookback_sessions=60,
    ), walk_forward_sector_data(180))

    def capture_job(payload):
        captured.append(payload)
        return JobRecord(id="materialized-fk-test", status="queued", created_at=utcnow(),
                         output_dir=str(tmp_path / "runs" / "materialized-fk-test"), request=payload)

    with patch.dict(os.environ, {
        "BNHMM_WEB_JOB_ROOT": str(tmp_path / "runs"),
        "BNHMM_STOCK_POOL_PATH": str(tmp_path / "pools.sqlite3"),
        "BNHMM_MARKET_DATASET_ROOT": str(dataset_root),
        "BNHMM_BOUNDARY_CACHE_ROOT": str(cache_root),
    }), TestClient(app) as client, patch("bnhmm.web.jobs.JobManager.submit", side_effect=capture_job):
        config = client.get("/api/config").json()
        listed = client.get("/api/market-datasets").json()
        dataset = next(item for item in listed if item["dataset_id"] == "fk-semiconductor-local")
        assert config["strategy_defaults"]["boundary-flow-fk"]["market_dataset_id"] == dataset["dataset_id"]
        assert config["strategy_defaults"]["boundary-flow-fk"]["start"] == "2025-10-26"
        assert config["strategy_defaults"]["boundary-flow-fk"]["end"] == "2026-10-27"
        assert dataset["source"] == "local-materialized"
        assert dataset["status"] == "ready"
        assert dataset["database_path"] == str(database.resolve())
        strategy_inputs = client.get("/api/strategy-datasets").json()
        assert strategy_inputs[0]["status"] == "ready"
        assert strategy_inputs[0]["run_mode"] == "reuse_only"
        assert client.get("/api/strategy-datasets/fk-semiconductor-local?start=2025-09-01&end=2026-10-27").json()["status"] == "stale"
        refresh = client.post(f"/api/market-datasets/{dataset['dataset_id']}/refresh")
        assert refresh.status_code == 409
        response = client.post("/api/jobs", json={
            "strategy": "boundary-flow-fk",
            "training_scope": "sector-pooled",
            "name": "materialized-fk-test",
            "market_dataset_id": dataset["dataset_id"],
            "dataset_id": dataset["dataset_id"],
            "symbols": symbols,
            "start": "2025-10-01",
            "end": "2026-10-26",
            "minimum_train_sessions": 120,
        })
        assert response.status_code == 201, response.text
        assert captured[0].database_path == str(database.resolve())
        assert captured[0].dataset_id == dataset["dataset_id"]
        assert captured[0].symbols == symbols
        database.write_bytes(b"changed source")
        rejected = client.post("/api/jobs", json={
            "strategy":"boundary-flow-fk", "training_scope":"sector-pooled", "name":"stale-test",
            "market_dataset_id":dataset["dataset_id"], "dataset_id":dataset["dataset_id"], "symbols":symbols,
            "start":"2025-10-01", "end":"2026-10-26", "minimum_train_sessions":120,
        })
        assert rejected.status_code == 409
        assert len(captured) == 1


def test_successful_account_login_stays_online_without_deepstock_entitlement(tmp_path: Path):
    calls = []

    def open_request(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            return Response({"access_token": "temporary-access-token-value", "user": {"username": "watchlist-user"}})
        raise __import__("urllib.error", fromlist=["HTTPError"]).HTTPError(
            request.full_url, 403, "forbidden", {}, None
        )

    with patch.dict(os.environ, {
        "BNHMM_WEB_JOB_ROOT": str(tmp_path / "runs"),
        "BNHMM_STOCK_POOL_PATH": str(tmp_path / "pools.sqlite3"),
        "BNHMM_QUANT_SERVER": "https://quant.example",
    }), patch("bnhmm.web.app.urlopen", side_effect=open_request), TestClient(app) as client:
        result = client.post("/api/verification/login", json={"username": "watchlist-user", "password": "ephemeral"})

    assert result.status_code == 200
    assert result.json()["verified"] is True
    assert result.json()["authorized"] is False
    assert result.json()["access_token"] == "temporary-access-token-value"
    assert result.json()["user"]["username"] == "watchlist-user"
