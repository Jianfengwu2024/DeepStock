from pathlib import Path

import pytest

from bnhmm.web.local_store import StockPoolStore


def test_stock_pools_persist_manual_members_and_delete_cascade(tmp_path: Path):
    path = tmp_path / "local-state" / "pools.sqlite3"
    store = StockPoolStore(path)
    pool = store.create("人工智能供应链")
    assert store.add_symbol(pool["id"], "000001")
    assert store.add_symbol(pool["id"], "600000")
    assert store.add_symbol(pool["id"], "000001")

    reopened = StockPoolStore(path)
    assert reopened.list()[0]["symbols"] == ["000001", "600000"]
    assert reopened.remove_symbol(pool["id"], "600000")
    assert reopened.delete(pool["id"])
    assert reopened.list() == []


def test_stock_pool_names_symbols_and_duplicates_are_validated(tmp_path: Path):
    store = StockPoolStore(tmp_path / "pools.sqlite3")
    with pytest.raises(ValueError):
        store.create("   ")
    pool = store.create("watch")
    with pytest.raises(ValueError):
        store.create("watch")
    with pytest.raises(ValueError):
        store.add_symbol(pool["id"], "bad symbol")


def test_quant_ai_pool_sync_is_linked_and_updates_in_place(tmp_path: Path):
    store = StockPoolStore(tmp_path / "pools.sqlite3")
    first = store.sync_cloud_pool({
        "pool_id": "ai-pool-1", "name": "人工智能供应链", "symbols": ["000001", "600000"],
        "query": "人工智能供应链 市值大于200亿", "run_id": "run-123", "created_at": "2026-10-21T12:00:00+08:00",
    })
    updated = store.sync_cloud_pool({
        "pool_id": "ai-pool-1", "name": "人工智能供应链", "symbols": ["000001", "300750"],
        "query": "人工智能供应链 市值大于300亿", "run_id": "run-456", "created_at": "2026-10-22T12:00:00+08:00",
    })

    assert updated["id"] == first["id"]
    assert updated["source"] == "quant-ai"
    assert updated["cloud_pool_id"] == "ai-pool-1"
    assert updated["symbols"] == ["000001", "300750"]
    assert updated["run_id"] == "run-456"
    assert "300亿" in updated["description"]

    with pytest.raises(ValueError, match="证券代码"):
        store.sync_cloud_pool({"pool_id": "bad", "name": "bad", "symbols": ["ABC"]})
