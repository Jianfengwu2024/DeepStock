from __future__ import annotations

import time
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from bnhmm.data.local_dataset import LocalMarketDatasetManager, LocalMarketDatasetStore
from bnhmm.types import BarBatch


def test_local_dataset_catalog_validates_pools_and_intervals(tmp_path: Path):
    store = LocalMarketDatasetStore(tmp_path / "datasets")
    dataset = store.create(
        name="Watchlist K lines",
        symbols=["000001", "600000", "000001"],
        intervals=["1d", "1m", "60m"],
        start_date=date(2025, 1, 1),
    )

    assert dataset["symbols"] == ["000001", "600000"]
    assert dataset["intervals"] == ["1m", "60m", "1d"]
    assert dataset["status"] == "queued"
    assert dataset["database_path"].endswith(dataset["dataset_id"] + ".duckdb")
    with pytest.raises(ValueError, match="K 线周期"):
        store.create(name="invalid", symbols=["000001"], intervals=["tick"], start_date=date(2025, 1, 1))


def test_register_materialized_dataset_preserves_external_database_path(tmp_path: Path):
    database = tmp_path / "completed.duckdb"
    database.write_bytes(b"independent local database")
    store = LocalMarketDatasetStore(tmp_path / "catalog")
    dataset = store.register_materialized(
        dataset_id="fk-semiconductor-local",
        name="FK semiconductor local",
        symbols=["000001", "600000"],
        intervals=["1m"],
        start_date=date(2024, 1, 1),
        last_refresh_date=date(2026, 10, 23),
        database_path=database,
        row_counts={"1m": 480},
        coverage={"1m": {"000001": {"rows": 240, "start": "2026-10-23 09:31:00", "end": "2026-10-23 15:00:00"}, "600000": {"rows": 240, "start": "2026-10-23 09:31:00", "end": "2026-10-23 15:00:00"}}},
    )
    assert dataset["status"] == "ready"
    assert dataset["database_path"] == str(database.resolve())
    assert dataset["row_counts"] == {"1m": 480}
    assert dataset["coverage"]["1m"]["000001"]["rows"] == 240
    assert dataset["source"] == "local-materialized"
    assert store.delete(dataset["dataset_id"])
    assert database.exists()


def test_materialized_dataset_is_not_enqueued_for_broker_refresh(tmp_path: Path, monkeypatch):
    database = tmp_path / "completed.duckdb"
    database.write_bytes(b"independent local database")
    root = tmp_path / "market-datasets"
    store = LocalMarketDatasetStore(root)
    dataset = store.register_materialized(
        dataset_id="fk-semiconductor-local",
        name="FK semiconductor local",
        symbols=["000001"],
        intervals=["1m"],
        start_date=date(2024, 1, 1),
        last_refresh_date=date(2026, 10, 20),
        database_path=database,
        row_counts={"1m": 240},
        coverage={"1m": {"000001": {"rows": 240, "start": "2026-10-20 09:31:00", "end": "2026-10-20 15:00:00"}}},
    )
    manager = LocalMarketDatasetManager(root, "http://unused-broker")
    try:
        with pytest.raises(RuntimeError, match="不支持 broker 增量更新"):
            manager.refresh(dataset["dataset_id"])
        manager._schedule_due(datetime(2026, 10, 26, 21, tzinfo=timezone(timedelta(hours=8))))
        assert manager._queue.qsize() == 0
        assert manager.store.get(dataset["dataset_id"])["status"] == "ready"
        assert database.exists()
    finally:
        manager.close()
def test_local_dataset_worker_materializes_every_interval_and_daily_refresh(tmp_path: Path, monkeypatch):
    import bnhmm.data.local_dataset as module

    calls = []
    source_options = []

    class Source:
        def fetch(self, symbol, interval, start, end):
            calls.append((symbol, interval, start, end))
            stamp = f"{start}T09:30:00"
            return BarBatch.from_records(symbol, [{
                "timestamp": stamp,
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10,
                "volume": 100,
            }])

    def create_source(*args, **kwargs):
        source_options.append(kwargs)
        return Source()

    monkeypatch.setattr(module, "create_bar_source", create_source)
    manager = LocalMarketDatasetManager(tmp_path / "datasets", "http://broker/api/v1/db")
    manager.set_busy_probe(lambda: False)
    try:
        created = manager.create(
            name="one symbol history",
            symbols=["000001"],
            intervals=["1m", "5m", "15m", "30m", "60m", "2h", "1d"],
            start_date=date.today() - timedelta(days=3),
        )
        deadline = time.monotonic() + 20
        current = manager.store.get(created["dataset_id"])
        while current["status"] in {"queued", "updating"} and time.monotonic() < deadline:
            time.sleep(0.05)
            current = manager.store.get(created["dataset_id"])

        assert current["status"] == "ready", current["error"]
        assert source_options[0]["timeout_seconds"] == 300
        assert source_options[0]["busy_retries"] == 3
        assert current["last_refresh_date"] == (date.today() - timedelta(days=3)).isoformat()
        assert set(current["row_counts"]) == {"1m", "5m", "15m", "30m", "60m", "2h", "1d"}
        assert len(calls) == 7

        import duckdb
        with duckdb.connect(current["database_path"], read_only=True) as db:
            assert db.execute("SELECT count(*) FROM kline_1m").fetchone()[0] == 1
            assert db.execute("SELECT count(*) FROM kline_1d").fetchone()[0] == 1
            assert db.execute("SELECT symbol FROM stock_universe").fetchone()[0] == "000001"
        first_counts = current["row_counts"]
        calls.clear()
        manager.refresh(created["dataset_id"])
        manager._queue.join()
        refreshed = manager.store.get(created["dataset_id"])
        assert refreshed["row_counts"] == first_counts
        assert refreshed["coverage"]["1m"]["000001"]["rows"] == 1
        assert len(calls) == 7
    finally:
        manager.close()


def test_refresh_waits_until_training_releases_local_dataset(tmp_path: Path, monkeypatch):
    import bnhmm.data.local_dataset as module

    entered = threading.Event()

    class Source:
        def fetch(self, symbol, interval, start, end):
            entered.set()
            return BarBatch.from_records(symbol, [])

    monkeypatch.setattr(module, "create_bar_source", lambda *args, **kwargs: Source())
    resource_lock = threading.Lock()
    resource_lock.acquire()
    manager = LocalMarketDatasetManager(tmp_path, "http://broker", resource_lock=resource_lock)
    try:
        dataset = manager.create(name="locked", symbols=["000001"], intervals=["1m"], start_date=date(2026, 1, 1))
        assert not entered.wait(0.1)
        assert manager.store.get(dataset["dataset_id"])["status"] == "queued"
        with pytest.raises(ValueError, match="训练或更新"):
            manager.delete(dataset["dataset_id"])
        resource_lock.release()
        assert entered.wait(5)
        manager._queue.join()
        assert manager.store.get(dataset["dataset_id"])["status"] == "partial"
    finally:
        if resource_lock.locked():
            resource_lock.release()
        manager.close()


def test_local_dataset_batches_stock_pool_by_interval_and_year(tmp_path: Path, monkeypatch):
    import bnhmm.data.local_dataset as module

    calls = []

    class Source:
        def fetch_many(self, symbols, interval, start, end):
            calls.append((tuple(symbols), interval, start, end))
            return {
                symbol: BarBatch.from_records(symbol, [{
                    "timestamp": f"{start}T09:30:00",
                    "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100,
                }])
                for symbol in symbols
            }

    monkeypatch.setattr(module, "create_bar_source", lambda *args, **kwargs: Source())
    manager = LocalMarketDatasetManager(tmp_path, "http://broker")
    try:
        dataset = manager.create(
            name="batched two-symbol pool",
            symbols=["000001", "600000"],
            intervals=["1m", "1d"],
            start_date=date.today().replace(month=1, day=1),
        )
        manager._queue.join()
        current = manager.store.get(dataset["dataset_id"])
        assert current["status"] == "ready", current["error"]
        assert len(calls) == 2
        assert all(call[0] == ("000001", "600000") for call in calls)
        assert {call[1] for call in calls} == {"1m", "1d"}
        assert all(call[2].year == call[3].year == date.today().year for call in calls)
        assert current["row_counts"] == {"1d": 2, "1m": 2}
    finally:
        manager.close()


def test_large_sector_downloads_use_monthly_symbol_batches(tmp_path: Path, monkeypatch):
    import bnhmm.data.local_dataset as module

    calls = []

    class Source:
        def fetch_many(self, symbols, interval, start, end):
            calls.append((tuple(symbols), interval, start, end))
            return {
                symbol: BarBatch.from_records(symbol, [{
                    "timestamp": f"{start}T09:30:00",
                    "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100,
                }])
                for symbol in symbols
            }

    monkeypatch.setattr(module, "create_bar_source", lambda *args, **kwargs: Source())
    manager = LocalMarketDatasetManager(tmp_path, "http://broker")
    try:
        symbols = [f"{index:06d}" for index in range(26)]
        dataset = manager.create(
            name="large sector one-minute history",
            symbols=symbols,
            intervals=["1m"],
            start_date=date.today().replace(month=1, day=1),
        )
        manager._queue.join()

        assert manager.store.get(dataset["dataset_id"])["status"] == "ready"
        assert calls
        assert max(len(call[0]) for call in calls) <= 25
        assert len(calls) > 10
        assert all(call[1] == "1m" and call[2] < call[3] for call in calls)
        assert {call[2].month for call in calls} == set(range(1, date.today().month + 1))
    finally:
        manager.close()


def test_daily_refresh_catches_up_after_weekend_and_limits_failed_retries(tmp_path: Path, monkeypatch):
    # Keep worker threads stopped to inspect scheduling independently.
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    manager = LocalMarketDatasetManager(tmp_path, "http://broker")
    dataset = manager.store.create(name="daily", symbols=["000001"], intervals=["1m"], start_date=date(2026, 1, 1))
    manager.store.update_status(dataset["dataset_id"], status="partial")
    monday = datetime(2026, 10, 19, 9, tzinfo=timezone(timedelta(hours=8)))
    assert manager.completed_day(monday) == date(2026, 10, 16)
    manager._schedule_due(monday)
    assert manager._queue.qsize() == 1
    manager._schedule_due(monday)
    assert manager._queue.qsize() == 1
    manager.store.update_status(dataset["dataset_id"], status="updating")
    attempted = manager.store.get(dataset["dataset_id"])["last_attempt_at"]
    manager.store.update_status(dataset["dataset_id"], status="partial")
    manager._schedule_due(datetime.fromisoformat(attempted) + timedelta(minutes=5))
    assert manager._queue.qsize() == 1


def test_cloud_strategy_dataset_is_ephemeral_and_cleanup_preserves_normal_data(tmp_path: Path):
    manager = LocalMarketDatasetManager(tmp_path / "datasets", "http://unused-broker")
    deleted_caches = []
    manager.set_expiration_cleanup(deleted_caches.append)
    try:
        cloud = manager.store.create(
            name="cloud strategy run", symbols=["000001"], intervals=["1m"],
            start_date=date(2025, 1, 1), purpose="cloud-strategy",
        )
        normal = manager.store.create(
            name="user market data", symbols=["000002"], intervals=["1m"],
            start_date=date(2025, 1, 1),
        )
        manager.store.update_status(cloud["dataset_id"], status="ready")
        manager.store.update_status(normal["dataset_id"], status="ready", refreshed_date=manager.completed_day(datetime.now(timezone(timedelta(hours=8)))))
        expiry = datetime.fromisoformat(manager.store.get(cloud["dataset_id"])["retention_expires_at"])
        assert expiry - datetime.now(timezone.utc) <= timedelta(days=7)
        assert manager.store.get(normal["dataset_id"])["retention_expires_at"] is None

        with manager.store._connect() as db:
            db.execute(
                "UPDATE market_datasets SET retention_expires_at=? WHERE dataset_id=?",
                [(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), cloud["dataset_id"]],
            )
        manager._cleanup_expired_cloud_datasets(datetime.now(timezone.utc))

        assert manager.store.get(cloud["dataset_id"]) is None
        assert manager.store.get(normal["dataset_id"]) is not None
        assert deleted_caches == [cloud["dataset_id"]]
    finally:
        manager.close()


def test_cloud_strategy_dataset_is_ephemeral_and_cleanup_preserves_normal_data(tmp_path: Path):
    manager = LocalMarketDatasetManager(tmp_path / "datasets", "http://unused-broker")
    deleted_caches = []
    manager.set_expiration_cleanup(deleted_caches.append)
    try:
        cloud = manager.store.create(
            name="cloud strategy run", symbols=["000001"], intervals=["1m"],
            start_date=date(2025, 1, 1), purpose="cloud-strategy",
        )
        normal = manager.store.create(
            name="user market data", symbols=["000002"], intervals=["1m"],
            start_date=date(2025, 1, 1),
        )
        manager.store.update_status(cloud["dataset_id"], status="ready")
        manager.store.update_status(normal["dataset_id"], status="ready")
        expiry = datetime.fromisoformat(manager.store.get(cloud["dataset_id"])["retention_expires_at"])
        assert expiry - datetime.now(timezone.utc) <= timedelta(days=7)
        assert manager.store.get(normal["dataset_id"])["retention_expires_at"] is None

        with manager.store._connect() as db:
            db.execute(
                "UPDATE market_datasets SET retention_expires_at=? WHERE dataset_id=?",
                [(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), cloud["dataset_id"]],
            )
        manager._cleanup_expired_cloud_datasets(datetime.now(timezone.utc))

        assert manager.store.get(cloud["dataset_id"]) is None
        assert manager.store.get(normal["dataset_id"]) is not None
        assert deleted_caches == [cloud["dataset_id"]]
    finally:
        manager.close()
