"""Locally materialized market datasets with bounded daily broker refreshes."""

from __future__ import annotations

import json
import os
import queue
import re
import sqlite3
import threading
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .source import BrokerError, create_bar_source


_INTERVAL_ORDER = ("1m", "5m", "15m", "30m", "60m", "2h", "1d")
_INTERVALS = set(_INTERVAL_ORDER)
_SYMBOL = re.compile(r"^[A-Za-z0-9._-]{1,24}$")
_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_LOCAL_TZ = timezone(timedelta(hours=8))
_FETCH_SYMBOL_BATCH_SIZE = 25
_MONTHLY_FETCH_THRESHOLD = _FETCH_SYMBOL_BATCH_SIZE


class LocalMarketDatasetStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.root / "catalog.sqlite3"
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS market_datasets (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    symbols_json TEXT NOT NULL,
                    intervals_json TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_refresh_date TEXT,
                    row_counts_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT
                )
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(market_datasets)")}
            for column in ("last_attempt_at", "coverage_json", "database_path_override", "retention_expires_at", "progress_json"):
                if column not in columns:
                    db.execute(f"ALTER TABLE market_datasets ADD COLUMN {column} TEXT")
            if "purpose" not in columns:
                db.execute("ALTER TABLE market_datasets ADD COLUMN purpose TEXT NOT NULL DEFAULT 'market'")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.catalog_path, timeout=20)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _metadata(row: sqlite3.Row, root: Path) -> dict[str, Any]:
        dataset_id = row["dataset_id"]
        symbols = json.loads(row["symbols_json"])
        intervals = json.loads(row["intervals_json"])
        database_path = row["database_path_override"] or str(root / f"{dataset_id}.duckdb")
        return {
            "dataset_id": dataset_id,
            "name": row["name"],
            "database_path": database_path,
            "symbols": symbols,
            "member_count": len(symbols),
            "intervals": intervals,
            "start_date": row["start_date"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_refresh_date": row["last_refresh_date"],
            "last_attempt_at": row["last_attempt_at"],
            "coverage": json.loads(row["coverage_json"] or "{}"),
            "row_counts": json.loads(row["row_counts_json"] or "{}"),
            "error": row["error"],
            "purpose": row["purpose"],
            "retention_expires_at": row["retention_expires_at"],
            "progress": json.loads(row["progress_json"]) if row["progress_json"] else None,
            "source": "local-materialized" if row["database_path_override"] else "local-market-broker",
        }

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM market_datasets ORDER BY created_at DESC, name COLLATE NOCASE"
            ).fetchall()
            return [self._metadata(row, self.root) for row in rows]

    def get(self, dataset_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM market_datasets WHERE dataset_id=?", [dataset_id]
            ).fetchone()
            return self._metadata(row, self.root) if row else None

    def create(
        self,
        *,
        name: str,
        symbols: list[str],
        intervals: list[str],
        start_date: date,
        purpose: str = "market",
    ) -> dict[str, Any]:
        clean_name = name.strip()
        clean_symbols = list(dict.fromkeys(item.strip() for item in symbols if item.strip()))
        if not intervals or any(item not in _INTERVALS for item in intervals):
            raise ValueError("至少选择一个支持的 K 线周期")
        clean_intervals = sorted(set(intervals), key=_INTERVAL_ORDER.index)
        if not 1 <= len(clean_name) <= 80:
            raise ValueError("数据集名称长度须为 1 到 80 个字符")
        if not clean_symbols or len(clean_symbols) > 2000:
            raise ValueError("数据集必须包含 1 至 2000 个证券")
        if any(not _SYMBOL.fullmatch(symbol) for symbol in clean_symbols):
            raise ValueError("数据集中包含格式无效的证券代码")
        if start_date > datetime.now(_LOCAL_TZ).date():
            raise ValueError("数据集开始日期不能晚于今天")
        if purpose not in {"market", "cloud-strategy"}:
            raise ValueError("数据集用途无效")
        now = datetime.now(timezone.utc).isoformat()
        dataset_id = uuid.uuid4().hex
        try:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO market_datasets(dataset_id,name,symbols_json,intervals_json,start_date,status,created_at,updated_at,purpose) VALUES(?,?,?,?,?,'queued',?,?,?)",
                    [dataset_id, clean_name, json.dumps(clean_symbols), json.dumps(clean_intervals), start_date.isoformat(), now, now, purpose],
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("本地数据集名称已存在") from exc
        return self.get(dataset_id)  # type: ignore[return-value]

    def update_status(self, dataset_id: str, *, status: str, row_counts: dict[str, int] | None = None,
                      error: str | None = None, refreshed_date: date | None = None,
                      coverage: dict | None = None, progress: dict[str, Any] | None = None) -> None:
        with self._connect() as db:
            row = db.execute("SELECT purpose,retention_expires_at FROM market_datasets WHERE dataset_id=?", [dataset_id]).fetchone()
            expires_at = None
            if row and row["purpose"] == "cloud-strategy" and row["retention_expires_at"] is None and status in {"ready", "partial", "failed"}:
                expires_at = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
            db.execute(
                "UPDATE market_datasets SET status=?,updated_at=?,last_refresh_date=COALESCE(?,last_refresh_date),row_counts_json=COALESCE(?,row_counts_json),error=?,retention_expires_at=COALESCE(retention_expires_at,?),progress_json=COALESCE(?,progress_json) WHERE dataset_id=?",
                [status, datetime.now(timezone.utc).isoformat(), refreshed_date.isoformat() if refreshed_date else None,
                 json.dumps(row_counts, sort_keys=True) if row_counts is not None else None, error, expires_at,
                 json.dumps(progress, sort_keys=True) if progress is not None else None, dataset_id],
            )
            if status == "updating":
                db.execute("UPDATE market_datasets SET last_attempt_at=? WHERE dataset_id=?",
                           [datetime.now(timezone.utc).isoformat(), dataset_id])
            if coverage is not None:
                db.execute("UPDATE market_datasets SET coverage_json=? WHERE dataset_id=?",
                           [json.dumps(coverage, sort_keys=True), dataset_id])

    def register_materialized(
        self,
        *,
        dataset_id: str,
        name: str,
        symbols: list[str],
        intervals: list[str],
        start_date: date,
        last_refresh_date: date,
        database_path: str | Path,
        row_counts: dict[str, int],
        coverage: dict[str, dict[str, dict[str, Any]]],
        status: str = "ready",
        error: str | None = None,
    ) -> dict[str, Any]:
        """Register an already materialized local DuckDB without broker refresh."""
        clean_id = dataset_id.strip()
        clean_name = name.strip()
        clean_symbols = list(dict.fromkeys(item.strip() for item in symbols if item.strip()))
        clean_intervals = sorted(set(intervals), key=_INTERVAL_ORDER.index)
        path = Path(database_path).expanduser().resolve()
        if not _DATASET_ID.fullmatch(clean_id):
            raise ValueError("数据集 ID 格式无效")
        if not 1 <= len(clean_name) <= 80:
            raise ValueError("数据集名称长度须为 1 到 80 个字符")
        if not clean_symbols or len(clean_symbols) > 2000:
            raise ValueError("数据集必须包含 1 至 2000 个证券")
        if any(not _SYMBOL.fullmatch(symbol) for symbol in clean_symbols):
            raise ValueError("数据集中包含格式无效的证券代码")
        if not clean_intervals or any(interval not in _INTERVALS for interval in clean_intervals):
            raise ValueError("至少选择一个支持的 K 线周期")
        if status not in {"ready", "partial"}:
            raise ValueError("已物化数据集状态必须为 ready 或 partial")
        if not path.is_file():
            raise ValueError(f"本地行情数据库不存在：{path}")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            conflict = db.execute(
                "SELECT dataset_id FROM market_datasets WHERE name=? AND dataset_id<>?",
                [clean_name, clean_id],
            ).fetchone()
            if conflict:
                raise ValueError("本地数据集名称已存在")
            db.execute(
                """
                INSERT INTO market_datasets(
                    dataset_id,name,symbols_json,intervals_json,start_date,status,
                    created_at,updated_at,last_refresh_date,row_counts_json,error,
                    coverage_json,database_path_override
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    name=excluded.name,symbols_json=excluded.symbols_json,
                    intervals_json=excluded.intervals_json,start_date=excluded.start_date,
                    status=excluded.status,updated_at=excluded.updated_at,
                    last_refresh_date=excluded.last_refresh_date,
                    row_counts_json=excluded.row_counts_json,error=excluded.error,
                    coverage_json=excluded.coverage_json,
                    database_path_override=excluded.database_path_override
                """,
                [
                    clean_id, clean_name, json.dumps(clean_symbols), json.dumps(clean_intervals),
                    start_date.isoformat(), status, now, now, last_refresh_date.isoformat(),
                    json.dumps(row_counts, sort_keys=True), error,
                    json.dumps(coverage, sort_keys=True), str(path),
                ],
            )
        return self.get(clean_id)  # type: ignore[return-value]

    def delete(self, dataset_id: str) -> bool:
        dataset = self.get(dataset_id)
        if dataset is None:
            return False
        if dataset["status"] in {"queued", "updating"}:
            raise ValueError("正在构建或更新的数据集不能删除")
        # Imported research databases live outside the managed dataset root and
        # must remain available after removing their catalog entry.
        if dataset["source"] == "local-market-broker":
            Path(dataset["database_path"]).unlink(missing_ok=True)
        with self._connect() as db:
            db.execute("DELETE FROM market_datasets WHERE dataset_id=?", [dataset_id])
        return True


class LocalMarketDatasetManager:
    def __init__(self, root: str | Path, broker_url: str, broker_token: str = "", *,
                 resource_lock: Any = None, expiration_cleanup: Callable[[str], None] | None = None) -> None:
        self.store = LocalMarketDatasetStore(root)
        self.broker_url = broker_url
        self.broker_token = broker_token
        self._queue: queue.Queue[tuple[str, bool] | None] = queue.Queue()
        self._closed = threading.Event()
        self._busy_probe: Callable[[], bool] = lambda: False
        self._dataset_busy_probe: Callable[[str], bool] = lambda _dataset_id: False
        self._expiration_cleanup: Callable[[str], None] = expiration_cleanup or (lambda _dataset_id: None)
        self.resource_lock = resource_lock if resource_lock is not None else threading.Lock()
        self._enqueue_lock = threading.Lock()
        self._thread = threading.Thread(target=self._work, name="deepstock-local-datasets", daemon=True)
        self._scheduler = threading.Thread(target=self._schedule, name="deepstock-dataset-refresh", daemon=True)
        self._thread.start()
        self._scheduler.start()
        for dataset in self.store.list():
            if dataset["status"] in {"queued", "updating"}:
                self._queue.put((dataset["dataset_id"], False))

    def create(self, **values) -> dict[str, Any]:
        dataset = self.store.create(**values)
        self._queue.put((dataset["dataset_id"], True))
        return dataset

    def refresh(self, dataset_id: str) -> dict[str, Any]:
        with self._enqueue_lock:
            dataset = self.store.get(dataset_id)
            if dataset is None:
                raise KeyError(dataset_id)
            if dataset["status"] in {"queued", "updating"}:
                raise RuntimeError("该数据集已有构建或更新任务")
            if dataset["source"] == "local-materialized":
                raise RuntimeError("已导入的本地研究数据集不支持 broker 增量更新")
            self.store.update_status(dataset_id, status="queued")
            self._queue.put((dataset_id, False))
        return {"dataset_id": dataset_id, "status": "queued"}

    def delete(self, dataset_id: str) -> bool:
        if not self.resource_lock.acquire(blocking=False):
            raise ValueError("本地数据正在用于训练或更新，请稍后删除")
        try:
            with self._enqueue_lock:
                dataset = self.store.get(dataset_id)
                if dataset is not None and dataset.get("purpose") == "cloud-strategy":
                    self._expiration_cleanup(dataset_id)
                return self.store.delete(dataset_id)
        finally:
            self.resource_lock.release()

    def close(self) -> None:
        self._closed.set()
        self._queue.put(None)
        self._thread.join(timeout=10)
        self._scheduler.join(timeout=2)

    def set_busy_probe(self, probe: Callable[[], bool]) -> None:
        self._busy_probe = probe

    def set_expiration_cleanup(self, cleanup: Callable[[str], None]) -> None:
        self._expiration_cleanup = cleanup

    def set_dataset_busy_probe(self, probe: Callable[[str], bool]) -> None:
        self._dataset_busy_probe = probe

    def _schedule(self) -> None:
        while not self._closed.is_set():
            self._schedule_due(datetime.now(_LOCAL_TZ))
            self._cleanup_expired_cloud_datasets(datetime.now(timezone.utc))
            if self._closed.wait(300):
                break

    def _cleanup_expired_cloud_datasets(self, now: datetime) -> None:
        if not self.resource_lock.acquire(blocking=False):
            return
        try:
            for dataset in self.store.list():
                expires = dataset.get("retention_expires_at")
                if dataset.get("purpose") != "cloud-strategy" or not expires:
                    continue
                if self._dataset_busy_probe(dataset["dataset_id"]):
                    continue
                if datetime.fromisoformat(expires) > now:
                    continue
                try:
                    self._expiration_cleanup(dataset["dataset_id"])
                    self.store.delete(dataset["dataset_id"])
                except (OSError, ValueError, RuntimeError):
                    continue
        finally:
            self.resource_lock.release()

    @staticmethod
    def completed_day(now: datetime) -> date:
        day = now.date() - timedelta(days=now.hour < 16)
        while day.weekday() >= 5:
            day -= timedelta(days=1)
        return day

    def _schedule_due(self, now: datetime) -> None:
        target = self.completed_day(now)
        for dataset in self.store.list():
            if dataset["source"] == "local-materialized" or dataset.get("purpose") == "cloud-strategy":
                continue
            attempted = dataset["last_attempt_at"]
            # One automatic attempt per completed day. Manual refresh remains
            # available after delayed ingestion or a broker failure.
            attempted_day = self.completed_day(datetime.fromisoformat(attempted).astimezone(_LOCAL_TZ)) if attempted else None
            if attempted_day and attempted_day >= target:
                continue
            if dataset["last_refresh_date"] and dataset["last_refresh_date"] >= target.isoformat():
                if dataset["status"] == "ready":
                    continue
            if dataset["status"] not in {"queued", "updating"}:
                try:
                    self.refresh(dataset["dataset_id"])
                except (KeyError, RuntimeError):
                    pass

    def _work(self) -> None:
        while not self._closed.is_set():
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            dataset_id, initial = item
            try:
                if self._busy_probe():
                    if not self._closed.wait(60):
                        self._queue.put(item)
                    continue
                while not self._closed.is_set():
                    if self.resource_lock.acquire(timeout=0.5):
                        try:
                            if not self._closed.is_set():
                                self._update(dataset_id, initial=initial)
                        finally:
                            self.resource_lock.release()
                        break
            except Exception as exc:
                current = self.store.get(dataset_id)
                if current:
                    self.store.update_status(dataset_id, status="failed" if initial else "partial", error=str(exc)[:1000])
            finally:
                self._queue.task_done()

    def _update(self, dataset_id: str, *, initial: bool) -> None:
        dataset = self.store.get(dataset_id)
        if dataset is None:
            return
        started = time.monotonic()
        self.store.update_status(dataset_id, status="updating", error=None, progress={
            "phase": "materializing_market_data", "message": "正在准备本机分钟行情数据",
            "completed_units": 0, "total_units": None, "unit": "行情批次", "fraction": None,
            "elapsed_seconds": 0, "eta_seconds": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        source = create_bar_source(
            self.broker_url,
            # Large read-only DuckDB scans may take minutes on the full market
            # file. A short client timeout does not cancel server-side DuckDB
            # work and can strand a broker reader after the client has moved on.
            timeout_seconds=float(os.getenv("BNHMM_MARKET_READ_TIMEOUT_SECONDS", "300")),
            broker_token=self.broker_token,
            busy_retries=3,
        )
        import duckdb

        database_path = dataset["database_path"]
        counts: dict[str, int] = {interval: 0 for interval in dataset["intervals"]}
        errors: list[str] = []
        completed_units = 0
        total_units = 0
        latest_end = self.completed_day(datetime.now(_LOCAL_TZ)) + timedelta(days=1)
        for interval in dataset["intervals"]:
            monthly = len(dataset["symbols"]) > _MONTHLY_FETCH_THRESHOLD
            cursor = date.fromisoformat(dataset["start_date"])
            while cursor < latest_end:
                next_boundary = date(cursor.year + (cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1) if monthly else date(cursor.year + 1, 1, 1)
                cursor = min(latest_end, next_boundary)
                total_units += max(1, (len(dataset["symbols"]) + _FETCH_SYMBOL_BATCH_SIZE - 1) // _FETCH_SYMBOL_BATCH_SIZE)
        total_units = max(total_units, 1) * len(dataset["intervals"])
        with duckdb.connect(database_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS stock_universe(symbol VARCHAR PRIMARY KEY)")
            db.executemany("INSERT OR IGNORE INTO stock_universe VALUES(?)", [(s,) for s in dataset["symbols"]])
            for interval in dataset["intervals"]:
                column_type = "DATE" if interval == "1d" else "TIMESTAMP"
                table = "kline_" + interval
                db.execute(f"CREATE TABLE IF NOT EXISTS {table}(symbol VARCHAR NOT NULL,timestamp {column_type} NOT NULL,open DOUBLE NOT NULL,high DOUBLE NOT NULL,low DOUBLE NOT NULL,close DOUBLE NOT NULL,volume BIGINT NOT NULL,paused BOOLEAN DEFAULT FALSE,suspicious BOOLEAN DEFAULT FALSE,PRIMARY KEY(symbol,timestamp))")
                symbol_starts = {}
                for symbol in dataset["symbols"]:
                    last = db.execute(f"SELECT MAX(timestamp) FROM {table} WHERE symbol=?", [symbol]).fetchone()[0]
                    last_day = last.date() if isinstance(last, datetime) else last
                    symbol_starts[symbol] = max(date.fromisoformat(dataset["start_date"]), last_day - timedelta(days=5)) if last else date.fromisoformat(dataset["start_date"])
                cursor = min(symbol_starts.values())
                end = self.completed_day(datetime.now(_LOCAL_TZ)) + timedelta(days=1)
                monthly_chunks = len(dataset["symbols"]) > _MONTHLY_FETCH_THRESHOLD
                while cursor < end:
                    if self._closed.is_set():
                        raise RuntimeError("数据集更新已停止，重启后可继续")
                    next_boundary = (
                        date(cursor.year + (cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1)
                        if monthly_chunks
                        else date(cursor.year + 1, 1, 1)
                    )
                    chunk_end = min(end, next_boundary)
                    active_symbols = [symbol for symbol in dataset["symbols"] if symbol_starts[symbol] < chunk_end]
                    failed_window = False
                    for offset in range(0, len(active_symbols), _FETCH_SYMBOL_BATCH_SIZE):
                        symbol_batch = active_symbols[offset:offset + _FETCH_SYMBOL_BATCH_SIZE]
                        try:
                            fetch_many = getattr(source, "fetch_many", None)
                            if fetch_many:
                                bars_by_symbol = fetch_many(symbol_batch, interval, cursor, chunk_end)
                            else:
                                bars_by_symbol = {
                                    symbol: source.fetch(symbol, interval, max(cursor, symbol_starts[symbol]), chunk_end)
                                    for symbol in symbol_batch
                                }
                        except BrokerError:
                            # Do not fan out a failed broker read into per-symbol rescans.
                            raise
                        except Exception as exc:
                            errors.append(f"{interval}/{cursor.isoformat()}:{chunk_end.isoformat()}: {str(exc)[:180]}")
                            failed_window = True
                            break
                        for symbol, bars in bars_by_symbol.items():
                            if not len(bars):
                                continue
                            first_allowed = symbol_starts[symbol]
                            selected = bars.timestamp >= np.datetime64(first_allowed, "ns")
                            if not selected.any():
                                continue
                            bars = bars.take(np.flatnonzero(selected))
                            db.execute(
                                f"INSERT OR REPLACE INTO {table} SELECT ?, unnest(?::{column_type}[]), "
                                "unnest(?::DOUBLE[]), unnest(?::DOUBLE[]), unnest(?::DOUBLE[]), "
                                "unnest(?::DOUBLE[]), unnest(?::BIGINT[]), false, false",
                                [symbol, bars.timestamp.astype("datetime64[us]").astype(str).tolist(),
                                 bars.open.tolist(), bars.high.tolist(), bars.low.tolist(),
                                 bars.close.tolist(), bars.volume.astype("int64").tolist()],
                            )
                        if errors and errors[-1].startswith(f"{interval}/{cursor.isoformat()}"):
                            break
                        completed_units += 1
                        elapsed = time.monotonic() - started
                        fraction = min(1.0, completed_units / total_units)
                        eta = elapsed * (total_units - completed_units) / completed_units if completed_units else None
                        self.store.update_status(dataset_id, status="updating", progress={
                            "phase": "materializing_market_data",
                            "message": f"正在读取 {interval} 行情：{cursor.isoformat()} 至 {chunk_end.isoformat()}",
                            "completed_units": completed_units, "total_units": total_units,
                            "unit": "行情批次", "fraction": fraction,
                            "elapsed_seconds": elapsed, "eta_seconds": max(0.0, eta) if eta is not None else None,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        })
                    if failed_window:
                        break
                    cursor = chunk_end
                counts[interval] = int(db.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
                if counts[interval] == 0:
                    errors.append(f"{interval}: 上游尚无所选证券的行情")
            coverage = {}
            for interval in dataset["intervals"]:
                coverage[interval] = {
                    symbol: {"rows": int(count), "start": str(first), "end": str(last)}
                    for symbol, count, first, last in db.execute(
                        f"SELECT symbol,count(*),min(timestamp),max(timestamp) FROM kline_{interval} GROUP BY symbol"
                    ).fetchall()
                }
                missing = set(dataset["symbols"]) - coverage[interval].keys()
                if missing:
                    errors.append(f"{interval}: {len(missing)} 个标的尚无行情")
        status = "partial" if errors else "ready"
        ends = [item["end"][:10] for values in coverage.values() for item in values.values()]
        self.store.update_status(
            dataset_id, status=status, row_counts=counts, coverage=coverage,
            error="\n".join(errors[:40]) if errors else None,
            refreshed_date=date.fromisoformat(min(ends)) if ends else None,
            progress={
                "phase": "completed" if status == "ready" else "failed",
                "message": "本机 1 分钟行情准备完成" if status == "ready" else "部分本机行情未能物化",
                "completed_units": completed_units, "total_units": total_units,
                "unit": "行情批次", "fraction": 1.0 if status == "ready" else completed_units / total_units,
                "elapsed_seconds": time.monotonic() - started, "eta_seconds": 0 if status == "ready" else None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
