"""Small local-only storage for manually curated stock pools."""

from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


_SYMBOL = re.compile(r"^[A-Za-z0-9._-]{1,24}$")


class StockPoolStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS stock_pools (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual',
                    cloud_pool_id TEXT,
                    description TEXT,
                    run_id TEXT,
                    cloud_created_at TEXT,
                    synced_at TEXT
                );
                CREATE TABLE IF NOT EXISTS stock_pool_symbols (
                    pool_id TEXT NOT NULL REFERENCES stock_pools(id) ON DELETE CASCADE,
                    symbol TEXT NOT NULL,
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (pool_id, symbol)
                );
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(stock_pools)")}
            for name, sql_type in (
                ("source", "TEXT NOT NULL DEFAULT 'manual'"),
                ("cloud_pool_id", "TEXT"),
                ("description", "TEXT"),
                ("run_id", "TEXT"),
                ("cloud_created_at", "TEXT"),
                ("synced_at", "TEXT"),
            ):
                if name not in columns:
                    db.execute(f"ALTER TABLE stock_pools ADD COLUMN {name} {sql_type}")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_stock_pools_cloud_id ON stock_pools(cloud_pool_id) WHERE cloud_pool_id IS NOT NULL")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def list(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id,name,created_at,source,cloud_pool_id,description,run_id,cloud_created_at,synced_at FROM stock_pools ORDER BY name COLLATE NOCASE"
            ).fetchall()
            result = []
            for row in rows:
                symbols = db.execute(
                    "SELECT symbol FROM stock_pool_symbols WHERE pool_id=? ORDER BY symbol",
                    [row["id"]],
                ).fetchall()
                result.append({
                    "id": row["id"],
                    "name": row["name"],
                    "created_at": row["created_at"],
                    "source": row["source"],
                    "cloud_pool_id": row["cloud_pool_id"],
                    "description": row["description"],
                    "run_id": row["run_id"],
                    "cloud_created_at": row["cloud_created_at"],
                    "synced_at": row["synced_at"],
                    "symbols": [item["symbol"] for item in symbols],
                })
            return result

    def create(self, name: str) -> dict:
        name = name.strip()
        if not 1 <= len(name) <= 80:
            raise ValueError("股票池名称长度须为 1 到 80 个字符")
        identifier = uuid.uuid4().hex
        created = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO stock_pools(id,name,created_at) VALUES(?,?,?)",
                    [identifier, name, created],
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("股票池名称已存在") from exc
        return {"id": identifier, "name": name, "created_at": created, "symbols": []}

    def delete(self, identifier: str) -> bool:
        with self._connect() as db:
            return db.execute("DELETE FROM stock_pools WHERE id=?", [identifier]).rowcount > 0

    def add_symbol(self, identifier: str, symbol: str) -> bool:
        symbol = symbol.strip().upper()
        if not _SYMBOL.fullmatch(symbol):
            raise ValueError("股票代码格式无效")
        with self._connect() as db:
            if db.execute("SELECT 1 FROM stock_pools WHERE id=?", [identifier]).fetchone() is None:
                return False
            db.execute(
                "INSERT OR IGNORE INTO stock_pool_symbols(pool_id,symbol,added_at) VALUES(?,?,?)",
                [identifier, symbol, datetime.now(timezone.utc).isoformat()],
            )
        return True

    def remove_symbol(self, identifier: str, symbol: str) -> bool:
        with self._connect() as db:
            if db.execute("SELECT 1 FROM stock_pools WHERE id=?", [identifier]).fetchone() is None:
                return False
            db.execute(
                "DELETE FROM stock_pool_symbols WHERE pool_id=? AND symbol=?",
                [identifier, symbol.strip().upper()],
            )
        return True

    def sync_cloud_pool(self, payload: dict) -> dict:
        cloud_id = str(payload["pool_id"]).strip()
        name = str(payload["name"]).strip()
        symbols = list(dict.fromkeys(str(item).strip()[-6:] for item in payload["symbols"]))
        if not cloud_id or len(cloud_id) > 128:
            raise ValueError("云端股票池 ID 无效")
        if not name or len(name) > 80:
            raise ValueError("云端股票池名称无效")
        if not symbols or len(symbols) > 20_000 or any(not re.fullmatch(r"\d{6}", item) for item in symbols):
            raise ValueError("云端股票池证券代码无效")
        pool_source = str(payload.get("source") or "quant-ai")
        if pool_source not in {"quant-ai", "quant-watchlist", "market-sector"}:
            raise ValueError("云端股票池来源无效")
        source_label = {"quant-watchlist": "Quant 自选股", "market-sector": "行业板块", "quant-ai": "Quant AI"}[pool_source]
        local_name = f"{source_label} · {name}"[:80]
        synced_at = datetime.now(timezone.utc).isoformat()
        local_id = "quant-ai-" + uuid.uuid5(uuid.NAMESPACE_URL, cloud_id).hex[:24]
        description = str(payload.get("description") or payload.get("query") or "")[:1000]
        try:
            with self._connect() as db:
                existing = db.execute(
                    "SELECT id FROM stock_pools WHERE cloud_pool_id=?", [cloud_id]
                ).fetchone()
                if existing:
                    local_id = existing["id"]
                    db.execute(
                        "UPDATE stock_pools SET name=?,source=?,description=?,run_id=?,cloud_created_at=?,synced_at=? WHERE id=?",
                        [local_name, pool_source, description, payload.get("run_id"), payload.get("created_at"), synced_at, local_id],
                    )
                    db.execute("DELETE FROM stock_pool_symbols WHERE pool_id=?", [local_id])
                else:
                    db.execute(
                        "INSERT INTO stock_pools(id,name,created_at,source,cloud_pool_id,description,run_id,cloud_created_at,synced_at) VALUES(?,?,?,?,?,?,?,?,?)",
                        [local_id, local_name, synced_at, pool_source, cloud_id, description, payload.get("run_id"), payload.get("created_at"), synced_at],
                    )
                db.executemany(
                    "INSERT INTO stock_pool_symbols(pool_id,symbol,added_at) VALUES(?,?,?)",
                    [(local_id, symbol, synced_at) for symbol in symbols],
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("同名股票池已存在；请重命名本地股票池后重新同步") from exc
        return next(pool for pool in self.list() if pool["id"] == local_id)
