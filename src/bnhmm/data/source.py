"""Bounded market-bar access through DuckDB or its HTTP broker."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from ..types import BarBatch


_TABLES = {
    "1m": "kline_1m",
    "60m": "kline_60m",
    "1d": "kline_1d",
}


class BrokerError(RuntimeError):
    pass


class SectorIndexMinuteDataError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MarketCapRecord:
    symbol: str
    session: np.datetime64
    circ_mv: float

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("market-cap symbol cannot be empty")
        if np.isnat(self.session):
            raise ValueError("market-cap session cannot be NaT")
        if not np.isfinite(self.circ_mv) or self.circ_mv <= 0.0:
            raise ValueError("market-cap value must be finite and positive")


def _market_cap_records(records: list[dict[str, Any]]) -> tuple[MarketCapRecord, ...]:
    rows: list[MarketCapRecord] = []
    seen: set[tuple[str, int]] = set()
    for record in records:
        value = float(record["circ_mv"])
        if not np.isfinite(value) or value <= 0.0:
            continue
        symbol = str(record["symbol"])
        session = np.datetime64(record["session"], "D")
        key = (symbol, int(session.astype(np.int64)))
        if key in seen:
            raise ValueError(f"重复流通市值记录：{symbol}/{session}")
        seen.add(key)
        rows.append(MarketCapRecord(symbol=symbol, session=session, circ_mv=value))
    return tuple(sorted(rows, key=lambda row: (row.session, row.symbol)))


def _market_cap_query(symbols: list[str] | tuple[str, ...]) -> tuple[str, list[str]]:
    selected = sorted(set(symbols))
    if not selected or any(not symbol for symbol in selected):
        raise ValueError("at least one non-empty market-cap symbol is required")
    placeholders = ", ".join("?" for _ in selected)
    query = (
        "SELECT symbol, timestamp AS session, circ_mv FROM kline_1d "
        f"WHERE symbol IN ({placeholders}) AND timestamp >= ? AND timestamp < ? "
        "AND circ_mv IS NOT NULL ORDER BY timestamp, symbol"
    )
    return query, selected


class BrokerBarSource:
    """Read-only source that never opens the broker-owned DuckDB file."""

    def __init__(
        self,
        broker_url: str,
        timeout_seconds: float = 120.0,
        token: str | None = None,
    ) -> None:
        self.broker_url = broker_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.token = token if token is not None else os.getenv(
            "BNHMM_MARKET_BROKER_TOKEN", os.getenv("QUANT_DB_BROKER_TOKEN", "")
        )

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: str | date | datetime,
        end: str | date | datetime,
    ) -> BarBatch:
        table = _TABLES.get(timeframe)
        if table is None:
            raise ValueError(f"unsupported timeframe: {timeframe}")
        if not symbol:
            raise ValueError("symbol cannot be empty")

        lower, upper = self._as_datetime(start), self._as_datetime(end)
        if upper <= lower:
            raise ValueError("end must be later than start")
        records: list[dict[str, Any]] = []
        cursor = lower
        while cursor < upper:
            month_end = datetime(
                cursor.year + (cursor.month == 12),
                1 if cursor.month == 12 else cursor.month + 1,
                1,
                tzinfo=cursor.tzinfo,
            )
            chunk_end = min(upper, month_end)
            payload = self.query(
                f"SELECT timestamp, open, high, low, close, volume "
                f"FROM {table} WHERE symbol = ? AND timestamp >= ? AND timestamp < ? "
                "AND COALESCE(paused, false) = false AND COALESCE(suspicious, false) = false "
                "ORDER BY timestamp",
                [symbol, self._iso(cursor), self._iso(chunk_end)],
            )
            columns = payload.get("columns", [])
            records.extend(
                {column: self._decode(value) for column, value in zip(columns, row, strict=True)}
                for row in payload.get("rows", [])
            )
            cursor = chunk_end
        return BarBatch.from_records(symbol, records)

    def fetch_sector_index(
        self,
        dataset_id: str,
        symbol: str,
        start: str | date | datetime,
        end: str | date | datetime,
    ) -> BarBatch:
        payload = self.query(
            "SELECT timestamp, open, high, low, close, volume "
            "FROM sector_index_1d WHERE dataset_id = ? AND symbol = ? "
            "AND timestamp >= ? AND timestamp < ? ORDER BY timestamp",
            [dataset_id, symbol, self._iso(start), self._iso(end)],
        )
        columns = payload.get("columns", [])
        records = [
            {column: self._decode(value) for column, value in zip(columns, row, strict=True)}
            for row in payload.get("rows", [])
        ]
        return BarBatch.from_records(symbol, records)

    def fetch_sector_index_minutes(
        self,
        dataset_id: str,
        symbol: str,
        start: str | date | datetime,
        end: str | date | datetime,
    ) -> BarBatch:
        if not dataset_id or not symbol:
            raise ValueError("dataset_id and sector index symbol are required")
        payload = self.query(
            "SELECT timestamp, open, high, low, close, volume "
            "FROM sector_index_1m WHERE dataset_id = ? AND symbol = ? "
            "AND timestamp >= ? AND timestamp < ? ORDER BY timestamp",
            [dataset_id, symbol, self._iso(start), self._iso(end)],
        )
        columns = payload.get("columns", [])
        records = [
            {column: self._decode(value) for column, value in zip(columns, row, strict=True)}
            for row in payload.get("rows", [])
        ]
        if not records:
            raise SectorIndexMinuteDataError(
                f"缺少真实板块一分钟指数数据：{dataset_id}/{symbol}"
            )
        return BarBatch.from_records(symbol, records)

    def fetch_market_caps(
        self,
        symbols: list[str] | tuple[str, ...],
        start: str | date | datetime,
        end: str | date | datetime,
    ) -> tuple[MarketCapRecord, ...]:
        sql, selected = _market_cap_query(symbols)
        payload = self.query(
            sql,
            [*selected, self._iso(start), self._iso(end)],
        )
        columns = payload.get("columns", [])
        records = [
            {column: self._decode(value) for column, value in zip(columns, row, strict=True)}
            for row in payload.get("rows", [])
        ]
        return _market_cap_records(records)

    def list_symbols(self, limit: int | None = None) -> list[str]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        query = "SELECT symbol FROM stock_universe ORDER BY symbol"
        params: list[Any] = []
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        payload = self.query(query, params)
        columns = payload.get("columns", [])
        if "symbol" not in columns:
            raise BrokerError("DuckDB broker did not return the symbol column")
        symbol_column = columns.index("symbol")
        return sorted({str(row[symbol_column]) for row in payload.get("rows", [])})

    def query(self, sql: str, params: list[Any] | None = None) -> dict[str, Any]:
        body = json.dumps(
            {"sql": sql, "params": params or [], "db_type": "duckdb", "read_only": True}
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Database-Broker-Token"] = self.token
        request = Request(
            f"{self.broker_url}/query",
            data=body,
            headers=headers,
            method="POST",
        )
        for attempt in range(6):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.load(response)
                break
            except HTTPError as exc:
                if exc.code == 409 and attempt < 5:
                    time.sleep(min(0.25 * (2**attempt), 4.0))
                    continue
                raise BrokerError(f"DuckDB broker query failed: {exc}") from exc
            except (URLError, TimeoutError) as exc:
                raise BrokerError(f"DuckDB broker query failed: {exc}") from exc
        else:  # pragma: no cover - the loop always breaks or raises
            raise BrokerError("DuckDB broker remained busy")
        if "detail" in payload:
            raise BrokerError(str(payload["detail"]))
        return payload

    @staticmethod
    def _decode(value: Any) -> Any:
        if isinstance(value, dict) and "__db_type__" in value:
            return value.get("value")
        return value

    @staticmethod
    def _iso(value: str | date | datetime) -> str:
        return value if isinstance(value, str) else value.isoformat()

    @staticmethod
    def _as_datetime(value: str | date | datetime) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("date range must use ISO date or datetime values") from exc


class DuckDBBarSource:
    """Read bounded batches from a static DuckDB replica in read-only mode."""

    def __init__(self, database_path: str) -> None:
        if not database_path:
            raise ValueError("database_path cannot be empty")
        self.database_path = database_path

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: str | date | datetime,
        end: str | date | datetime,
    ) -> BarBatch:
        table = _TABLES.get(timeframe)
        if table is None:
            raise ValueError(f"unsupported timeframe: {timeframe}")
        if not symbol:
            raise ValueError("symbol cannot be empty")
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("DuckDBBarSource requires the 'data' extra") from exc

        query = (
            f"SELECT timestamp, open, high, low, close, volume FROM {table} "
            "WHERE symbol = ? AND timestamp >= ? AND timestamp < ? "
            "AND COALESCE(paused, false) = false "
            "AND COALESCE(suspicious, false) = false ORDER BY timestamp"
        )
        with duckdb.connect(self.database_path, read_only=True) as connection:
            rows = connection.execute(
                query,
                [symbol, BrokerBarSource._iso(start), BrokerBarSource._iso(end)],
            ).fetchall()
        columns = ("timestamp", "open", "high", "low", "close", "volume")
        return BarBatch.from_records(symbol, (dict(zip(columns, row)) for row in rows))

    def fetch_sector_index(
        self,
        dataset_id: str,
        symbol: str,
        start: str | date | datetime,
        end: str | date | datetime,
    ) -> BarBatch:
        if not dataset_id or not symbol:
            raise ValueError("dataset_id and sector index symbol are required")
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("DuckDBBarSource requires the 'data' extra") from exc
        query = (
            "SELECT timestamp, open, high, low, close, volume FROM sector_index_1d "
            "WHERE dataset_id = ? AND symbol = ? AND timestamp >= ? AND timestamp < ? "
            "ORDER BY timestamp"
        )
        with duckdb.connect(self.database_path, read_only=True) as connection:
            rows = connection.execute(
                query,
                [
                    dataset_id,
                    symbol,
                    BrokerBarSource._iso(start),
                    BrokerBarSource._iso(end),
                ],
            ).fetchall()
        columns = ("timestamp", "open", "high", "low", "close", "volume")
        return BarBatch.from_records(symbol, (dict(zip(columns, row)) for row in rows))

    def fetch_sector_index_minutes(
        self,
        dataset_id: str,
        symbol: str,
        start: str | date | datetime,
        end: str | date | datetime,
    ) -> BarBatch:
        if not dataset_id or not symbol:
            raise ValueError("dataset_id and sector index symbol are required")
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("DuckDBBarSource requires the 'data' extra") from exc
        query = (
            "SELECT timestamp, open, high, low, close, volume "
            "FROM sector_index_1m WHERE dataset_id = ? AND symbol = ? "
            "AND timestamp >= ? AND timestamp < ? ORDER BY timestamp"
        )
        with duckdb.connect(self.database_path, read_only=True) as connection:
            exists = connection.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema = 'main' AND table_name = 'sector_index_1m'"
            ).fetchone()[0]
            if not exists:
                raise SectorIndexMinuteDataError(
                    "缺少真实板块一分钟指数数据：sector_index_1m 表不存在"
                )
            rows = connection.execute(
                query,
                [
                    dataset_id,
                    symbol,
                    BrokerBarSource._iso(start),
                    BrokerBarSource._iso(end),
                ],
            ).fetchall()
        if not rows:
            raise SectorIndexMinuteDataError(
                f"缺少真实板块一分钟指数数据：{dataset_id}/{symbol}"
            )
        columns = ("timestamp", "open", "high", "low", "close", "volume")
        return BarBatch.from_records(symbol, (dict(zip(columns, row)) for row in rows))

    def fetch_market_caps(
        self,
        symbols: list[str] | tuple[str, ...],
        start: str | date | datetime,
        end: str | date | datetime,
    ) -> tuple[MarketCapRecord, ...]:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("DuckDBBarSource requires the 'data' extra") from exc
        query, selected = _market_cap_query(symbols)
        params = [*selected, BrokerBarSource._iso(start), BrokerBarSource._iso(end)]
        with duckdb.connect(self.database_path, read_only=True) as connection:
            rows = connection.execute(query, params).fetchall()
        columns = ("symbol", "session", "circ_mv")
        return _market_cap_records([dict(zip(columns, row)) for row in rows])

    def list_symbols(self, limit: int | None = None) -> list[str]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("DuckDBBarSource requires the 'data' extra") from exc
        query = "SELECT DISTINCT symbol FROM kline_1m ORDER BY symbol"
        params: list[int] = []
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with duckdb.connect(self.database_path, read_only=True) as connection:
            return [str(row[0]) for row in connection.execute(query, params).fetchall()]


def create_bar_source(
    source: str,
    timeout_seconds: float = 120.0,
    broker_token: str | None = None,
):
    """Select the local read-only broker or a static DuckDB replica."""
    value = str(source).strip()
    if value.lower().startswith(("http://", "https://")):
        return BrokerBarSource(
            value, timeout_seconds=timeout_seconds, token=broker_token,
        )
    return DuckDBBarSource(value)


def describe_bar_source(source: str) -> str:
    value = str(source).strip()
    if value.lower().startswith(("http://", "https://")):
        return value.rstrip("/")
    from pathlib import Path

    return str(Path(value).expanduser().resolve())

