"""Bounded market-bar access through the shared DuckDB HTTP broker."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..types import BarBatch


_TABLES = {
    "1m": "kline_1m",
    "60m": "kline_60m",
    "1d": "kline_1d",
}


class BrokerError(RuntimeError):
    pass


class BrokerBarSource:
    """Read-only source that never opens the broker-owned DuckDB file."""

    def __init__(self, broker_url: str, timeout_seconds: float = 120.0) -> None:
        self.broker_url = broker_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

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

        payload = self.query(
            f"SELECT timestamp, open, high, low, close, volume "
            f"FROM {table} WHERE symbol = ? AND timestamp >= ? AND timestamp < ? "
            "AND COALESCE(paused, false) = false AND COALESCE(suspicious, false) = false "
            "ORDER BY timestamp",
            [symbol, self._iso(start), self._iso(end)],
        )
        columns = payload.get("columns", [])
        records = [
            {column: self._decode(value) for column, value in zip(columns, row, strict=True)}
            for row in payload.get("rows", [])
        ]
        return BarBatch.from_records(symbol, records)

    def query(self, sql: str, params: list[Any] | None = None) -> dict[str, Any]:
        body = json.dumps(
            {"sql": sql, "params": params or [], "db_type": "duckdb", "read_only": True}
        ).encode("utf-8")
        request = Request(
            f"{self.broker_url}/query",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise BrokerError(f"DuckDB broker query failed: {exc}") from exc
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

