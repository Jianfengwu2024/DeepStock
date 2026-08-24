"""Fetch and persist real sector-index daily bars."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable


_CALLBACK = re.compile(r"^[A-Za-z0-9_]+\((.*)\)\s*;?\s*$", re.DOTALL)


@dataclass(frozen=True, slots=True)
class SectorIndexRow:
    symbol: str
    timestamp: date | str
    open: float
    high: float
    low: float
    close: float
    volume: int
    amount: float

    def as_tuple(self, dataset_id: str) -> tuple[object, ...]:
        timestamp = (
            datetime.strptime(self.timestamp, "%Y-%m-%d").date()
            if isinstance(self.timestamp, str)
            else self.timestamp
        )
        return (
            dataset_id,
            self.symbol,
            timestamp,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.amount,
            "10jqka",
        )


def parse_10jqka_year(payload: str, *, symbol: str) -> tuple[SectorIndexRow, ...]:
    """Parse one bounded yearly response without evaluating its callback."""

    match = _CALLBACK.fullmatch(payload.strip())
    if match is None:
        raise ValueError("sector-index response is not a supported callback")
    body = json.loads(match.group(1))
    encoded = body.get("data")
    if not isinstance(encoded, str):
        raise ValueError("sector-index response does not contain daily data")
    rows: list[SectorIndexRow] = []
    for raw in encoded.split(";"):
        if not raw:
            continue
        fields = raw.split(",")
        if len(fields) < 7:
            raise ValueError("sector-index daily row is incomplete")
        timestamp = datetime.strptime(fields[0], "%Y%m%d").date()
        rows.append(
            SectorIndexRow(
                symbol=symbol,
                timestamp=timestamp,
                open=float(fields[1]),
                high=float(fields[2]),
                low=float(fields[3]),
                close=float(fields[4]),
                volume=int(float(fields[5])),
                amount=float(fields[6]),
            )
        )
    if not rows:
        raise ValueError("sector-index yearly response is empty")
    return tuple(rows)


def upsert_sector_index(
    database_path: str | Path,
    *,
    dataset_id: str,
    rows: Iterable[SectorIndexRow],
) -> int:
    """Create the dedicated table and idempotently store index bars."""

    if not dataset_id:
        raise ValueError("dataset_id cannot be empty")
    materialized = tuple(rows)
    if not materialized:
        return 0
    import duckdb

    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sector_index_1d(
                dataset_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                timestamp DATE NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume BIGINT NOT NULL,
                amount DOUBLE NOT NULL,
                source VARCHAR NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
                PRIMARY KEY(dataset_id, symbol, timestamp)
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO sector_index_1d(
                dataset_id, symbol, timestamp, open, high, low, close,
                volume, amount, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id, symbol, timestamp) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                amount = excluded.amount,
                source = excluded.source,
                updated_at = now()
            """,
            [row.as_tuple(dataset_id) for row in materialized],
        )
    return len(materialized)
