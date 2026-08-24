"""Export and import real sector-index minute bars in bounded chunks."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

from bnhmm.data.sector_index import (
    SectorIndexMinuteAudit,
    SectorIndexMinuteRow,
    audit_sector_index_minutes,
    upsert_sector_index_minutes,
)
from bnhmm.data.source import BrokerBarSource, SectorIndexMinuteDataError


def _as_date(value: str) -> date:
    return date.fromisoformat(value)


def _month_chunks(start: date, end: date) -> Iterator[tuple[date, date]]:
    current = start
    while current < end:
        if current.month == 12:
            boundary = date(current.year + 1, 1, 1)
        else:
            boundary = date(current.year, current.month + 1, 1)
        following = min(boundary, end)
        yield current, following
        current = following


def export_from_broker(
    broker_url: str,
    output_path: str | Path,
    *,
    dataset_id: str,
    symbol: str,
    start: date,
    end: date,
    source_table: str = "sector_index_1m",
) -> int:
    if source_table not in {"sector_index_1m", "kline_1m"}:
        raise ValueError("source_table must be sector_index_1m or kline_1m")
    source = BrokerBarSource(broker_url)
    records: list[tuple[object, ...]] = []
    for chunk_start, chunk_end in _month_chunks(start, end):
        if source_table == "sector_index_1m":
            sql = (
                "SELECT symbol, timestamp, open, high, low, close, volume, amount "
                "FROM sector_index_1m WHERE dataset_id = ? AND symbol = ? "
                "AND timestamp >= ? AND timestamp < ? ORDER BY timestamp"
            )
            params = [
                dataset_id,
                symbol,
                chunk_start.isoformat(),
                chunk_end.isoformat(),
            ]
        else:
            sql = (
                "SELECT symbol, timestamp, open, high, low, close, volume, amount "
                "FROM kline_1m WHERE symbol = ? AND timestamp >= ? AND timestamp < ? "
                "ORDER BY timestamp"
            )
            params = [symbol, chunk_start.isoformat(), chunk_end.isoformat()]
        payload = source.query(sql, params)
        columns = payload.get("columns", [])
        for raw in payload.get("rows", []):
            row = {
                column: source._decode(value)
                for column, value in zip(columns, raw, strict=True)
            }
            records.append(tuple(row[name] for name in (
                "symbol", "timestamp", "open", "high", "low", "close", "volume", "amount"
            )))
    if not records:
        raise SectorIndexMinuteDataError(
            f"缺少真实板块一分钟指数数据：{dataset_id}/{symbol}"
        )
    timestamps = {(str(row[0]), str(row[1])) for row in records}
    if len(timestamps) != len(records):
        raise SectorIndexMinuteDataError("真实板块一分钟指数数据包含重复时间戳")

    import duckdb

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as connection:
        connection.execute(
            """
            CREATE TABLE exported(
                symbol VARCHAR, timestamp TIMESTAMP, open DOUBLE, high DOUBLE,
                low DOUBLE, close DOUBLE, volume BIGINT, amount DOUBLE
            )
            """
        )
        connection.executemany(
            "INSERT INTO exported VALUES (?, ?, ?, ?, ?, ?, ?, ?)", records
        )
        connection.execute(
            "COPY exported TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(output)]
        )
    return len(records)


def import_parquet(
    input_path: str | Path,
    database_path: str | Path,
    *,
    dataset_id: str,
    symbol: str,
) -> SectorIndexMinuteAudit:
    import duckdb

    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with duckdb.connect() as connection:
        raw_rows = connection.execute(
            """
            SELECT symbol, timestamp, open, high, low, close, volume, amount
            FROM read_parquet(?) WHERE symbol = ? ORDER BY timestamp
            """,
            [str(source), symbol],
        ).fetchall()
    if not raw_rows:
        raise SectorIndexMinuteDataError(
            f"导入文件不包含真实板块一分钟指数数据：{dataset_id}/{symbol}"
        )
    rows = tuple(
        SectorIndexMinuteRow(
            symbol=str(row[0]),
            timestamp=row[1],
            open=float(row[2]),
            high=float(row[3]),
            low=float(row[4]),
            close=float(row[5]),
            volume=int(row[6]),
            amount=float(row[7]),
        )
        for row in raw_rows
    )
    upsert_sector_index_minutes(database_path, dataset_id=dataset_id, rows=rows)
    return audit_sector_index_minutes(
        database_path, dataset_id=dataset_id, symbol=symbol
    )


def _print_audit(audit: SectorIndexMinuteAudit) -> None:
    print(f"rows={audit.rows}")
    print(f"distinct_timestamps={audit.distinct_timestamps}")
    print(f"duplicate_rows={audit.duplicate_rows}")
    print(f"first_timestamp={audit.first_timestamp}")
    print(f"last_timestamp={audit.last_timestamp}")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export")
    export.add_argument("--broker-url", required=True)
    export.add_argument("--output-path", required=True)
    export.add_argument("--dataset-id", required=True)
    export.add_argument("--symbol", required=True)
    export.add_argument("--start", type=_as_date, required=True)
    export.add_argument("--end", type=_as_date, required=True)
    export.add_argument(
        "--source-table",
        choices=("sector_index_1m", "kline_1m"),
        default="sector_index_1m",
    )

    import_command = subparsers.add_parser("import")
    import_command.add_argument("--input-path", required=True)
    import_command.add_argument("--database-path", required=True)
    import_command.add_argument("--dataset-id", required=True)
    import_command.add_argument("--symbol", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--database-path", required=True)
    audit.add_argument("--dataset-id", required=True)
    audit.add_argument("--symbol", required=True)

    args = parser.parse_args()
    if args.command == "export":
        if args.end <= args.start:
            parser.error("end must be after start")
        rows = export_from_broker(
            args.broker_url,
            args.output_path,
            dataset_id=args.dataset_id,
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            source_table=args.source_table,
        )
        print(f"exported_rows={rows}")
        return 0
    if args.command == "import":
        result = import_parquet(
            args.input_path,
            args.database_path,
            dataset_id=args.dataset_id,
            symbol=args.symbol,
        )
        _print_audit(result)
        return 0
    result = audit_sector_index_minutes(
        args.database_path, dataset_id=args.dataset_id, symbol=args.symbol
    )
    _print_audit(result)
    return 0 if result.rows and not result.duplicate_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
