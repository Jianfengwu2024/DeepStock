from __future__ import annotations

import unittest
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.sync_sector_index_1m import export_from_broker, import_parquet

from bnhmm.data.sector_index import (
    SectorIndexRow,
    SectorIndexMinuteRow,
    audit_sector_index_minutes,
    parse_10jqka_year,
    upsert_sector_index,
    upsert_sector_index_minutes,
)
from bnhmm.data.source import (
    BrokerBarSource,
    DuckDBBarSource,
    SectorIndexMinuteDataError,
)


class SectorIndexSourceTests(unittest.TestCase):
    def test_parses_bounded_year_response(self) -> None:
        payload = (
            'quotebridge_v6_line_bk_881121_01_2024('
            '{"data":"20240102,7035.381,7040.000,6925.206,6927.422,'
            '1003518330,34598815000.000,,,,0;'
            '20240103,6902.829,6910.968,6747.251,6795.765,'
            '1048345130,35016473000.000,,,,0"})'
        )

        rows = parse_10jqka_year(payload, symbol="881121")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].symbol, "881121")
        self.assertEqual(str(rows[0].timestamp), "2024-01-02")
        self.assertAlmostEqual(rows[0].open, 7035.381)
        self.assertAlmostEqual(rows[0].close, 6927.422)
        self.assertEqual(rows[0].volume, 1003518330)

    def test_upsert_is_idempotent_and_source_reads_only_requested_index(self) -> None:
        try:
            import duckdb  # noqa: F401
        except ImportError:
            self.skipTest("duckdb is not installed")

        rows = (
            SectorIndexRow(
                symbol="881121",
                timestamp="2024-01-02",
                open=7035.381,
                high=7040.0,
                low=6925.206,
                close=6927.422,
                volume=1003518330,
                amount=34598815000.0,
            ),
            SectorIndexRow(
                symbol="881121",
                timestamp="2024-01-03",
                open=6902.829,
                high=6910.968,
                low=6747.251,
                close=6795.765,
                volume=1048345130,
                amount=35016473000.0,
            ),
        )
        with TemporaryDirectory() as directory:
            database = Path(directory, "sector.duckdb")
            upsert_sector_index(database, dataset_id="semiconductor", rows=rows)
            upsert_sector_index(database, dataset_id="semiconductor", rows=rows)

            batch = DuckDBBarSource(str(database)).fetch_sector_index(
                "semiconductor", "881121", "2024-01-01", "2024-02-01"
            )

        self.assertEqual(batch.symbol, "881121")
        self.assertEqual(len(batch), 2)
        self.assertAlmostEqual(float(batch.close[-1]), 6795.765)

    def test_minute_upsert_is_idempotent_and_source_reads_requested_index(self) -> None:
        try:
            import duckdb  # noqa: F401
        except ImportError:
            self.skipTest("duckdb is not installed")

        rows = (
            SectorIndexMinuteRow(
                symbol="881121",
                timestamp=datetime(2024, 1, 2, 9, 30),
                open=7035.381,
                high=7038.0,
                low=7034.0,
                close=7037.5,
                volume=10_000,
                amount=20_000.0,
            ),
            SectorIndexMinuteRow(
                symbol="881121",
                timestamp=datetime(2024, 1, 2, 9, 31),
                open=7037.5,
                high=7039.0,
                low=7036.0,
                close=7038.5,
                volume=12_000,
                amount=24_000.0,
            ),
        )
        with TemporaryDirectory() as directory:
            database = Path(directory, "sector.duckdb")
            upsert_sector_index_minutes(
                database, dataset_id="semiconductor", rows=rows
            )
            upsert_sector_index_minutes(
                database, dataset_id="semiconductor", rows=rows
            )

            batch = DuckDBBarSource(str(database)).fetch_sector_index_minutes(
                "semiconductor", "881121", "2024-01-02", "2024-01-03"
            )

        self.assertEqual(batch.symbol, "881121")
        self.assertEqual(len(batch), 2)
        self.assertEqual(str(batch.timestamp[0]), "2024-01-02T09:30:00.000000000")
        self.assertAlmostEqual(float(batch.close[-1]), 7038.5)

    def test_missing_minute_index_never_falls_back_to_daily_data(self) -> None:
        try:
            import duckdb  # noqa: F401
        except ImportError:
            self.skipTest("duckdb is not installed")

        with TemporaryDirectory() as directory:
            database = Path(directory, "sector.duckdb")
            upsert_sector_index(
                database,
                dataset_id="semiconductor",
                rows=(
                    SectorIndexRow(
                        symbol="881121",
                        timestamp="2024-01-02",
                        open=7035.381,
                        high=7040.0,
                        low=6925.206,
                        close=6927.422,
                        volume=1003518330,
                        amount=34598815000.0,
                    ),
                ),
            )

            with self.assertRaisesRegex(
                SectorIndexMinuteDataError, "缺少真实板块一分钟指数数据"
            ):
                DuckDBBarSource(str(database)).fetch_sector_index_minutes(
                    "semiconductor", "881121", "2024-01-02", "2024-01-03"
                )

    def test_broker_source_fetches_the_dedicated_minute_index_table(self) -> None:
        source = BrokerBarSource("http://127.0.0.1:9000/api/v1/db")
        calls: list[tuple[str, list[object]]] = []

        def query(sql: str, params: list[object] | None = None) -> dict[str, object]:
            calls.append((sql, list(params or [])))
            return {
                "columns": ["timestamp", "open", "high", "low", "close", "volume"],
                "rows": [
                    ["2024-01-02T09:30:00", 7035.0, 7038.0, 7034.0, 7037.5, 10000]
                ],
            }

        source.query = query  # type: ignore[method-assign]
        batch = source.fetch_sector_index_minutes(
            "semiconductor", "881121", "2024-01-02", "2024-01-03"
        )

        self.assertEqual(len(batch), 1)
        self.assertIn("sector_index_1m", calls[0][0])
        self.assertEqual(calls[0][1][:2], ["semiconductor", "881121"])

    def test_minute_audit_reports_coverage_and_duplicate_count(self) -> None:
        try:
            import duckdb  # noqa: F401
        except ImportError:
            self.skipTest("duckdb is not installed")

        rows = (
            SectorIndexMinuteRow(
                symbol="881121",
                timestamp="2024-01-02 09:30:00",
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=1,
                amount=1.0,
            ),
            SectorIndexMinuteRow(
                symbol="881121",
                timestamp="2024-01-02 09:31:00",
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=1,
                amount=1.0,
            ),
        )
        with TemporaryDirectory() as directory:
            database = Path(directory, "sector.duckdb")
            upsert_sector_index_minutes(
                database, dataset_id="semiconductor", rows=rows
            )
            audit = audit_sector_index_minutes(
                database, dataset_id="semiconductor", symbol="881121"
            )

        self.assertEqual(audit.rows, 2)
        self.assertEqual(audit.distinct_timestamps, 2)
        self.assertEqual(audit.duplicate_rows, 0)
        self.assertEqual(str(audit.first_timestamp), "2024-01-02 09:30:00")
        self.assertEqual(str(audit.last_timestamp), "2024-01-02 09:31:00")

    def test_parquet_import_is_idempotent(self) -> None:
        try:
            import duckdb
        except ImportError:
            self.skipTest("duckdb is not installed")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "881121-1m.parquet"
            target = root / "sector.duckdb"
            connection = duckdb.connect()
            connection.execute(
                """
                COPY (
                    SELECT '881121'::VARCHAR AS symbol,
                           '2024-01-02 09:30:00'::TIMESTAMP AS timestamp,
                           1.0::DOUBLE AS open, 1.1::DOUBLE AS high,
                           0.9::DOUBLE AS low, 1.05::DOUBLE AS close,
                           100::BIGINT AS volume, 200.0::DOUBLE AS amount
                ) TO ? (FORMAT PARQUET)
                """,
                [str(parquet)],
            )
            connection.close()

            first = import_parquet(
                parquet,
                target,
                dataset_id="semiconductor",
                symbol="881121",
            )
            second = import_parquet(
                parquet,
                target,
                dataset_id="semiconductor",
                symbol="881121",
            )

        self.assertEqual(first.rows, 1)
        self.assertEqual(second.rows, 1)
        self.assertEqual(second.duplicate_rows, 0)

    def test_export_rejects_duplicate_source_timestamps(self) -> None:
        class DuplicateBroker:
            @staticmethod
            def _decode(value: object) -> object:
                return value

            def query(
                self, sql: str, params: list[object] | None = None
            ) -> dict[str, object]:
                return {
                    "columns": [
                        "symbol",
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "amount",
                    ],
                    "rows": [
                        [
                            "881121",
                            "2024-01-02T09:30:00",
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1,
                            1.0,
                        ]
                    ],
                }

        with TemporaryDirectory() as directory, patch(
            "scripts.sync_sector_index_1m.BrokerBarSource",
            return_value=DuplicateBroker(),
        ):
            with self.assertRaisesRegex(
                SectorIndexMinuteDataError, "重复时间戳"
            ):
                export_from_broker(
                    "http://broker",
                    Path(directory, "index.parquet"),
                    dataset_id="semiconductor",
                    symbol="881121",
                    start=date(2024, 1, 1),
                    end=date(2024, 3, 1),
                )


if __name__ == "__main__":
    unittest.main()
