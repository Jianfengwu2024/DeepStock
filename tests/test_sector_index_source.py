from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bnhmm.data.sector_index import (
    SectorIndexRow,
    parse_10jqka_year,
    upsert_sector_index,
)
from bnhmm.data.source import DuckDBBarSource


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


if __name__ == "__main__":
    unittest.main()
