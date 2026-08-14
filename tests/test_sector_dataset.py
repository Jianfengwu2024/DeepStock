from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bnhmm.data.sector_dataset import SectorDatasetCatalog


class SectorDatasetCatalogTests(unittest.TestCase):
    def test_discovers_available_members_from_immutable_duckdb(self) -> None:
        try:
            import duckdb
        except ImportError:
            self.skipTest("duckdb is not installed")

        with TemporaryDirectory() as directory:
            path = Path(directory, "semiconductor.duckdb")
            with duckdb.connect(str(path)) as connection:
                connection.execute(
                    """
                    CREATE TABLE sector_datasets AS SELECT
                        'semiconductor' AS dataset_id,
                        '半导体板块' AS name,
                        '同花顺行业' AS source,
                        '881121' AS source_code,
                        3 AS member_count,
                        2 AS available_member_count,
                        DATE '2024-01-01' AS data_start,
                        DATE '2026-08-04' AS data_end,
                        'test' AS description
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE sector_dataset_members(
                        dataset_id VARCHAR, symbol VARCHAR, available BOOLEAN
                    )
                    """
                )
                connection.executemany(
                    "INSERT INTO sector_dataset_members VALUES (?, ?, ?)",
                    [
                        ("semiconductor", "000001", True),
                        ("semiconductor", "000002", True),
                        ("semiconductor", "000003", False),
                    ],
                )

            dataset = SectorDatasetCatalog(directory).get("semiconductor")

        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.name, "半导体板块")
        self.assertEqual(dataset.symbols, ("000001", "000002"))
        self.assertEqual(dataset.available_member_count, 2)


if __name__ == "__main__":
    unittest.main()
