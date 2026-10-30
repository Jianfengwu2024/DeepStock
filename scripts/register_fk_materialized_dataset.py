"""Register the completed local semiconductor FK DuckDB in the DeepStock UI catalog."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from bnhmm.data.local_dataset import LocalMarketDatasetStore


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "local-state" / "runs" / "fk-semiconductor-duckdb-20261023-20261025-143121"
PROVENANCE = ROOT / "local-state" / "broker-fk-datasets" / "fk-semiconductor-duckdb-20261023" / "provenance.json"
DATABASE = ROOT / "local-state" / "broker-fk-datasets" / "fk-semiconductor-duckdb-20261023" / "fk-semiconductor-duckdb-20261023.duckdb"
CATALOG_ROOT = ROOT / "local-state" / "market-datasets"


def main() -> None:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    coverage = {
        "1m": {
            item["symbol"]: {
                "rows": int(item["rows"]),
                "start": item["first"],
                "end": item["last"],
            }
            for item in provenance["materialized_coverage"]
        }
    }
    row_counts = {"1m": int(provenance["rows"])}
    store = LocalMarketDatasetStore(CATALOG_ROOT)
    dataset = store.register_materialized(
        dataset_id=provenance["dataset_id"],
        name="FK-贝叶斯-半导体（本地 DuckDB）",
        symbols=list(provenance["symbols"]),
        intervals=["1m"],
        start_date=date.fromisoformat(provenance["warmup_start"]),
        last_refresh_date=date.fromisoformat(provenance["evaluation_end_inclusive"]),
        database_path=DATABASE,
        row_counts=row_counts,
        coverage=coverage,
        status="ready",
    )
    catalog = {
        "dataset_id": dataset["dataset_id"],
        "database_path": dataset["database_path"],
        "catalog_path": str(store.catalog_path),
        "symbols": dataset["member_count"],
        "rows_1m": row_counts["1m"],
        "first": provenance["warmup_start"],
        "last": provenance["evaluation_end_inclusive"],
        "run_report": str(RUN / "半导体运行结果.md"),
    }
    (RUN / "dataset-registration.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(catalog, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
