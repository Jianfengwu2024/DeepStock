from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bnhmm.data.source import BrokerBarSource, DuckDBBarSource


def _make_database(path: Path, rows: list[tuple[object, ...]]) -> None:
    duckdb = pytest.importorskip("duckdb")
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            CREATE TABLE kline_1d (
                symbol VARCHAR NOT NULL,
                timestamp DATE NOT NULL,
                circ_mv DOUBLE
            )
            """
        )
        connection.executemany("INSERT INTO kline_1d VALUES (?, ?, ?)", rows)


def test_duckdb_source_fetches_only_bounded_historical_market_caps(
    tmp_path: Path,
) -> None:
    database = tmp_path / "sector.duckdb"
    _make_database(
        database,
        [
            ("000001", "2023-12-29", 50.0),
            ("000001", "2024-01-31", 100.0),
            ("000002", "2024-01-31", 300.0),
            ("000003", "2024-01-31", 900.0),
            ("000001", "2024-03-01", 200.0),
        ],
    )

    rows = DuckDBBarSource(str(database)).fetch_market_caps(
        ["000002", "000001"], "2024-01-01", "2024-03-01"
    )

    assert [(row.symbol, row.session, row.circ_mv) for row in rows] == [
        ("000001", np.datetime64("2024-01-31"), 100.0),
        ("000002", np.datetime64("2024-01-31"), 300.0),
    ]


def test_market_cap_source_ignores_nonpositive_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    database = tmp_path / "sector.duckdb"
    _make_database(
        database,
        [
            ("000001", "2024-01-31", 100.0),
            ("000002", "2024-01-31", 0.0),
            ("000003", "2024-01-31", -1.0),
            ("000004", "2024-01-31", float("nan")),
        ],
    )

    rows = DuckDBBarSource(str(database)).fetch_market_caps(
        ["000001", "000002", "000003", "000004"],
        "2024-01-01",
        "2024-02-01",
    )

    assert [(row.symbol, row.circ_mv) for row in rows] == [("000001", 100.0)]


def test_market_cap_source_rejects_duplicate_symbol_sessions(tmp_path: Path) -> None:
    database = tmp_path / "sector.duckdb"
    _make_database(
        database,
        [
            ("000001", "2024-01-31", 100.0),
            ("000001", "2024-01-31", 110.0),
        ],
    )

    with pytest.raises(ValueError, match="重复流通市值"):
        DuckDBBarSource(str(database)).fetch_market_caps(
            ["000001"], "2024-01-01", "2024-02-01"
        )


def test_broker_source_uses_the_same_bounded_market_cap_contract() -> None:
    source = BrokerBarSource("http://127.0.0.1:9000/api/v1/db")
    calls: list[tuple[str, list[object]]] = []

    def query(sql: str, params: list[object] | None = None) -> dict[str, object]:
        calls.append((sql, list(params or [])))
        return {
            "columns": ["symbol", "session", "circ_mv"],
            "rows": [
                ["000001", "2024-01-31", 100.0],
                ["000002", "2024-01-31", 300.0],
            ],
        }

    source.query = query  # type: ignore[method-assign]
    rows = source.fetch_market_caps(
        ["000002", "000001"], "2024-01-01", "2024-03-01"
    )

    assert [row.symbol for row in rows] == ["000001", "000002"]
    assert "kline_1d" in calls[0][0]
    assert "timestamp >= ?" in calls[0][0]
    assert "timestamp < ?" in calls[0][0]
    assert calls[0][1] == ["000001", "000002", "2024-01-01", "2024-03-01"]
