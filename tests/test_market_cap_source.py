from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bnhmm.data.source import (
    BrokerError,
    BrokerBarSource,
    DuckDBBarSource,
    create_bar_source,
    describe_bar_source,
)


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


def test_broker_source_lists_symbols_and_factory_preserves_broker_url() -> None:
    source = BrokerBarSource("http://127.0.0.1:9000/api/v1/db")
    source.query = lambda _sql, _params=None: {  # type: ignore[method-assign]
        "columns": ["symbol"],
        "rows": [["000002"], ["000001"], ["000001"]],
    }

    assert source.list_symbols(10) == ["000001", "000002"]
    assert isinstance(
        create_bar_source("http://127.0.0.1:9000/api/v1/db"), BrokerBarSource
    )
    assert describe_bar_source("http://127.0.0.1:9000/api/v1/db") == (
        "http://127.0.0.1:9000/api/v1/db"
    )


def test_broker_symbol_search_is_prefix_bounded_and_parameterized() -> None:
    source = BrokerBarSource("http://127.0.0.1:9000/api/v1/db")
    captured = {}

    def query(sql, params=None):
        captured.update(sql=sql, params=params)
        return {"columns": ["symbol"], "rows": [["000001"], ["000002"]]}

    source.query = query
    assert source.list_symbols(2, q="000") == ["000001", "000002"]
    assert "FROM stock_universe WHERE symbol LIKE ? ORDER BY symbol LIMIT ?" in captured["sql"]
    assert captured["params"] == ["000%", 2]


def test_broker_source_sends_database_token_without_putting_it_in_url() -> None:
    from unittest.mock import patch

    source = BrokerBarSource("http://127.0.0.1:9000/api/v1/db", token="test-secret")
    captured = {}

    def response(request, timeout):
        captured["url"] = request.full_url
        captured["token"] = request.get_header("X-database-broker-token")
        return _JsonResponse({"columns": ["symbol"], "rows": [["000001"]]})

    with patch("bnhmm.data.source.urlopen", side_effect=response):
        assert source.list_symbols(1) == ["000001"]
    assert captured == {
        "url": "http://127.0.0.1:9000/api/v1/db/query",
        "token": "test-secret",
    }


def test_broker_query_retries_a_busy_read_pool():
    from urllib.error import HTTPError
    from unittest.mock import patch

    source = BrokerBarSource("http://127.0.0.1:9000/api/v1/db")
    calls = 0

    def response(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(request.full_url, 409, "busy", {}, None)
        return _JsonResponse({"columns": ["value"], "rows": [[1]]})

    with patch("bnhmm.data.source.urlopen", side_effect=response), patch(
        "bnhmm.data.source.time.sleep"
    ) as sleep:
        assert source.query("SELECT 1")["rows"] == [[1]]
    assert calls == 2
    sleep.assert_called_once_with(0.25)


def test_broker_source_reports_read_pool_busy_detail():
    import io
    from urllib.error import HTTPError
    from unittest.mock import patch

    source = BrokerBarSource("http://127.0.0.1:9000/api/v1/db", busy_retries=0)
    response = HTTPError(
        "http://127.0.0.1:9000/api/v1/db/query", 409, "Conflict", {},
        io.BytesIO(b'{"detail":"DuckDB connection pool is busy"}'),
    )

    with patch("bnhmm.data.source.urlopen", side_effect=response):
        with pytest.raises(BrokerError, match="connection pool is busy"):
            source.query("SELECT 1")


def test_broker_fetch_many_scans_one_symbol_universe_range():
    source = BrokerBarSource("http://127.0.0.1:9000/api/v1/db")
    captured = {}

    def query(sql, params=None):
        captured.update(sql=sql, params=params)
        if "FROM kline_coverage" in sql:
            return {
                "columns": ["symbol", "first_ts", "last_ts"],
                "rows": [
                    ["000001", "2020-01-01", "2025-01-01"],
                    ["600000", "2020-01-01", "2025-01-01"],
                ],
            }
        return {
            "columns": ["symbol", "timestamp", "open", "high", "low", "close", "volume"],
            "rows": [
                ["000001", "2024-01-02T09:30:00", 10, 11, 9, 10, 100],
                ["600000", "2024-01-02T09:30:00", 20, 21, 19, 20, 200],
            ],
        }

    source.query = query
    result = source.fetch_many(["000001", "600000"], "1m", "2024-01-01", "2025-01-01")

    assert set(result) == {"000001", "600000"}
    assert len(result["000001"]) == len(result["600000"]) == 1
    assert "symbol IN (?,?)" in captured["sql"]
    assert "ORDER BY" not in captured["sql"]
    assert captured["params"] == ["000001", "600000", "2024-01-01T00:00:00", "2025-01-01T00:00:00"]


def test_broker_bar_fetch_splits_large_ranges_into_half_open_months() -> None:
    source = BrokerBarSource("http://127.0.0.1:9000/api/v1/db")
    calls = []

    def query(sql, params=None):
        calls.append(params)
        return {"columns": ["timestamp", "open", "high", "low", "close", "volume"], "rows": []}

    source.query = query
    assert len(source.fetch("000001", "1m", "2026-07-15", "2026-10-01")) == 0
    assert calls == [
        ["000001", "2026-07-15T00:00:00", "2026-08-01T00:00:00"],
        ["000001", "2026-08-01T00:00:00", "2026-09-01T00:00:00"],
        ["000001", "2026-09-01T00:00:00", "2026-10-01T00:00:00"],
    ]


class _JsonResponse:
    def __init__(self, payload):
        import io
        self._body = io.BytesIO(__import__("json").dumps(payload).encode())

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self._body.close()

    def __iter__(self):
        return iter(self._body)

    def read(self, *args):
        return self._body.read(*args)
