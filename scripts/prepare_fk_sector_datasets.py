"""Create local one-minute datasets for the three rolling60 FK sectors."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bnhmm.data.source import BrokerBarSource  # noqa: E402
from bnhmm.web.local_store import StockPoolStore  # noqa: E402


SECTORS = {
    "semiconductor": {"name": "半导体", "board": "BK1036"},
    "artificial_intelligence": {"name": "人工智能", "board": "BK0800"},
    "new_energy": {"name": "新能源", "board": "BK0493"},
}
BOARD_URL = "https://push2.eastmoney.com/api/qt/clist/get"
LOCAL_API = os.getenv("BNHMM_LOCAL_API", "http://127.0.0.1:8765")


def _json_request(url: str, timeout: int = 25) -> dict:
    request = Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json, text/plain, */*",
    })
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        if "push2.eastmoney.com/api/qt/clist/get?" not in url:
            raise
        target = url.removeprefix("https://").removeprefix("http://").replace("&", "%26")
        proxy_url = "https://r.jina.ai/http://" + target
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                proxy_request = Request(proxy_url, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/plain"})
                with urlopen(proxy_request, timeout=timeout) as response:
                    text = response.read(2_000_001).decode("utf-8", errors="strict")
                marker = "Markdown Content:"
                if marker not in text or len(text) > 2_000_000:
                    raise ValueError("public sector membership fallback returned invalid data")
                body = text.split(marker, 1)[1].strip()
                json_start = body.find("{")
                if json_start < 0:
                    raise ValueError("public sector source is temporarily unavailable through the fallback")
                payload, _ = json.JSONDecoder().raw_decode(body[json_start:])
                break
            except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError("current sector-membership source is unavailable; no local datasets were created") from last_error
    if not isinstance(payload, dict):
        raise RuntimeError("external board source returned an invalid response")
    return payload


def board_symbols(board: str) -> list[str]:
    params = {
        "pn": 1, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
        "fid": "f3", "fs": f"b:{board}", "fields": "f12,f14",
    }
    first = _json_request(BOARD_URL + "?" + urlencode(params))
    data = first.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("diff"), list):
        raise RuntimeError(f"board member response missing: {board}")
    total = int(data.get("total") or 0)
    if total < 4 or total > 2_000:
        raise RuntimeError(f"board membership count is outside the supported range: {board}")
    symbols = {str(row.get("f12") or "") for row in data["diff"] if isinstance(row, dict)}
    pages = math.ceil(total / 100)
    for page in range(2, pages + 1):
        params["pn"] = page
        payload = _json_request(BOARD_URL + "?" + urlencode(params))
        rows = (payload.get("data") or {}).get("diff")
        if not isinstance(rows, list):
            raise RuntimeError(f"board member page missing: {board}/{page}")
        symbols.update(str(row.get("f12") or "") for row in rows if isinstance(row, dict))
        time.sleep(0.15)
    result = sorted(symbol for symbol in symbols if len(symbol) == 6 and symbol.isdigit())
    if len(result) != total:
        raise RuntimeError(f"board membership was incomplete: {board}; expected {total}, received {len(result)}")
    return result


def local_members(source: BrokerBarSource, symbols: list[str], end: date) -> tuple[list[str], dict[str, dict[str, str]]]:
    available: dict[str, dict[str, str]] = {}
    for offset in range(0, len(symbols), 500):
        batch = symbols[offset:offset + 500]
        placeholders = ",".join("?" for _ in batch)
        result = source.query(
            "SELECT symbol,first_ts,last_ts FROM kline_coverage "
            f"WHERE interval=? AND symbol IN ({placeholders}) AND last_ts>=?",
            ["1m", *batch, end.isoformat()],
        )
        columns = result.get("columns", [])
        positions = {name: columns.index(name) for name in ("symbol", "first_ts", "last_ts")}
        for row in result.get("rows", []):
            symbol = str(row[positions["symbol"]])
            available[symbol] = {
                "first_ts": str(source._decode(row[positions["first_ts"]]))[:19],
                "last_ts": str(source._decode(row[positions["last_ts"]]))[:19],
            }
    selected = [symbol for symbol in symbols if symbol in available]
    return selected, available


def api(method: str, path: str, payload: dict | None = None):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        LOCAL_API + path,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method=method,
    )
    with urlopen(request, timeout=30) as response:
        value = json.load(response)
    if not isinstance(value, (dict, list)):
        raise RuntimeError("local DeepStock API returned an invalid response")
    return value


def prepare(evaluation_start: date | None = None) -> dict:
    source = BrokerBarSource(os.getenv("BNHMM_MARKET_SOURCE", "http://127.0.0.1:9000/api/v1/db"), timeout_seconds=90)
    latest = source.query("SELECT MAX(last_ts) AS last_ts FROM kline_coverage WHERE interval=?", ["1m"])
    raw_latest = latest.get("rows", [[None]])[0][0]
    latest_day = date.fromisoformat(str(source._decode(raw_latest))[:10])
    evaluation_end = latest_day
    evaluation_start = evaluation_start or (evaluation_end - timedelta(days=365))
    dataset_start = evaluation_start - timedelta(days=450)

    catalog = api("GET", "/api/market-datasets")
    existing_datasets = catalog if isinstance(catalog, list) else []
    pool_store = StockPoolStore(os.getenv("BNHMM_STOCK_POOL_PATH", str(ROOT / "local-state" / "stock-pools.sqlite3")))
    stamp = date.today().strftime("%Y%m%d")
    manifest = {
        "created_on": date.today().isoformat(),
        "source": "Eastmoney sector membership and local read-only 1m coverage",
        "evaluation_start": evaluation_start.isoformat(),
        "evaluation_end": evaluation_end.isoformat(),
        "dataset_start": dataset_start.isoformat(),
        "warmup_calendar_days": 450,
        "sectors": {},
    }

    prepared: dict[str, dict] = {}
    for sector_id, definition in SECTORS.items():
        members = board_symbols(definition["board"])
        symbols, coverage = local_members(source, members, evaluation_end)
        if len(symbols) < 4:
            raise RuntimeError(f"not enough local one-minute coverage for {definition['name']}")
        prepared[sector_id] = {"members": members, "symbols": symbols, "coverage": coverage}

    for sector_id, definition in SECTORS.items():
        members = prepared[sector_id]["members"]
        symbols = prepared[sector_id]["symbols"]
        coverage = prepared[sector_id]["coverage"]
        dataset_name = f"FK-Bayesian-{definition['name']}"
        existing = next((item for item in existing_datasets if item.get("name") == dataset_name), None)
        if existing:
            if set(existing.get("symbols", [])) != set(symbols):
                raise RuntimeError(f"existing local dataset membership differs: {dataset_name}; preserve it and resolve manually")
            if existing["status"] in {"queued", "updating"}:
                dataset = existing
            elif date.fromisoformat(existing["start_date"]) > dataset_start:
                raise RuntimeError(f"existing local dataset lacks FK warmup history: {dataset_name}")
            else:
                api("POST", f"/api/market-datasets/{existing['dataset_id']}/refresh")
                dataset = {**existing, "status": "queued"}
        else:
            pool_name = f"FK-Bayesian-{definition['name']}-{stamp}"
            pool = next((item for item in pool_store.list() if item["name"] == pool_name), None)
            if pool is None:
                pool = pool_store.create(pool_name)
            for symbol in symbols:
                pool_store.add_symbol(pool["id"], symbol)
            dataset = api("POST", "/api/market-datasets", {
                "name": dataset_name,
                "stock_pool_id": pool["id"],
                "intervals": ["1m"],
                "start_date": dataset_start.isoformat(),
            })
        manifest["sectors"][sector_id] = {
            "name": definition["name"],
            "membership_board": definition["board"],
            "board_members": len(members),
            "locally_covered_members": len(symbols),
            "excluded_without_1m_coverage": len(members) - len(symbols),
            "dataset_id": dataset["dataset_id"],
            "dataset_name": dataset_name,
            "status": dataset["status"],
            "symbols": symbols,
            "coverage": coverage,
        }
        print(json.dumps({key: value for key, value in manifest["sectors"][sector_id].items() if key not in {"symbols", "coverage"}}, ensure_ascii=False))

    output = ROOT / "local-state" / "fk-sector-universes.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(output), "evaluation_start": evaluation_start.isoformat(), "evaluation_end": evaluation_end.isoformat(), "dataset_start": dataset_start.isoformat()}, ensure_ascii=False))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-start", type=date.fromisoformat)
    args = parser.parse_args()
    try:
        prepare(args.evaluation_start)
        return 0
    except (HTTPError, URLError, OSError, ValueError, RuntimeError) as exc:
        print(f"FK sector data preparation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
