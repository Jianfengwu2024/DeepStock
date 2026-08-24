"""Synchronize bounded yearly sector-index bars into a sector dataset."""

from __future__ import annotations

import argparse
from urllib.request import Request, urlopen

from bnhmm.data.sector_index import parse_10jqka_year, upsert_sector_index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-path", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error("end-year must not precede start-year")

    total = 0
    for year in range(args.start_year, args.end_year + 1):
        url = f"https://d.10jqka.com.cn/v6/line/bk_{args.symbol}/01/{year}.js"
        request = Request(
            url,
            headers={
                "Referer": "https://q.10jqka.com.cn/",
                "User-Agent": "Mozilla/5.0",
            },
        )
        with urlopen(request, timeout=30.0) as response:
            payload = response.read().decode("utf-8")
        rows = parse_10jqka_year(payload, symbol=args.symbol)
        total += upsert_sector_index(
            args.database_path,
            dataset_id=args.dataset_id,
            rows=rows,
        )
        print(f"{year}: {len(rows)} rows")
    print(f"stored {total} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

