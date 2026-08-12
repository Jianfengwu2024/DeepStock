"""Operational commands for data smoke checks and storage migration."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .config import ProfileConfig
from .data.profile import PriceVolumeProfileBuilder
from .data.source import BrokerBarSource
from .storage import apply_migrations


def _profile_smoke(args: argparse.Namespace) -> int:
    source = BrokerBarSource(args.broker_url, timeout_seconds=args.timeout)
    bars = source.fetch(args.symbol, "1m", args.start, args.end)
    builder = PriceVolumeProfileBuilder(ProfileConfig())
    states = builder.build(bars)
    summary = {
        "symbol": args.symbol,
        "bars": len(bars),
        "first_bar": str(bars.timestamp[0]) if len(bars) else None,
        "last_bar": str(bars.timestamp[-1]) if len(bars) else None,
        "hour_states": sum(state.timeframe == "60m" for state in states),
        "day_states": sum(state.timeframe == "1d" for state in states),
        "profile_bins": len(states[0].distribution) if states else 0,
        "all_profiles_normalized": all(
            abs(float(state.distribution.sum()) - 1.0) < 1e-9 for state in states
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if states else 2


def _migrate(args: argparse.Namespace) -> int:
    applied = apply_migrations(args.dsn)
    print(json.dumps({"applied": applied}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bnhmm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("profile-smoke")
    smoke.add_argument("--broker-url", required=True)
    smoke.add_argument("--symbol", required=True)
    smoke.add_argument("--start", required=True)
    smoke.add_argument("--end", required=True)
    smoke.add_argument("--timeout", type=float, default=180.0)
    smoke.set_defaults(handler=_profile_smoke)

    migrate = subparsers.add_parser("migrate")
    migrate.add_argument("--dsn", required=True)
    migrate.set_defaults(handler=_migrate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

