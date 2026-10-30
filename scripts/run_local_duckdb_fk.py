"""Run semiconductor FK using only the existing local broker-owned DuckDB."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

import duckdb
import numpy as np

from bnhmm.data.source import BrokerBarSource, BrokerError

ROOT = Path(__file__).resolve().parents[1]


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def broker_token() -> str:
    token = os.getenv("BNHMM_MARKET_BROKER_TOKEN", "")
    if token:
        return token
    config = Path(r"G:\projects\quant-platform\project\quant-platform\local-state\security\server.env")
    for line in config.read_text(encoding="utf-8-sig").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "QUANT_DB_BROKER_TOKEN":
            return value.strip().strip("\"'")
    raise RuntimeError("local broker credential missing")


def continuous_minute_mask(timestamps: np.ndarray) -> np.ndarray:
    """Exclude auction and after-hours observations from FK continuous bars."""
    minutes = (timestamps.astype("datetime64[m]") - timestamps.astype("datetime64[D]")).astype(np.int64)
    return ((minutes >= 570) & (minutes < 690)) | ((minutes >= 780) & (minutes < 900))


def shift_start_to_end(timestamps: np.ndarray) -> np.ndarray:
    """The shared market tables use start labels; FK uses minute closes."""
    continuous = continuous_minute_mask(timestamps)
    if not np.all(continuous):
        raise ValueError("minute timestamps violate the shared DuckDB start-label convention")
    return timestamps + np.timedelta64(1, "m")


def run(args: argparse.Namespace) -> None:
    state_path = args.work_root / "status.json"
    state: dict = {"sector": "semiconductor", "status": "starting", "source": args.broker, "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}

    def status(phase: str, **details: object) -> None:
        state.update(status=phase, updated_at=datetime.now(timezone.utc).isoformat(), **details)
        save_json(state_path, state)
        print(json.dumps({"status":phase, **details}, ensure_ascii=False), flush=True)

    token = broker_token()
    source = BrokerBarSource(args.broker, token=token, timeout_seconds=300, busy_retries=0)

    def metadata(route: str) -> dict:
        request = Request(args.broker.rstrip("/") + "/" + route, headers={"X-Database-Broker-Token":token})
        with urlopen(request, timeout=20) as response:
            return json.load(response)

    def wait_readable() -> None:
        last_owner = None
        while True:
            current = metadata("stats")
            maintenance = current.get("maintenance", {})
            if not maintenance.get("active"):
                return
            owner = maintenance.get("owner")
            if owner != last_owner or state["status"] != "waiting_for_local_duckdb":
                status("waiting_for_local_duckdb", maintenance_owner=owner)
                last_owner = owner
            time.sleep(10)

    def query(sql: str, params: list | None = None) -> dict:
        while True:
            wait_readable()
            try:
                return source.query(sql, params)
            except BrokerError as exc:
                if "maintenance lease is active" not in str(exc):
                    raise

    try:
        with args.members.open(encoding="utf-8-sig", newline="") as stream:
            members = list(csv.DictReader(stream))
        symbols = sorted({row["成分股代码"].split(".")[0] for row in members})
        if len(symbols) < 4 or any(len(s) != 6 or not s.isdigit() for s in symbols):
            raise ValueError("invalid semiconductor membership snapshot")
        status("queued", members=len(symbols), membership_path=str(args.members), membership_mtime=datetime.fromtimestamp(args.members.stat().st_mtime, timezone.utc).isoformat())
        health = metadata("health")
        state["shared_database_path"] = health["path"]
        placeholders = ",".join("?" for _ in symbols)
        payload = query(f"SELECT symbol,first_ts,last_ts FROM kline_coverage WHERE interval=? AND symbol IN ({placeholders}) ORDER BY symbol", ["1m", *symbols])
        columns = payload["columns"]
        coverage = [{name: source._decode(value) for name,value in zip(columns,row,strict=True)} for row in payload["rows"]]
        absent = sorted(set(symbols) - {str(row["symbol"]) for row in coverage})
        if absent and not args.allow_missing_members:
            raise ValueError(f"local DuckDB has no minute coverage for {len(absent)} members: {','.join(absent)}")
        requested_symbols = symbols.copy()
        if absent:
            symbols = [symbol for symbol in symbols if symbol not in absent]
            status("coverage_checked", members=len(symbols), requested_members=len(requested_symbols), excluded_without_local_minute_coverage=absent)
        if len(symbols) < 4:
            raise ValueError("fewer than four locally covered semiconductor members")
        end = max(date.fromisoformat(str(row["last_ts"])[:10]) for row in coverage)
        start = end - timedelta(days=365)
        lower = start - timedelta(days=450)
        exclusive_end = end + timedelta(days=1)
        dataset_id = f"fk-semiconductor-duckdb-{end:%Y%m%d}"
        dataset_root = ROOT / "local-state" / "broker-fk-datasets" / dataset_id
        dataset_root.mkdir(parents=True, exist_ok=True)
        database = dataset_root / f"{dataset_id}.duckdb"
        manifest = dataset_root / "provenance.json"
        provenance = {"dataset_id":dataset_id,"source":args.broker,"shared_database_path":health["path"],"membership_path":str(args.members),"membership_mtime":state["membership_mtime"],"symbols":symbols,"requested_symbols":requested_symbols,"excluded_without_local_minute_coverage":absent,"source_coverage":coverage,"evaluation_start":str(start),"evaluation_end_inclusive":str(end),"warmup_start":str(lower),"end_exclusive":str(exclusive_end),"minute_labels":"shared start-labelled Beijing timestamps shifted +1 minute to FK close labels","price_basis":"existing DuckDB raw OHLC; no adjustment inferred","universe_policy":"fixed local membership snapshot; historical survivorship bias possible"}
        save_json(manifest,provenance)
        status("reading_local_duckdb", dataset_id=dataset_id, database_path=str(database), evaluation_start=str(start), evaluation_end=str(end))
        with duckdb.connect(str(database)) as db:
            db.execute("SET threads=4")
            db.execute("SET memory_limit='6GB'")
            db.execute("CREATE TABLE IF NOT EXISTS stock_universe(symbol VARCHAR PRIMARY KEY)")
            db.executemany("INSERT OR IGNORE INTO stock_universe VALUES (?)",[(s,) for s in symbols])
            db.execute("CREATE TABLE IF NOT EXISTS kline_1m(symbol VARCHAR NOT NULL,timestamp TIMESTAMP NOT NULL,open DOUBLE NOT NULL,high DOUBLE NOT NULL,low DOUBLE NOT NULL,close DOUBLE NOT NULL,volume BIGINT NOT NULL,paused BOOLEAN DEFAULT FALSE,suspicious BOOLEAN DEFAULT FALSE,PRIMARY KEY(symbol,timestamp))")
            db.execute("CREATE TABLE IF NOT EXISTS completed_reads(chunk VARCHAR PRIMARY KEY)")
            cursor = lower
            completed = 0
            excluded_off_session = 0
            while cursor < exclusive_end:
                next_month = date(cursor.year+(cursor.month==12),1 if cursor.month==12 else cursor.month+1,1)
                upper = min(next_month,exclusive_end)
                for offset in range(0,len(symbols),25):
                    chunk=f"{cursor}:{upper}:{offset}"
                    if db.execute("SELECT 1 FROM completed_reads WHERE chunk=?",[chunk]).fetchone():
                        continue
                    group=symbols[offset:offset+25]
                    marks=",".join("?" for _ in group)
                    response=query(f"SELECT symbol,timestamp,open,high,low,close,volume FROM kline_1m WHERE symbol IN ({marks}) AND timestamp>=? AND timestamp<? AND COALESCE(paused,false)=false AND COALESCE(suspicious,false)=false ORDER BY symbol,timestamp", [*group,str(cursor),str(upper)])
                    rows=response["rows"]
                    if rows:
                        index={name:response["columns"].index(name) for name in ("symbol","timestamp","open","high","low","close","volume")}
                        timestamps=np.asarray([source._decode(row[index["timestamp"]]) for row in rows],dtype="datetime64[ns]")
                        continuous=continuous_minute_mask(timestamps)
                        excluded_off_session+=int(np.sum(~continuous))
                        rows=[row for row,keep in zip(rows,continuous,strict=True) if keep]
                        timestamps=timestamps[continuous]
                        shifted=shift_start_to_end(timestamps).astype("datetime64[us]").astype(str).tolist()
                        db.execute("BEGIN")
                        try:
                            db.execute("INSERT OR REPLACE INTO kline_1m SELECT unnest(?::VARCHAR[]),unnest(?::TIMESTAMP[]),unnest(?::DOUBLE[]),unnest(?::DOUBLE[]),unnest(?::DOUBLE[]),unnest(?::DOUBLE[]),unnest(?::BIGINT[]),false,false", [[row[index["symbol"]] for row in rows],shifted,*[[row[index[name]] for row in rows] for name in ("open","high","low","close","volume")]])
                            db.execute("INSERT INTO completed_reads VALUES (?)",[chunk])
                            db.execute("COMMIT")
                        except Exception:
                            db.execute("ROLLBACK")
                            raise
                    else:
                        db.execute("INSERT INTO completed_reads VALUES (?)",[chunk])
                    completed+=1
                    status("reading_local_duckdb", through_month=str(cursor)[:7], chunk_offset=offset, chunks_completed=completed, last_chunk_rows=len(rows), excluded_off_session_rows=excluded_off_session)
                db.execute("CHECKPOINT")
                cursor=upper
            actual=db.execute("SELECT symbol,count(*),CAST(min(timestamp) AS VARCHAR),CAST(max(timestamp) AS VARCHAR) FROM kline_1m GROUP BY symbol ORDER BY symbol").fetchall()
            if set(symbols) != {row[0] for row in actual}:
                raise ValueError("materialized minute coverage is incomplete")
            provenance["materialized_coverage"]=[dict(zip(("symbol","rows","first","last"),row,strict=True)) for row in actual]
            provenance["rows"]=sum(row[1] for row in actual)
            provenance["excluded_off_session_rows_this_attempt"]=excluded_off_session
            save_json(manifest,provenance)
            db.execute("CHECKPOINT")
        status("building_fk_cache", rows=provenance["rows"])
        from bnhmm.boundary_fk.cache import BoundaryDataCache, BoundaryDataCacheKey, BoundaryDataCacheMiss, SECTOR_CACHE_SLOT
        from bnhmm.boundary_fk.precompute import build_sector_boundary_cache
        from bnhmm.boundary_fk.experiment import run_boundary_flow_experiment
        from bnhmm.data.source import DuckDBBarSource
        cache_root=ROOT/"local-state"/"boundary-cache"
        cache=BoundaryDataCache(cache_root)
        key=BoundaryDataCacheKey.from_source(dataset_id=dataset_id,source_path=database,symbols=tuple(symbols),start=str(start),end=str(exclusive_end),lookback_sessions=SECTOR_CACHE_SLOT)
        try:
            cache.load(key,evaluation_start=str(start),evaluation_end=str(exclusive_end))
        except BoundaryDataCacheMiss:
            build_sector_boundary_cache(DuckDBBarSource(str(database)),cache,dataset_id=dataset_id,source_path=database,symbols=tuple(symbols),start=start,end=exclusive_end,progress=lambda phase,details: status(phase,**details))
        output=ROOT/"local-state"/"runs"/f"{dataset_id}-{datetime.now():%Y%m%d-%H%M%S}"
        output.mkdir(parents=True,exist_ok=True)
        save_json(output/"local-data-provenance.json",provenance)
        status("running_fk_strategy", output_dir=str(output))
        run_boundary_flow_experiment(SimpleNamespace(duckdb_path=str(database),dataset_id=dataset_id,symbols=",".join(symbols),start=str(start),end=str(exclusive_end),cache_root=str(cache_root),output_dir=str(output),minimum_train_sessions=120,max_folds=None,fk_paths=10000,seed=7,neutral_sigma=0.35,drift_half_life_sessions=60.0,minimum_direction_probability=0.51,minimum_signal_to_noise=0.10,minimum_survival_probability=0.35,relative_strength_lookback_sessions=20,transaction_cost_bps=5.0,maximum_drawdown=0.05))
        status("completed")
    except Exception as exc:
        status("failed",error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker",default="http://127.0.0.1:9000/api/v1/db")
    parser.add_argument("--members",type=Path,default=Path(r"G:\projects\quant-data\raw\基本数据\同花顺概念板块\板块详情_同花顺一级概念\板块成分股_全部股票\半导体.csv"))
    parser.add_argument("--work-root",type=Path,default=ROOT/"local-state"/"fk-semiconductor-run")
    parser.add_argument("--allow-missing-members",action="store_true",help="Record and exclude members with no local minute coverage")
    run(parser.parse_args())
