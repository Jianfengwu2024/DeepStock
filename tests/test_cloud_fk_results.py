from datetime import date, timedelta
from types import SimpleNamespace
from threading import Event
import pytest
from bnhmm.web.cloud_strategy import CloudStrategyWorker, daily_result_series, public_progress


def test_published_nav_uses_real_returns_and_costs_not_internal_stop_nav():
    rows = daily_result_series([
        dict(signal_time='2026-10-01',target_time='2026-10-02',gross_strategy_return=.1,turnover=1,sector_equity=999,actual_weights=[.2,.3],sector={'actual_return':.2}),
        dict(signal_time='2026-10-02',target_time='2026-10-03',gross_strategy_return=-.05,turnover=0,sector_equity=1,actual_weights=[],sector={'actual_return':-.1}),
    ],10)
    assert rows[0]['net_equity'] == pytest.approx(1.099)
    assert rows[1]['net_equity'] == pytest.approx(1.099*.95)
    assert rows[1]['drawdown'] == pytest.approx(-.05)
    assert rows[1]['benchmark_equity'] == pytest.approx(1.08)
    assert rows[0]['exposure'] == .5
    assert daily_result_series([{'signal_time':'2026-10-01'}],5) == []


def test_public_progress_does_not_export_paths_pids_or_unrelated_state():
    assert public_progress({'phase':'walk_forward','completed_units':25,'pid':123,'path':'secret'}) == {'phase':'walk_forward','completed_units':25}


def test_saved_publication_requires_a_completed_matching_local_run():
    record=SimpleNamespace(status='completed',request=SimpleNamespace(strategy='boundary-flow-fk',market_dataset_id='semi',dataset_id='semi',start=date(2026,1,1),end=date(2026,1,4)))
    worker=CloudStrategyWorker(server='https://quant.example',jobs=SimpleNamespace(get=lambda _:record),boundary_data=None,market_datasets=SimpleNamespace(store=SimpleNamespace(get=lambda _: {'name':'FK-贝叶斯-半导体（本地 DuckDB）'})))
    calls=[]
    worker._read_result=lambda *args: {'source_run_id':args[0],'evaluation_end':args[3].isoformat()}
    worker._request=lambda *args,**kwargs: calls.append((args,kwargs)) or {'job_id':'remote','status':'completed'}
    with pytest.raises(PermissionError): worker.publish_saved_run('local','semiconductor')
    worker.publish_saved_run('local','semiconductor',token='private-session')
    assert calls[0][0] == ('POST','/api/v1/deepstock/fk/results')
    assert calls[0][1]['payload']['result']['evaluation_end'] == '2026-01-03'
    with pytest.raises(ValueError): worker.publish_saved_run('local','new_energy',token='private-session')
    record.status='running'
    with pytest.raises(ValueError): worker.publish_saved_run('local','semiconductor',token='private-session')


def test_cloud_worker_syncs_pool_only_after_its_local_store_accepts_it():
    class PoolStore:
        def __init__(self):
            self.payload = None
        def sync_cloud_pool(self, payload):
            self.payload = payload
            return {"id": "local-pool-1"}

    store = PoolStore()
    worker = CloudStrategyWorker(
        server="https://quant.example",
        jobs=SimpleNamespace(),
        boundary_data=None,
        market_datasets=None,
        stock_pools=store,
    )
    calls = []
    worker._request = lambda *args, **kwargs: calls.append((args, kwargs)) or {"sync_status": "synced"}
    payload = {"pool_id": "ai-pool-1", "sync_token": "a" * 32, "name": "AI观察池", "symbols": ["000001", "000002", "000003", "000004"]}

    worker._sync_pool(payload, "quant-session")

    assert store.payload is payload
    assert calls[0][0] == ("POST", "/api/v1/deepstock/fk/worker/pools/ai-pool-1/result")
    assert calls[0][1]["payload"] == {"sync_token": "a" * 32, "local_pool_id": "local-pool-1"}


def test_cloud_worker_builds_a_week_retained_minute_dataset_for_the_synced_pool():
    start, end = date(2026, 10, 1), date(2026, 10, 23)
    dataset = {
        "dataset_id": "cloud-dataset-1", "status": "ready", "symbols": ["000001", "000002", "000003", "000004"],
        "start_date": "2025-06-08", "intervals": ["1m"], "coverage": {}, "database_path": "local-only.duckdb",
    }
    calls = []
    class Store:
        def list(self): return []
        def get(self, dataset_id): return dataset if dataset_id == dataset["dataset_id"] else None
    class Manager:
        store = Store()
        def create(self, **kwargs): calls.append(kwargs); return dataset
    worker = CloudStrategyWorker(
        server="https://quant.example", jobs=SimpleNamespace(), boundary_data=None,
        market_datasets=Manager(), stock_pools=None,
    )
    worker._heartbeat = lambda *args, **kwargs: {"cancelled": False}

    result = worker._build_cloud_dataset("fk-job-1", "ai-pool-1", "半导体观察池", tuple(dataset["symbols"]), start, end, "token", Event())

    assert result is dataset
    assert calls[0]["symbols"] == dataset["symbols"]
    assert calls[0]["intervals"] == ["1m"]
    assert calls[0]["start_date"] == date(2026, 10, 1) - timedelta(days=450)
    assert calls[0]["purpose"] == "cloud-strategy"


def test_cloud_worker_reuses_existing_dataset_when_a_job_retries():
    from datetime import datetime, timezone
    start = date(2026, 10, 1)
    dataset = {
        "dataset_id": "cloud-dataset-1", "status": "ready", "symbols": ["000001", "000002", "000003", "000004"],
        "start_date": (start - timedelta(days=450)).isoformat(), "purpose": "cloud-strategy",
    }
    class Store:
        def list(self): return [{**dataset, "name": "云端 FK · 观察池 · 12345678"}]
        def get(self, dataset_id): return dataset if dataset_id == dataset["dataset_id"] else None
    class Manager:
        store = Store()
        def create(self, **kwargs): raise AssertionError("retry must reuse the existing temporary dataset")
    worker = CloudStrategyWorker(server="https://quant.example", jobs=SimpleNamespace(), boundary_data=None, market_datasets=Manager())
    worker._heartbeat = lambda *args, **kwargs: {"cancelled":False}

    reused = worker._build_cloud_dataset("fk-job-12345678", "pool-1", "观察池", tuple(dataset["symbols"]), start, date(2026,10,20), "token", Event())

    assert reused["dataset_id"] == dataset["dataset_id"]


def test_cloud_worker_builds_fk_cache_before_running_and_uses_inclusive_end(tmp_path):
    calls = []
    class Boundary:
        class Cache:
            def inspect(self, key): calls.append({"cache_key":key})
        cache = Cache()
        def submit(self, **kwargs): calls.append(kwargs); return {"status": "running"}
        def status(self, dataset_id): return {"status": "ready"}
    worker = CloudStrategyWorker(
        server="https://quant.example", jobs=SimpleNamespace(), boundary_data=Boundary(),
        market_datasets=None,
    )
    worker._heartbeat = lambda *args, **kwargs: {"cancelled": False}
    source_path = tmp_path / "local-only.duckdb"
    source_path.write_bytes(b"local-only")
    dataset = {"dataset_id": "cloud-dataset-1", "database_path": str(source_path)}
    start, end = date(2026, 10, 1), date(2026, 10, 23)

    worker._build_cloud_fk_cache("fk-job-1", dataset, ("000001", "000002", "000003", "000004"), start, end, "token", Event())

    assert calls[0]["dataset_id"] == dataset["dataset_id"]
    assert calls[0]["start"] == "2026-10-01"
    assert calls[0]["end"] == "2026-10-24"
