import threading
import time
from datetime import date
from pathlib import Path

import duckdb

from bnhmm.data.local_dataset import LocalMarketDatasetManager
from bnhmm.data.source import BrokerBarSource


def test_bulk_materialization_uses_one_export_preserves_bound_and_resumes_atomically(tmp_path, monkeypatch):
    import bnhmm.data.local_dataset as module
    requests = []
    symbols = [f"{i:06d}" for i in range(26)]
    class Source(BrokerBarSource):
        def local_export_request(self, method, identifier=None, payload=None):
            requests.append((method,payload))
            if method == "POST":
                artifact = tmp_path/"abc.parquet"
                with duckdb.connect() as db:
                    db.execute("CREATE TABLE bars(symbol VARCHAR,timestamp TIMESTAMP,open DOUBLE,high DOUBLE,low DOUBLE,close DOUBLE,volume BIGINT)")
                    db.executemany("INSERT INTO bars VALUES (?, '2026-01-02 09:31:00',1,2,1,2,100)",[(symbol,) for symbol in symbols])
                    db.execute("COPY bars TO ? (FORMAT PARQUET)", [str(artifact)])
                return {"export_id":"abc","status":"completed","path":str(artifact),"row_count":26}
            if method == "DELETE":
                (tmp_path/"abc.parquet").unlink()
                return {"deleted":True}
            raise AssertionError(method)
    monkeypatch.setattr(module,"create_bar_source",lambda *args,**kwargs:Source("http://127.0.0.1:9000/api/v1/db"))
    manager=LocalMarketDatasetManager(tmp_path/"datasets","http://127.0.0.1:9000/api/v1/db")
    try:
        dataset=manager.create(name="bulk",symbols=symbols,intervals=["1m"],start_date=date(2026,1,1),end_date=date(2026,1,3),purpose="cloud-strategy")
        manager._queue.join()
        saved=manager.store.get(dataset["dataset_id"])
        assert saved["status"]=="ready", saved["error"]
        assert saved["row_counts"]=={"1m":26}
        assert len([r for r in requests if r[0]=="POST"])==1
        assert requests[0][1]["end"]=="2026-01-04"
        assert len(saved["coverage"]["1m"])==26
        manager.refresh(saved["dataset_id"])
        manager._queue.join()
        saved=manager.store.get(dataset["dataset_id"])
        assert saved["status"]=="ready"
        assert saved["row_counts"]=={"1m":26}
        assert not (tmp_path/"abc.parquet").exists()
    finally:
        manager.close()
