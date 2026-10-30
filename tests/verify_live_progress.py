"""Prove persistent FK inputs are reused while measured progress advances."""
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8765"
ROOT = Path(__file__).resolve().parents[1]


def api(route, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    with urlopen(Request(BASE + route, data=body, headers={"Content-Type":"application/json"}), timeout=25) as response:
        return json.load(response)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def main():
    inputs = api("/api/strategy-datasets")[0]
    assert inputs["status"] == "ready" and inputs["run_mode"] == "reuse_only"
    path = Path(inputs["dataset_path"])
    before = digest(path)
    for _ in range(2):
        reused = api("/api/boundary-data/build", {"dataset_id":inputs["dataset_id"],"start":inputs["start"],"end":inputs["end"]})
        assert reused["reused"] is True
    assert digest(path) == before
    market = next(d for d in api("/api/market-datasets") if d["dataset_id"] == inputs["dataset_id"])
    request = {
        "strategy":"boundary-flow-fk", "training_scope":"sector-pooled", "name":"FK-monitor-" + time.strftime("%Y%m%d-%H%M%S"),
        "market_dataset_id":market["dataset_id"], "dataset_id":market["dataset_id"], "symbols":market["symbols"],
        "start":inputs["start"], "end":inputs["end"], "device":"cpu", "minimum_train_sessions":120,
        "boundary_fk_paths":10000, "seed":7,
    }
    job = api("/api/jobs", request)
    snapshots = []
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        job = api("/api/jobs/" + job["id"])
        p = job.get("progress") or {}
        snapshots.append({"status":job["status"], **{key:p.get(key) for key in ("phase","fraction","completed_units","total_units","backtest_fraction","elapsed_seconds","eta_seconds","current_fold","total_folds")}})
        if job["status"] in {"completed","failed","cancelled","interrupted"}:
            break
        time.sleep(.12)
    assert job["status"] == "completed", job
    assert any(s["phase"] == "walk_forward" and (s.get("completed_units") or 0) > 0 and s["fraction"] < 1 for s in snapshots)
    assert any(s["eta_seconds"] is not None and s["eta_seconds"] > 0 for s in snapshots)
    p = job["progress"]
    assert p["fraction"] == 1 and p["eta_seconds"] == 0
    assert p["backtest_completed_units"] == p["backtest_total_units"] == 242
    assert digest(path) == before
    used = json.loads((Path(job["output_dir"]) / "strategy-dataset-used.json").read_text(encoding="utf-8"))
    assert used["run_mode"] == "reuse_only" and used["source_fingerprint"] == inputs["source_fingerprint"]
    try:
        api("/api/jobs", {**request,"name":request["name"]+"-missing","start":"2024-01-01"})
        raise AssertionError("missing strategy inputs were allowed")
    except HTTPError as exc:
        assert exc.code == 409
    report = {"passed":True,"job_id":job["id"],"output_dir":job["output_dir"],"input_file_unchanged":True,"repeat_prepare_reused":True,
              "source_fingerprint":inputs["source_fingerprint"],"progress_snapshots":snapshots,"completed_days":242,"missing_inputs_rejected":True}
    destination = ROOT / "local-state" / "validation" / "live-progress-verification.json"
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))


if __name__ == "__main__":
    main()
