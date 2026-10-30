import json
from bnhmm.progress import ProgressReporter


def test_progress_counts_and_eta_follow_measured_phase_work(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("bnhmm.progress.time.monotonic", lambda: clock[0])
    reporter = ProgressReporter(tmp_path)
    reporter.update("walk_forward", completed_units=0, total_units=100, unit="交易日")
    before = json.loads(reporter.path.read_text(encoding="utf-8"))
    assert before["eta_seconds"] is None
    clock[0] = 20
    reporter.update(completed_units=25)
    measured = json.loads(reporter.path.read_text(encoding="utf-8"))
    assert measured["fraction"] == .25
    assert measured["backtest_completed_units"] == 25
    assert measured["eta_seconds"] == 60
    assert measured["elapsed_seconds"] == 20
    reporter.update("evaluation", message="正在汇总")
    evaluation = json.loads(reporter.path.read_text(encoding="utf-8"))
    assert evaluation["backtest_completed_units"] == 25
    assert evaluation["fraction"] is None
    assert evaluation["eta_seconds"] is None
    clock[0] = 24
    reporter.update("completed")
    complete = json.loads(reporter.path.read_text(encoding="utf-8"))
    assert complete["fraction"] == 1
    assert complete["eta_seconds"] == 0


def test_progress_survives_transient_file_replace_failure(tmp_path, monkeypatch):
    reporter = ProgressReporter(tmp_path)
    monkeypatch.setattr("bnhmm.progress.os.replace", lambda *_: (_ for _ in ()).throw(PermissionError()))
    reporter.update("starting", message="继续计算")
