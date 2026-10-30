from __future__ import annotations

from pathlib import Path
import json
import sys
from unittest.mock import patch

import pytest
from bnhmm.boundary_fk.cache import BoundaryDataCache, BoundaryDataCacheKey
from boundary_fk_fixtures import walk_forward_sector_data

from bnhmm.web.boundary_data import BoundaryDataBuildManager, build_boundary_data_command


def test_web_boundary_data_command_builds_one_sector_state_cache(tmp_path: Path) -> None:
    command = build_boundary_data_command(
        python="python",
        database_path="/data/semiconductor.duckdb",
        symbols=("A", "B", "C", "D"),
        dataset_id="semiconductor",
        start="2024-01-01",
        end="2025-12-31",
        cache_root=tmp_path / "cache",
    )

    assert command[:4] == ["python", "-m", "bnhmm.cli", "boundary-data-build"]
    assert "--lookbacks" not in command
    assert command[command.index("--cache-root") + 1] == str(tmp_path / "cache")


def test_service_discovers_saved_inputs_and_reuses_without_spawn(tmp_path):
    source = tmp_path / "market.duckdb"
    source.write_bytes(b"independent data")
    root = tmp_path / "inputs"
    symbols = ("A", "B", "C", "D")
    key = BoundaryDataCacheKey.from_source(dataset_id="semiconductor", source_path=source,
        symbols=symbols, start="2025-01-01", end="2026-01-01", lookback_sessions=60)
    cache = BoundaryDataCache(root)
    artifact = cache.save(key, walk_forward_sector_data(180))
    before = artifact.stat().st_mtime_ns
    dataset = {"dataset_id":"semiconductor", "name":"半导体", "database_path":str(source), "symbols":list(symbols)}
    service = BoundaryDataBuildManager(root, tmp_path, sys.executable)
    assert service.describe(dataset)["status"] == "ready"
    with patch("bnhmm.web.boundary_data.subprocess.run", side_effect=AssertionError("no rebuild")), patch("bnhmm.web.boundary_data.threading.Thread", side_effect=AssertionError("no build thread")):
        for _ in range(2):
            result = service.submit(database_path=str(source), symbols=symbols, dataset_id="semiconductor", start="2025-02-01", end="2025-12-01")
            assert result["reused"] is True
    restarted = BoundaryDataBuildManager(root, tmp_path, sys.executable)
    assert restarted.describe(dataset)["status"] == "ready"
    assert restarted.status("semiconductor")["reused"] is True
    assert artifact.stat().st_mtime_ns == before
    source.write_bytes(b"source changed")
    with pytest.raises(ValueError, match="运行中不会临时构建"):
        restarted.require_ready(dataset, start="2025-02-01", end="2025-12-01")


def test_restarted_build_is_interrupted_instead_of_running_forever(tmp_path):
    root = tmp_path / "inputs"
    path = root / "semiconductor" / "build-status.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"dataset_id":"semiconductor", "status":"running"}), encoding="utf-8")
    service = BoundaryDataBuildManager(root, tmp_path, sys.executable)
    dataset = {"dataset_id":"semiconductor", "name":"半导体", "database_path":str(tmp_path / "missing.duckdb"), "symbols":["A","B","C","D"]}
    assert service.describe(dataset)["status"] == "interrupted"
