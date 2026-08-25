from __future__ import annotations

from pathlib import Path

from bnhmm.web.boundary_data import build_boundary_data_command


def test_web_boundary_data_command_builds_all_windows(tmp_path: Path) -> None:
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
    assert command[command.index("--lookbacks") + 1] == "10,20,30,40,60,90"
    assert command[command.index("--cache-root") + 1] == str(tmp_path / "cache")
