from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_start_script_allows_slow_history_indexing() -> None:
    script = (ROOT / "start.sh").read_text(encoding="utf-8")

    assert 'timeout_seconds="${BNHMM_START_TIMEOUT_SECONDS:-90}"' in script
    assert 'seq 1 "${timeout_seconds}"' in script
    assert "within ${timeout_seconds} seconds" in script
