from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_start_script_allows_slow_history_indexing() -> None:
    script = (ROOT / "start.sh").read_text(encoding="utf-8")

    assert 'timeout_seconds="${BNHMM_START_TIMEOUT_SECONDS:-90}"' in script
    assert 'seq 1 "${timeout_seconds}"' in script
    assert "within ${timeout_seconds} seconds" in script


def test_quant_service_scripts_manage_market_sync_daemon() -> None:
    start = Path(r"G:\projects\start-data-services.bat").read_text(encoding="utf-8")
    stop = Path(r"G:\projects\stop-data-services.bat").read_text(encoding="utf-8")
    launcher = Path(r"G:\projects\market-sync-daemon.bat").read_text(encoding="utf-8")
    api_launcher = Path(r"G:\projects\fapi.bat").read_text(encoding="utf-8")

    assert "market-sync-daemon.bat" in start
    assert "app.services.market_sync_daemon" in stop
    assert "QUANT_SECURE_ENV_FILE=%SECURITY_DIR%\\server.env" in launcher
    assert "QUANT_ADMIN_MARKET_SYNC_ENABLED=1" in launcher
    assert "QUANT_DUCKDB_READ_ONLY=1" in launcher
    assert "app.services.market_sync_daemon" in launcher
    assert "QUANT_MARKET_SYNC_DAEMON_ENABLED=1" in api_launcher
