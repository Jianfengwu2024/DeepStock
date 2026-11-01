"""Local worker for authenticated Quant Platform sector-FK jobs."""

from __future__ import annotations

import json
import math
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from ..boundary_fk.cache import BoundaryDataCacheKey, BoundaryDataCacheMiss, SECTOR_CACHE_SLOT
from .jobs import WalkForwardJobRequest


SECTORS = {
    "semiconductor": {"name": "半导体", "dataset_name": "FK-Bayesian-半导体"},
    "artificial_intelligence": {"name": "人工智能", "dataset_name": "FK-Bayesian-人工智能"},
    "new_energy": {"name": "新能源", "dataset_name": "FK-Bayesian-新能源"},
    "custom": {"name": "所选股票池", "dataset_name": ""},
}
_WARMUP_DAYS = 450
_POLL_SECONDS = 8
_PROGRESS_FIELDS = {"phase", "message", "fraction", "completed_units", "total_units", "unit",
    "backtest_fraction", "backtest_completed_units", "backtest_total_units", "backtest_unit",
    "elapsed_seconds", "eta_seconds", "updated_at", "current_fold", "total_folds", "activity", "signal_date"}


def public_progress(value: dict[str, Any] | None) -> dict[str, Any] | None:
    return {key: item for key, item in value.items() if key in _PROGRESS_FIELDS} if isinstance(value, dict) else None


def daily_result_series(predictions: list[dict[str, Any]], cost_bps: float) -> list[dict[str, Any]]:
    """Reconstruct continuous NAV from saved returns; never use reset stop NAV."""
    net = gross = benchmark = peak = 1.0
    rows = []
    for item in sorted(predictions, key=lambda row: row["signal_time"]):
        daily_gross, turnover = item.get("gross_strategy_return"), item.get("turnover")
        if any(not isinstance(value, (float, int)) or not math.isfinite(value) for value in (daily_gross, turnover, cost_bps)):
            return []
        daily_net = daily_gross - turnover * cost_bps / 10_000
        net *= 1 + daily_net
        gross *= 1 + daily_gross
        peak = max(peak, net)
        sector = item.get("sector") or {}
        actual = sector.get("actual_return")
        benchmark = benchmark * (1 + actual) if benchmark is not None and isinstance(actual, (int,float)) and math.isfinite(actual) else None
        weights = item.get("actual_weights")
        rows.append({"date": str(item.get("target_time") or item["signal_time"])[:10],
            "net_equity": net, "gross_equity": gross, "benchmark_equity": benchmark, "drawdown": net/peak-1,
            "net_return": daily_net, "exposure": sum(weights) if isinstance(weights,list) else None,
            "target_exposure": item.get("target_exposure"), "state_probabilities": sector.get("state_probabilities")})
    return rows


class CloudStrategyWorker:
    def __init__(self, *, server: str, jobs: Any, boundary_data: Any, market_datasets: Any, stock_pools: Any = None) -> None:
        parsed = urlsplit(server.rstrip("/"))
        if parsed.scheme != "https" or not parsed.hostname or parsed.path not in {"", "/"}:
            raise ValueError("Quant Platform worker must use an HTTPS origin")
        self.server = server.rstrip("/")
        self.jobs = jobs
        self.boundary_data = boundary_data
        self.market_datasets = market_datasets
        self.stock_pools = stock_pools
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._token = ""
        self._user: dict[str, Any] | None = None
        self._current_job: str | None = None
        self._last_error: str | None = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "connected": bool(self._thread and self._thread.is_alive() and self._token),
                "user": self._user,
                "current_job_id": self._current_job,
                "last_error": self._last_error,
            }

    def disconnect(self) -> dict[str, Any]:
        with self._lock:
            self._stop.set()
            thread = self._thread
            self._token = ""
            self._user = None
            self._current_job = None
        if thread and thread is not threading.current_thread():
            thread.join(timeout=2)
        return self.status()

    def connect(self, token: str) -> dict[str, Any]:
        if not token or len(token) > 4096:
            raise ValueError("Quant Platform access token is invalid")
        session = self._request("GET", "/api/v1/auth/me", token=token)
        user = session.get("user") if isinstance(session, dict) else None
        if not isinstance(user, dict):
            raise ValueError("Quant Platform session could not be verified")
        self._request("GET", "/api/v1/deepstock/catalog", token=token)
        self.disconnect()
        with self._lock:
            self._stop = threading.Event()
            stop_event = self._stop
            self._token = token
            self._user = {key: user.get(key) for key in ("id", "username", "display_name", "role") if user.get(key) is not None}
            self._last_error = None
            self._thread = threading.Thread(target=self._run, args=(stop_event, token), name="deepstock-cloud-fk-worker", daemon=True)
            self._thread.start()
        return self.status()

    def _request(
        self,
        method: str,
        path: str,
        *,
        token: str,
        payload: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = Request(
            self.server + path,
            data=body,
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read(1_600_001)
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise PermissionError("Quant Platform session expired or lacks DeepStock access") from None
            raise RuntimeError(f"Quant Platform worker API returned HTTP {exc.code}") from None
        except (URLError, TimeoutError) as exc:
            raise RuntimeError("Quant Platform worker API is unavailable") from exc
        if len(raw) > 1_600_000:
            raise RuntimeError("Quant Platform worker response exceeded its size limit")
        try:
            value = json.loads(raw) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Quant Platform worker response was invalid JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError("Quant Platform worker response was not an object")
        return value

    def _run(self, stop_event: threading.Event, token: str) -> None:
        while not stop_event.is_set():
            try:
                task = self._request("POST", "/api/v1/deepstock/fk/worker/next", token=token, payload={"worker_id": "deepstock-local"}, timeout=20)
                pool_sync = task.get("pool_sync")
                if isinstance(pool_sync, dict):
                    self._sync_pool(pool_sync, token)
                    continue
                job = task.get("job")
                if not isinstance(job, dict):
                    stop_event.wait(_POLL_SECONDS)
                    continue
                identifier = str(job.get("job_id") or "")
                with self._lock:
                    self._current_job = identifier
                    self._last_error = None
                try:
                    result = self._execute(job, token, stop_event)
                    self._request(
                        "POST",
                        f"/api/v1/deepstock/fk/worker/jobs/{quote(identifier, safe='')}/result",
                        token=token,
                        payload={"result": result},
                        timeout=45,
                    )
                except Exception as exc:
                    code = str(exc) if str(exc) in {"unsupported_cloud_job", "invalid_evaluation_window",
                        "local_sector_dataset_missing", "local_minute_dataset_not_ready",
                        "local_sector_minute_coverage_incomplete", "local_dataset_does_not_cover_fk_warmup",
                        "local_dataset_does_not_cover_evaluation_end", "local_fk_strategy_dataset_not_ready",
                        "invalid_cloud_stock_pool", "local_cloud_stock_pool_missing", "local_cloud_dataset_create_failed",
                        "local_cloud_dataset_missing", "local_cloud_dataset_failed", "local_fk_strategy_dataset_failed",
                        "local_fk_job_missing", "local_fk_job_failed", "cloud_job_cancelled",
                        "local_worker_disconnected"} else type(exc).__name__
                    with self._lock:
                        self._last_error = code
                    self._send_failure(identifier, token, code)
                finally:
                    with self._lock:
                        self._current_job = None
            except PermissionError:
                self.disconnect()
                return
            except Exception as exc:
                with self._lock:
                    self._last_error = type(exc).__name__
                stop_event.wait(_POLL_SECONDS)

    def _sync_pool(self, pool: dict[str, Any], token: str) -> None:
        pool_id = str(pool.get("pool_id") or "")
        sync_token = str(pool.get("sync_token") or "")
        outcome: dict[str, Any]
        try:
            if not pool_id or len(sync_token) != 32:
                raise ValueError("invalid_cloud_pool_sync")
            if self.stock_pools is None:
                raise RuntimeError("local_stock_pool_sync_failed")
            local = self.stock_pools.sync_cloud_pool(pool)
            outcome = {"sync_token": sync_token, "local_pool_id": local["id"]}
        except Exception:
            outcome = {"sync_token": sync_token, "error": "local_stock_pool_sync_failed"}
        self._request(
            "POST",
            f"/api/v1/deepstock/fk/worker/pools/{quote(pool_id, safe='')}/result",
            token=token,
            payload=outcome,
            timeout=30,
        )

    def _send_failure(self, identifier: str, token: str, code: str) -> None:
        try:
            self._request(
                "POST",
                f"/api/v1/deepstock/fk/worker/jobs/{quote(identifier, safe='')}/result",
                token=token,
                payload={"error": code},
                timeout=20,
            )
        except Exception:
            pass

    def _heartbeat(self, identifier: str, token: str, phase: str, progress: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/v1/deepstock/fk/worker/jobs/{quote(identifier, safe='')}/heartbeat",
            token=token,
            payload={"phase": phase, **({"progress": public_progress(progress)} if progress else {})},
            timeout=20,
        )

    def _execute(self, job: dict[str, Any], token: str, stop_event: threading.Event) -> dict[str, Any]:
        identifier = str(job.get("job_id") or "")
        sector_id = str(job.get("sector") or "")
        sector = SECTORS.get(sector_id)
        if not identifier or sector is None:
            raise ValueError("unsupported_cloud_job")
        start = date.fromisoformat(str(job["start"]))
        end = date.fromisoformat(str(job["end"]))
        if end < start or (end - start).days > 370:
            raise ValueError("invalid_evaluation_window")
        warmup_start = start - timedelta(days=_WARMUP_DAYS)
        pool_id = str(job.get("pool_id") or "")
        pool_name = str(job.get("pool_name") or "")
        if pool_id and pool_name:
            sector = {**sector, "name": pool_name}
        pool_symbols = job.get("pool_symbols")
        if pool_id:
            if not isinstance(pool_symbols, list) or not 4 <= len(pool_symbols) <= 2000:
                raise ValueError("invalid_cloud_stock_pool")
            symbols = tuple(dict.fromkeys(str(item) for item in pool_symbols))
            if len(symbols) != len(pool_symbols) or any(len(item) != 6 or not item.isdigit() for item in symbols):
                raise ValueError("invalid_cloud_stock_pool")
            if self.stock_pools is None:
                raise RuntimeError("local_cloud_stock_pool_missing")
            synced_pool = next((item for item in self.stock_pools.list() if item.get("cloud_pool_id") == pool_id), None)
            if synced_pool is None or set(synced_pool["symbols"]) != set(symbols):
                raise ValueError("local_cloud_stock_pool_missing")
            dataset = self._build_cloud_dataset(identifier, pool_id, pool_name, symbols, start, end, token, stop_event)
        else:
            dataset = next(
                (item for item in self.market_datasets.store.list() if item["name"] == sector["dataset_name"] or item["name"].startswith("FK-贝叶斯-" + sector["name"])),
                None,
            )
            if dataset is None:
                raise ValueError("local_sector_dataset_missing")
            if dataset["status"] not in {"ready", "partial"} or "1m" not in dataset["intervals"]:
                raise ValueError("local_minute_dataset_not_ready")
            symbols = tuple(dataset["symbols"])
        if len(symbols) < 4 or any(not dataset["coverage"].get("1m", {}).get(symbol, {}).get("rows") for symbol in symbols):
            raise ValueError("local_sector_minute_coverage_incomplete")
        if date.fromisoformat(dataset["start_date"]) > warmup_start:
            raise ValueError("local_dataset_does_not_cover_fk_warmup")
        latest = min(date.fromisoformat(str(dataset["coverage"]["1m"][symbol]["end"])[:10]) for symbol in symbols)
        if latest < start:
            raise ValueError("local_dataset_does_not_cover_evaluation_start")
        if latest < end:
            requested_end = end
            end = latest
            self._heartbeat(identifier, token, "adjusting_evaluation_window", {
                "phase": "adjusting_evaluation_window",
                "message": f"所选股票的共同分钟行情截至 {latest.isoformat()}；样本外结束日已从 {requested_end.isoformat()} 调整到该日",
                "activity": "按所有股票共同覆盖的真实行情范围运行",
            })

        self._heartbeat(identifier, token, "validating saved FK strategy dataset")
        cache_key = BoundaryDataCacheKey.from_source(
            dataset_id=dataset["dataset_id"],
            source_path=dataset["database_path"],
            symbols=symbols,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            lookback_sessions=SECTOR_CACHE_SLOT,
        )
        if pool_id:
            self._build_cloud_fk_cache(identifier, dataset, symbols, start, end, token, stop_event)
        else:
            try:
                self.boundary_data.cache.inspect(cache_key)
            except BoundaryDataCacheMiss:
                raise RuntimeError("local_fk_strategy_dataset_not_ready")

        request = WalkForwardJobRequest(
            strategy="boundary-flow-fk",
            name=f"Cloud-{sector_id}-{identifier[-8:]}",
            database_path=dataset["database_path"],
            market_dataset_id=dataset["dataset_id"],
            dataset_id=dataset["dataset_id"],
            symbols=list(symbols),
            training_scope="sector-pooled",
            start=start,
            end=end + timedelta(days=1),
            device="cpu",
            minimum_train_sessions=120,
            boundary_fk_paths=10_000,
        )
        local_job = self.jobs.submit(request)
        while not stop_event.is_set():
            current = self.jobs.get(local_job.id)
            if current is None:
                raise RuntimeError("local_fk_job_missing")
            if current.status == "completed":
                return self._read_result(local_job.id, dataset, start, end, sector)
            if current.status in {"failed", "interrupted", "cancelled"}:
                raise RuntimeError("local_fk_job_failed")
            progress = self.jobs.progress(local_job.id)
            remote = self._heartbeat(identifier, token, (progress or {}).get("phase") or current.status, progress)
            if remote.get("cancelled"):
                self.jobs.cancel(local_job.id)
                raise RuntimeError("cloud_job_cancelled")
            stop_event.wait(3)
        self.jobs.cancel(local_job.id)
        raise RuntimeError("local_worker_disconnected")

    def _build_cloud_dataset(self, identifier, pool_id, pool_name, symbols, start, end, token, stop_event):
        warmup = start - timedelta(days=_WARMUP_DAYS)
        dataset_name = f"云端 FK · {pool_name[:35]} · {identifier[-8:]}"
        self._heartbeat(identifier, token, "building_local_strategy_dataset", {
            "phase": "building_local_strategy_dataset", "message": "正在本机生成股票池专属 1 分钟策略数据集",
            "unit": "数据准备阶段", "fraction": 0,
        })
        existing = next((item for item in self.market_datasets.store.list() if item["name"] == dataset_name), None)
        if existing is not None:
            same_request = set(existing["symbols"]) == set(symbols) and existing["start_date"] == warmup.isoformat()
            if not same_request or existing.get("purpose") != "cloud-strategy":
                raise RuntimeError("local_cloud_dataset_create_failed")
            if existing.get("retention_expires_at") and datetime.fromisoformat(existing["retention_expires_at"]) <= datetime.now().astimezone():
                self.market_datasets.delete(existing["dataset_id"])
                existing = None
            elif existing["status"] == "failed":
                self.market_datasets.delete(existing["dataset_id"])
                existing = None
        if existing is None:
            try:
                dataset = self.market_datasets.create(
                    name=dataset_name,
                    symbols=list(symbols),
                    intervals=["1m"],
                    start_date=warmup,
                    purpose="cloud-strategy",
                    end_date=end,
                )
            except (RuntimeError, ValueError, OSError) as exc:
                raise RuntimeError("local_cloud_dataset_create_failed") from exc
        else:
            dataset = existing
        dataset_id = dataset["dataset_id"]
        del pool_id, end
        while not stop_event.is_set():
            current = self.market_datasets.store.get(dataset_id)
            if current is None:
                raise RuntimeError("local_cloud_dataset_missing")
            if current["status"] in {"ready", "partial"}:
                return current
            if current["status"] == "failed":
                raise RuntimeError("local_cloud_dataset_failed")
            local_progress = current.get("progress") or {}
            progress = {
                "phase": "building_local_strategy_dataset", "message": "本机正在从本地行情服务物化所选股票池",
                "unit": local_progress.get("unit") or "行情批次",
                "completed_units": local_progress.get("completed_units"),
                "total_units": local_progress.get("total_units"),
                "fraction": local_progress.get("fraction"),
                "elapsed_seconds": local_progress.get("elapsed_seconds"),
                "eta_seconds": local_progress.get("eta_seconds"),
                "activity": local_progress.get("message"),
            }
            response = self._heartbeat(identifier, token, "building_local_strategy_dataset", progress)
            if response.get("cancelled"):
                raise RuntimeError("cloud_job_cancelled")
            stop_event.wait(3)
        raise RuntimeError("local_worker_disconnected")

    def _build_cloud_fk_cache(self, identifier, dataset, symbols, start, end, token, stop_event):
        self._heartbeat(identifier, token, "building_fk_strategy_dataset", {
            "phase": "building_fk_strategy_dataset", "message": "分钟行情已就绪，正在构建 FK 策略特征数据集",
            "unit": "FK 特征构建阶段", "fraction": 0,
        })
        try:
            self.boundary_data.submit(
                database_path=dataset["database_path"],
                symbols=symbols,
                dataset_id=dataset["dataset_id"],
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
            )
        except (RuntimeError, ValueError, OSError) as exc:
            raise RuntimeError("local_fk_strategy_dataset_failed") from exc
        while not stop_event.is_set():
            current = self.boundary_data.status(dataset["dataset_id"])
            if current["status"] in {"ready", "completed"}:
                self.boundary_data.cache.inspect(BoundaryDataCacheKey.from_source(
                    dataset_id=dataset["dataset_id"], source_path=dataset["database_path"],
                    symbols=symbols, start=start.isoformat(), end=(end+timedelta(days=1)).isoformat(),
                    lookback_sessions=SECTOR_CACHE_SLOT,
                ))
                return
            if current["status"] in {"failed", "interrupted", "stale"}:
                raise RuntimeError("local_fk_strategy_dataset_failed")
            progress = current.get("progress") or {}
            response = self._heartbeat(identifier, token, "building_fk_strategy_dataset", {
                "phase": "building_fk_strategy_dataset", "message": "正在构建并校验本次股票池的 FK 特征缓存",
                "unit": progress.get("unit") or "FK 特征缓存",
                "completed_units": progress.get("completed_units"),
                "total_units": progress.get("total_units"),
                "fraction": progress.get("fraction"),
                "elapsed_seconds": progress.get("elapsed_seconds"),
                "eta_seconds": progress.get("eta_seconds"),
                "activity": progress.get("message"),
            })
            if response.get("cancelled"):
                raise RuntimeError("cloud_job_cancelled")
            stop_event.wait(3)
        raise RuntimeError("local_worker_disconnected")

    def _read_result(
        self,
        job_id: str,
        dataset: dict[str, Any],
        start: date,
        end: date,
        sector: dict[str, str],
    ) -> dict[str, Any]:
        record = self.jobs.get(job_id)
        root = Path(record.output_dir)
        report = json.loads((root / "sector-direction-evaluation.json").read_text(encoding="utf-8"))
        policy = json.loads((root / "boundary-policy-evaluation.json").read_text(encoding="utf-8"))
        predictions = json.loads((root / "feynman-kac-predictions.json").read_text(encoding="utf-8"))
        model = next((item for item in report.get("models", []) if item.get("model_name") == report.get("preferred_model")), None)
        stage = next((item for item in policy if item.get("model_name") == report.get("preferred_model")), None)
        if not isinstance(model, dict) or not isinstance(stage, dict):
            raise RuntimeError("local_fk_result_missing")
        latest = []
        for item in predictions:
            sector_result = item.get("sector") if isinstance(item, dict) else None
            if not isinstance(sector_result, dict):
                continue
            latest.append({
                "signal_time": item.get("signal_time"),
                "available_at": item.get("available_at"),
                "target_time": item.get("target_time"),
                "training_cutoff": item.get("training_cutoff"),
                "model_version": item.get("model_version"),
                "direction": sector_result.get("direction"),
                "state_probabilities": sector_result.get("state_probabilities"),
                "expected_drift": sector_result.get("expected_drift"),
                "survival_confidence": sector_result.get("survival_confidence"),
                "bullish_drift_probability": sector_result.get("bullish_drift_probability"),
                "selected_symbols": item.get("selected_symbols", [])[:40],
                "target_exposure": item.get("target_exposure"),
                "sector_equity": item.get("sector_equity"),
            })
        return {
            "sector_name": sector["name"],
            "model_version": report.get("preferred_model"),
            "evaluation_start": start.isoformat(),
            "evaluation_end": end.isoformat(),
            "data_start": dataset["start_date"],
            "data_end": dataset["last_refresh_date"],
            "symbol_count": len(dataset["symbols"]),
            "observations": report.get("observations"),
            "baselines": report.get("baselines"),
            "metrics": model.get("metrics"),
            "monthly_direction_accuracy": model.get("monthly_direction_accuracy", []),
            "promotion_eligible": stage.get("promotion_eligible"),
            "failure_reason": stage.get("failure_reason"),
            "latest_predictions": latest[-20:],
            "prediction_history": latest,
            "daily_series": daily_result_series(predictions, record.request.transaction_cost_bps),
            "training_history": self._training_history(root),
            "run_parameters": {"transaction_cost_bps": record.request.transaction_cost_bps,
                "minimum_train_sessions": record.request.minimum_train_sessions,
                "paths": record.request.boundary_fk_paths, "seed": record.request.seed,
                "dataset_id": dataset["dataset_id"],
                "source_fingerprint": self._source_fingerprint(root)},
            "source_run_id": job_id,
            "progress": public_progress(self.jobs.progress(job_id)),
        }

    @staticmethod
    def _source_fingerprint(root: Path) -> str | None:
        path = root / "strategy-dataset-used.json"
        return json.loads(path.read_text(encoding="utf-8")).get("source_fingerprint") if path.is_file() else None

    @staticmethod
    def _training_history(root: Path) -> list[dict[str, Any]]:
        path = root / "boundary-model-posterior.json"
        if not path.is_file():
            return []
        rows = []
        for item in json.loads(path.read_text(encoding="utf-8")):
            rows.append({"training_cutoff": item["training_cutoff"],
                "fit_rows": item.get("hurdle_fit_rows", 0),
                "direction_fit_rows": item.get("hurdle_direction_fit_rows", item.get("direction_fit_rows", 0))})
        return rows

    def publish_saved_run(self, job_id: str, sector_id: str, *, token: str | None = None) -> dict[str, Any]:
        sector = SECTORS.get(sector_id)
        record = self.jobs.get(job_id)
        if sector is None or record is None or record.status != "completed" or record.request.strategy != "boundary-flow-fk":
            raise ValueError("请选择已完成的 FK-贝叶斯任务和对应板块")
        dataset = self.market_datasets.store.get(record.request.market_dataset_id or record.request.dataset_id)
        if not dataset or not (dataset["name"] == sector["dataset_name"] or dataset["name"].startswith("FK-贝叶斯-" + sector["name"])):
            raise ValueError("训练结果与所选板块不匹配")
        with self._lock:
            access_token = token or self._token
        if not access_token:
            raise PermissionError("请先在用户验证页面登录 Quant Platform")
        result = self._read_result(job_id, dataset, record.request.start, record.request.end - timedelta(days=1), sector)
        return self._request("POST", "/api/v1/deepstock/fk/results", token=access_token,
            payload={"sector": sector_id, "result": result}, timeout=45)
