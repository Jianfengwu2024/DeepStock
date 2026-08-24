# Sector Regime BNHMM Implementation Plan

> **Execution constraint:** Implement inline in this task with `superpowers:executing-plans`; the user explicitly prohibited subagents and multi-agent development.

**Goal:** Build a leakage-safe two-layer strategy that first predicts the next-session semiconductor index regime from recent sector river images, then buys only the strongest constituents on confident bullish days.

**Architecture:** Reuse the monthly frozen river topology and daily river slices. Add a compact Bayesian CNN encoder with an explicit hidden-state transition matrix, a three-regime head, and a coarse next-state decoder; join its sample-out predictions with the existing constituent relative-strength model through a long-only gate.

**Tech Stack:** Python 3.12, NumPy, PyTorch, DuckDB, FastAPI/Pydantic, plain JavaScript/CSS, pytest.

---

### Task 1: Real sector-index data

**Files:**
- Create: `scripts/sync_sector_index.py`
- Modify: `src/bnhmm/data/source.py`
- Test: `tests/test_sector_index_source.py`

- [ ] Write parsing and DuckDB-source tests for the `sector_index_1d` schema and verify they fail because the parser/source API does not exist.
- [ ] Implement the bounded 10jqka yearly response parser, idempotent table upsert, and `fetch_sector_index()` methods for direct DuckDB and broker sources.
- [ ] Run `PYTHONPATH=src python -m pytest tests/test_sector_index_source.py -q` and commit.
- [ ] Import `881121` data for 2023-2026 into the 145 semiconductor database and verify date coverage.

### Task 2: Causal images and dynamic labels

**Files:**
- Create: `src/bnhmm/sector_flow/regime_data.py`
- Modify: `src/bnhmm/sector_flow/types.py`
- Test: `tests/test_sector_flow_regime_data.py`

- [ ] Write failing tests proving fixed image shape, valid-mask preservation, short context windows, next-session alignment, and volatility-scaled three-state labels that never use the target return in their scale.
- [ ] Implement `SectorStateImage`, `SectorRegimeLabel`, `SectorRegimeSample`, image pooling, context assembly, and causal label construction.
- [ ] Run the focused tests and commit.

### Task 3: Bayesian neural HMM

**Files:**
- Create: `src/bnhmm/sector_flow/regime_model.py`
- Test: `tests/test_sector_flow_regime_model.py`

- [ ] Write failing tests for tensor shapes, row-stochastic transition probabilities, sequential posterior filtering, three-state probability normalization, coarse next-image output, and Monte Carlo uncertainty.
- [ ] Implement a two-block residual CNN, 12-state evidence encoder, explicit transition matrix, Bayesian regime head, and Bayesian coarse-state decoder.
- [ ] Run the focused tests and commit.

### Task 4: Training and sample-out evaluation

**Files:**
- Create: `src/bnhmm/sector_flow/regime_training.py`
- Create: `src/bnhmm/sector_flow/regime_evaluation.py`
- Test: `tests/test_sector_flow_regime_training.py`
- Test: `tests/test_sector_flow_regime_evaluation.py`

- [ ] Write failing tests for class-balanced loss, auxiliary state loss, matured-target filtering, posterior prediction records, balanced accuracy, macro F1, Brier, Log Loss, confusion matrix, and baseline comparison.
- [ ] Implement deterministic seeded training, Bayesian KL scaling, posterior sampling and evaluation.
- [ ] Run the focused tests and commit.

### Task 5: Two-layer long-only policy

**Files:**
- Create: `src/bnhmm/sector_flow/regime_policy.py`
- Test: `tests/test_sector_flow_regime_policy.py`

- [ ] Write failing tests proving that only confident bullish predictions open positions, neutral/bearish/uncertain predictions remain in cash, top-K uses constituent relative scores, and costs apply only when trades occur.
- [ ] Implement daily open-to-close policy observations and timing/selection metrics.
- [ ] Run the focused tests and commit.

### Task 6: Pipeline, CLI and artifacts

**Files:**
- Modify: `src/bnhmm/sector_flow/pipeline.py`
- Modify: `src/bnhmm/sector_flow/__init__.py`
- Modify: `src/bnhmm/cli.py`
- Test: `tests/test_sector_flow_pipeline.py`
- Test: `tests/test_sector_flow_cli.py`

- [ ] Write failing integration tests for `river-regime-v3`, index requirements, strict fold cutoffs, joined stock predictions, and regime/policy artifact manifests.
- [ ] Add configuration fields and the v3 walk-forward branch while preserving v2 read compatibility.
- [ ] Write `sector-regime-predictions.json`, `sector-regime-evaluation.json`, `sector-policy-evaluation.json`, `sector-state-forecasts.npz`, and a Chinese report.
- [ ] Run focused pipeline and CLI tests and commit.

### Task 7: DeepStock Lab configuration and results

**Files:**
- Modify: `src/bnhmm/web/jobs.py`
- Modify: `src/bnhmm/web/app.py`
- Modify: `src/bnhmm/web/static/index.html`
- Modify: `src/bnhmm/web/static/app.js`
- Modify: `src/bnhmm/web/static/styles.css`
- Modify: `src/bnhmm/web/static/river.html`
- Modify: `src/bnhmm/web/static/river.js`
- Modify: `src/bnhmm/web/static/river.css`
- Test: `tests/test_web_jobs.py`
- Test: `tests/test_web_config.py`
- Test: `tests/test_web_static.py`
- Test: `tests/test_river_web.py`

- [ ] Write failing request/default/static tests for the index and BNHMM controls, three-state result panel, policy metrics, and next-state view.
- [ ] Implement validated job arguments, strategy defaults, result rendering, probability bars, policy summary, and predicted-versus-actual state canvases.
- [ ] Run web tests and JavaScript syntax checks, then commit.

### Task 8: Verification, deployment and experiment

**Files:**
- Create: `docs/sector-regime-bnhmm-report.md`

- [ ] Run the complete local test suite with `PYTHONPATH=src python -m pytest -q`.
- [ ] Sync to 145, run the complete production `rheos` test suite, restart the service, and verify `/health`.
- [ ] Run the 2024-2025 semiconductor walk-forward experiment and record prediction, baseline, policy, uncertainty and resource metrics in the Chinese report.
- [ ] Verify desktop and mobile UI, including populated canvases and no overlap.
- [ ] Run TriadMind sync, verify and govern; report runtime-artifact limitations separately.

