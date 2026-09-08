# Rolling Anchor FK Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This project must remain single-agent.

**Goal:** Replace daily-reset FK inputs with causal 60-session continuous price paths, remove failed strategies, and rebuild all derived datasets.

**Architecture:** Extend the FK normalized panel with raw price/validity, then let the whole-sector state builder create one rolling 60-session coordinate per signal day using the close immediately before the window. Versioned caches make all old derived data unreadable. The Web and CLI expose only BNHMM and boundary-flow FK.

**Tech Stack:** Python 3.12+, NumPy, SciPy, FastAPI, vanilla JavaScript, pytest, DuckDB.

**Spec:** `docs/superpowers/specs/2026-09-08-rolling-anchor-fk-cleanup-design.md`

## Global Constraints

- No future data may enter an anchor, feature, threshold, fold or model posterior.
- A 60-session sample uses one fixed per-symbol anchor from the immediately preceding completed session.
- Original DuckDB bars and sector membership definitions must not be deleted.
- Only BNHMM and boundary-flow FK remain user-selectable.
- Development is single-agent only.

---

### Task 1: Define The Continuous 60-Session State Contract

**Files:**
- Modify: `src/bnhmm/boundary_fk/types.py`
- Modify: `src/bnhmm/boundary_fk/normalization.py`
- Modify: `src/bnhmm/boundary_fk/branch_state.py`
- Modify: `src/bnhmm/boundary_fk/sector_state.py`
- Test: `tests/test_boundary_fk_normalization.py`
- Test: `tests/test_sector_fk_state.py`

**Interfaces:**
- `NormalizedBoundaryPanel` additionally exposes `close` and `price_valid` arrays aligned with timestamps and symbols.
- `SectorDailyStateBuilder(..., history_sessions=60)` emits `rolling_log_position_60d`, `rolling_drift_60d`, `rolling_diffusion_60d`, and `rolling_path_efficiency_60d`.

- [ ] Write a failing synthetic test with 61 sessions, an overnight gap, and monotone prices; assert the last state uses session 0 close as anchor and does not reset at any daily boundary.
- [ ] Run `python -m pytest tests/test_boundary_fk_normalization.py tests/test_sector_fk_state.py -q` and confirm failure because the rolling coordinate fields do not exist.
- [ ] Preserve raw close and validity in `CausalBoundaryNormalizer.transform`.
- [ ] Build the equal-weight sector coordinate and its path increments over exactly 60 sessions, including overnight transitions.
- [ ] Change the next-day label to close-to-close return and run the focused tests to green.

### Task 2: Feed The Correct State Into Bayesian FK

**Files:**
- Modify: `src/bnhmm/boundary_fk/branch_model.py`
- Modify: `src/bnhmm/boundary_fk/hurdle_model.py`
- Modify: `src/bnhmm/boundary_fk/pipeline.py`
- Test: `tests/test_boundary_fk_branch_model.py`
- Test: `tests/test_boundary_fk_hurdle_model.py`
- Test: `tests/test_sector_fk_pipeline.py`

**Interfaces:**
- FK direction feature set is `("rolling_log_position_60d", "rolling_path_efficiency_60d")`.
- Hurdle movement feature set additionally includes `rolling_diffusion_60d` while its direction design remains directional.

- [ ] Write failing tests asserting the new named features are the only default FK direction inputs and that future rows do not affect fitted posteriors.
- [ ] Run the focused tests and confirm failure on the old daily feature names.
- [ ] Update the low-parameter model configuration and artifact metadata.
- [ ] Run the focused tests to green and verify every prediction retains `training_cutoff`, `signal_time`, and `target_time`.

### Task 3: Invalidate Old Derived Data

**Files:**
- Modify: `src/bnhmm/boundary_fk/cache.py`
- Modify: `src/bnhmm/boundary_fk/precompute.py`
- Test: `tests/test_boundary_fk_cache.py`
- Test: `tests/test_boundary_fk_precompute.py`

**Interfaces:**
- `CACHE_FORMAT_VERSION = 3`.
- Construction protocol is `sector-rolling-anchor-60d-v2`.

- [ ] Write a failing cache test proving a v2/daily-reset manifest cannot load.
- [ ] Run the cache tests and confirm the old cache is still accepted.
- [ ] Bump format/construction versions and include the anchor protocol in manifests and build summaries.
- [ ] Run the cache/precompute tests to green.

### Task 4: Remove Failed Strategies

**Files:**
- Modify: `src/bnhmm/web/jobs.py`
- Modify: `src/bnhmm/web/app.py`
- Modify: `src/bnhmm/web/static/index.html`
- Modify: `src/bnhmm/web/static/app.js`
- Modify: `src/bnhmm/cli.py`
- Delete: `src/bnhmm/fk_stock/`
- Delete: `src/bnhmm/percolation_branching/`
- Delete: `src/bnhmm/stock_transformer/`
- Delete: `src/bnhmm/repair/`
- Delete only model/training files under `src/bnhmm/sector_flow/`; preserve panel/types utilities required by FK.
- Test: `tests/test_web_jobs.py`
- Test: `tests/test_web_static.py`
- Test: `tests/test_web_config.py`

**Interfaces:**
- `WalkForwardJobRequest.strategy` accepts only `bnhmm` and `boundary-flow-fk`.
- `build_command` has exactly two current dispatch paths.

- [ ] Write failing tests that reject every removed strategy and assert only two strategy controls exist in HTML.
- [ ] Run the Web tests and confirm they fail against the old strategy catalog.
- [ ] Remove request fields, dispatch functions, UI panels and CLI subcommands that are exclusive to failed algorithms.
- [ ] Delete unreferenced implementation modules and their exclusive tests.
- [ ] Run import, Web and CLI tests to green.

### Task 5: Rebuild And Validate On 145

**Files:**
- Modify: `docs/sector-direct-direction-report-20260908.md`
- Generated only on 145: boundary caches and Web job artifacts.

**Interfaces:**
- Three rebuilt datasets report `anchor_protocol=rolling-60-session-previous-close`.

- [ ] Run the full local suite with `PYTHONPATH=src python -m pytest -q`.
- [ ] Push the feature branch and fast-forward the 145 worktree.
- [ ] Verify no active Web job, stop only port 8877, and delete the verified derived cache/job roots while preserving raw DuckDB files.
- [ ] Rebuild semiconductor, artificial-intelligence and new-energy 60-session caches from raw one-minute bars.
- [ ] Run monthly walk-forward FK for all three datasets and save the new Chinese comparison report.
- [ ] Restart port 8877, verify `/health`, confirm only two strategies in the browser, and inspect console errors.
