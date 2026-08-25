# Short Convection Windows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Precompute 1, 2, 3, and 5-session reverse-flow concentration arrays once and make the 60-day FK strategy read the selected cached horizon, defaulting to 3 sessions.

**Architecture:** Add a fixed horizon axis to the existing convection arrays in `TwoBranchDataset`. The state builder pools trailing minute cells with the signal day's frozen membership, the NPZ cache stores all horizons together, and the gate selects one horizon without touching minute data.

**Tech Stack:** Python 3.12, NumPy, Pydantic, pytest, NPZ, vanilla JavaScript.

---

### Task 1: Define Multi-Horizon Data

**Files:**
- Modify: `tests/test_two_branch_fk_state.py`
- Modify: `src/bnhmm/boundary_fk/types.py`
- Modify: `src/bnhmm/boundary_fk/branch_state.py`

- [ ] Add failing tests for array shape `(sessions, 2, 4)` and horizon order `(1, 2, 3, 5)`.
- [ ] Add a causal fixture where earlier days alter 2/3/5-day values but not 1-day concentration.
- [ ] Implement trailing pooled symbol-minute concentration using the signal day's membership.
- [ ] Run the state tests until green.

### Task 2: Select A Cached Horizon

**Files:**
- Modify: `tests/test_boundary_fk_convection.py`
- Modify: `tests/test_boundary_fk_pipeline.py`
- Modify: `src/bnhmm/boundary_fk/convection.py`
- Modify: `src/bnhmm/boundary_fk/pipeline.py`

- [ ] Add failing tests proving the gate defaults to horizon 3 and selects another horizon explicitly.
- [ ] Add `convection_lookback_sessions` validation with allowed values `1/2/3/5`.
- [ ] Include the selected horizon in prediction and diagnostic artifacts.

### Task 3: Persist All Horizons Once

**Files:**
- Modify: `tests/test_boundary_fk_cache.py`
- Modify: `src/bnhmm/boundary_fk/cache.py`

- [ ] Add a failing round-trip test for the four-dimensional-window arrays and horizon metadata.
- [ ] Upgrade the cache construction version to `v4-short-convection-windows`.
- [ ] Save and load all horizons in one NPZ file.

### Task 4: Expose The Strategy Parameter

**Files:**
- Modify: `tests/test_web_jobs.py`
- Modify: `tests/test_web_config.py`
- Modify: `tests/test_web_static.py`
- Modify: `src/bnhmm/cli.py`
- Modify: `src/bnhmm/boundary_fk/experiment.py`
- Modify: `src/bnhmm/web/jobs.py`
- Modify: `src/bnhmm/web/app.py`
- Modify: `src/bnhmm/web/static/index.html`
- Modify: `src/bnhmm/web/static/app.js`

- [ ] Add a default-3 request field and CLI argument restricted to `1/2/3/5`.
- [ ] Pass the selected horizon into the pipeline without rebuilding data.
- [ ] Add a compact selector to the PathEfficiencyFK configuration and show the selected horizon in results.
- [ ] Bump static asset versions and run focused Web tests.

### Task 5: Verify And Run Real Data

**Files:**
- Modify: `docs/path-efficiency-convection-gate-report.md`

- [ ] Run JavaScript syntax check and the full local test suite.
- [ ] Sync to 145 and run the full `rheos` test suite.
- [ ] Build the 60-day topology cache once; verify it contains horizons `1/2/3/5`.
- [ ] Run the default 3-day gate against the unchanged baseline.
- [ ] Record correlation, returns, drawdown, exposure and concentration-band diagnostics in Chinese.

