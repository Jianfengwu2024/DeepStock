# Reverse Flow Concentration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the path-efficiency experiment's reverse-flow share with a normalized joint symbol-minute concentration metric and validate it on real semiconductor data.

**Architecture:** Compute normalized HHI inside the existing two-branch daily state builder, persist it through a cache-version bump, and reuse the existing causal quantile gate. Keep the FK forecast and portfolio policy unchanged so the experiment isolates only the new convection statistic.

**Tech Stack:** Python 3.12, NumPy, pytest, NPZ cache, FastAPI static UI.

---

### Task 1: Lock The Concentration Contract

**Files:**
- Modify: `tests/test_two_branch_fk_state.py`

- [ ] Add a test where equal reverse total flow is concentrated in one symbol-minute cell versus spread uniformly across several cells.
- [ ] Assert the concentrated case has a strictly higher normalized HHI.
- [ ] Add exact endpoint tests: one-cell concentration equals `1.0`; uniform flow across all valid cells equals `0.0`.
- [ ] Run `pytest -q tests/test_two_branch_fk_state.py` and verify the old ratio implementation fails these assertions.

### Task 2: Implement Joint Normalized HHI

**Files:**
- Modify: `src/bnhmm/boundary_fk/branch_state.py`
- Modify: `tests/test_two_branch_fk_state.py`

- [ ] Replace `opposing / total_flow` with normalized HHI over all valid symbol-minute cells.
- [ ] Return invalid when there are fewer than two observations, no branch direction, no total flow, or no opposing flow.
- [ ] Keep `convection_total_flow` as the opposing absolute flow used only for diagnostics and validity.
- [ ] Run `pytest -q tests/test_two_branch_fk_state.py tests/test_boundary_fk_convection.py` and verify all tests pass.

### Task 3: Invalidate Old Caches And Rename Diagnostics

**Files:**
- Modify: `src/bnhmm/boundary_fk/cache.py`
- Modify: `src/bnhmm/boundary_fk/reporting.py`
- Modify: `src/bnhmm/web/static/app.js`
- Modify: `tests/test_boundary_fk_cache.py`
- Modify: `tests/test_boundary_fk_reporting.py`
- Modify: `tests/test_web_static.py`

- [ ] Bump `CACHE_CONSTRUCTION_VERSION` from the old share version to a concentration-specific version.
- [ ] Update Chinese labels and report wording from reverse-flow share to reverse-flow concentration.
- [ ] Preserve artifact names so existing API discovery remains stable.
- [ ] Run the focused cache, report, and web tests.

### Task 4: Regression Verification

**Files:**
- Modify only files required by failing regression tests.

- [ ] Run `node --check src/bnhmm/web/static/app.js`.
- [ ] Run `pytest -q` and require all tests to pass.
- [ ] Run `git diff --check` and inspect the final diff.

### Task 5: Real Data Experiment And Deployment

**Files:**
- Modify: `docs/path-efficiency-convection-gate-report.md`

- [ ] Apply the commits to `/home/wujianfeng/private/bnhmm-engine` on `192.168.1.145`.
- [ ] Rebuild the semiconductor 60-day cache with the new construction version.
- [ ] Run `PathEfficiencyFK-60D-Baseline` and the concentration-gate experiment over the same walk-forward interval.
- [ ] Record return, drawdown, exposure, turnover, gate triggers, band diagnostics, and Rank correlation in Chinese.
- [ ] Run the full server test suite in the `rheos` environment, push GitLab `main`, restart `bnhmm-web.service`, and verify `/health`.

