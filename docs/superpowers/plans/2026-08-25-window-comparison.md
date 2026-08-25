# PathEfficiencyFK Window Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one persisted six-window PathEfficiencyFK comparison with a dedicated results page.

**Architecture:** Add a small `ComparisonManager` beside `JobManager`; it submits six ordinary jobs and derives batch state and metrics from them. Add REST endpoints and a focused frontend comparison view that reuses existing job artifacts.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, vanilla JavaScript, HTML Canvas, unittest/pytest.

---

### Task 1: Persisted comparison manager

**Files:**
- Create: `src/bnhmm/web/comparisons.py`
- Create: `tests/test_web_comparisons.py`

- [ ] Write failing tests proving the manager creates exactly `(90, 60, 40, 30, 20, 10)`, gives each child a unique name, survives reload, and derives partial/final status without stopping after one failed child.
- [ ] Run `python -m pytest tests/test_web_comparisons.py -q` and confirm failure because the module is absent.
- [ ] Implement `ComparisonRecord`, `ComparisonManager.create`, `list`, `get`, `delete`, and `payload` using JSON files under `_comparisons`.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Comparison API

**Files:**
- Modify: `src/bnhmm/web/app.py`
- Modify: `src/bnhmm/web/jobs.py`
- Create: `tests/test_web_comparison_api.py`

- [ ] Write failing API-level tests for dataset replacement, PathEfficiencyFK-only validation, six child submissions, and metrics/equity aggregation.
- [ ] Run `python -m pytest tests/test_web_comparison_api.py -q` and confirm the routes are missing.
- [ ] Initialize `ComparisonManager` in lifespan and add POST/GET/DELETE comparison routes.
- [ ] Build each equity curve from `gross_strategy_return`, `turnover`, and the shared transaction cost.
- [ ] Run focused API and existing job tests.

### Task 3: Comparison entry and results view

**Files:**
- Modify: `src/bnhmm/web/static/index.html`
- Modify: `src/bnhmm/web/static/app.js`
- Modify: `src/bnhmm/web/static/styles.css`
- Modify: `tests/test_web_static.py`

- [ ] Add failing static assertions for the six-window button, comparison list, summary metrics, table, canvas, and child-job navigation.
- [ ] Run `python -m pytest tests/test_web_static.py -q` and confirm the new hooks are absent.
- [ ] Add the PathEfficiencyFK-only submit button and a comparison list beside history tasks.
- [ ] Add the dedicated comparison view, progress polling, six-row table, readable summary, overlaid equity chart, and child result navigation.
- [ ] Run static tests and `node --check src/bnhmm/web/static/app.js`.

### Task 4: Verification and real run

**Files:**
- Modify: `docs/path-efficiency-window-comparison-report.md`

- [ ] Run `python -m pytest -q` and `git diff --check` locally.
- [ ] Sync topology and run TriadMind verification.
- [ ] Deploy to 145, run the server test suite with the `rheos` Python environment, and restart the service.
- [ ] Submit the semiconductor six-window comparison through the API/UI and wait for all child jobs.
- [ ] Verify the comparison page at desktop and mobile widths, including nonblank equity lines and working child links.
- [ ] Record the six-window metrics and plain-language conclusion in the report, commit, and push `main`.
