# History Panel Actions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move edit and batch-delete controls into the history panel header and remove page-level and single-result deletion controls.

**Architecture:** Keep the existing batch-selection state and API deletion flow. Change only the control placement and state rendering: the history header owns both actions, while the secondary selection strip owns only select-all and selected-count information.

**Tech Stack:** Static HTML, CSS, vanilla JavaScript, Python unittest static contracts, browser smoke test.

---

### Task 1: Lock the layout contract

**Files:**
- Modify: `tests/test_web_static.py`

- [ ] Add a static test that extracts the workspace header and history navigation HTML and asserts `edit-history-button` and `batch-delete-button` occur only inside the history navigation.
- [ ] Assert `delete-button` and `deleteSelectedJob` no longer exist.
- [ ] Run `PYTHONPATH=src python -m pytest tests/test_web_static.py -q` and confirm failure against the old placement.

### Task 2: Move controls and simplify behavior

**Files:**
- Modify: `src/bnhmm/web/static/index.html`
- Modify: `src/bnhmm/web/static/app.js`
- Modify: `src/bnhmm/web/static/styles.css`

- [ ] Remove the edit button from `.header-actions` and the single-result delete button from `.run-actions`.
- [ ] Add icon-only edit and delete buttons to `.section-title` inside `.run-history`, retaining the existing IDs so batch behavior stays connected.
- [ ] Remove the duplicate delete button from `history-batch-actions`; keep only select-all and selected-count.
- [ ] Update `renderHistoryEditor` so edit toggles pencil/check icons and delete is disabled unless editing has selected eligible jobs.
- [ ] Remove `deleteSelectedJob` and its event listener; retain `deleteSelectedJobs` as the only deletion path.
- [ ] Add fixed history-header action dimensions and change the selection strip to two columns.
- [ ] Bump static asset versions in `index.html`.
- [ ] Run the static test and `node --check src/bnhmm/web/static/app.js` until both pass.

### Task 3: Verify and deploy

**Files:**
- Modify: `docs/superpowers/specs/2026-08-25-history-panel-actions-design.md`
- Modify: `docs/superpowers/plans/2026-08-25-history-panel-actions.md`

- [ ] Run the full test suite, `git diff --check`, JavaScript syntax check, and TriadMind sync/verify.
- [ ] Sync changed files to `/home/wujianfeng/private/bnhmm-engine`, normalize line endings, run focused server tests, and restart `bnhmm-web.service`.
- [ ] Browser-check that global header has no edit/delete, history header has both, delete enables after selecting a completed result, and the selected result toolbar has no delete.
- [ ] Commit on server `main`, push GitLab, and verify `/health`.
