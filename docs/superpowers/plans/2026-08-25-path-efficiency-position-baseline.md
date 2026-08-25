# Path Efficiency Position Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pure signed path efficiency the only TwoBranchFK drift model and map each branch's bull/flat/bear forecast to 50%/25%/0% of total capital.

**Architecture:** Replace the boolean hold-state machine with absolute per-branch target exposures, so a flat signal always means a fixed half branch budget rather than recursively halving the previous position. Simplify the walk-forward pipeline to fit and evaluate only `path_efficiency_only`, while preserving old files only as historical artifacts. Remove multi-signal ablation rendering from new reports and the Web UI.

**Tech Stack:** Python 3.12, NumPy, pytest, FastAPI static UI, existing Bayesian branch regression and FK forecaster.

---

### Task 1: Absolute Branch Target Exposure

**Files:**
- Modify: `src/bnhmm/boundary_fk/policy.py`
- Test: `tests/test_boundary_fk_policy.py`

- [ ] **Step 1: Write failing position-map tests**

Replace hold-state expectations with explicit target exposure assertions:

```python
def test_branch_directions_map_to_absolute_capital_exposure() -> None:
    decision = BoundaryPolicy().decide(
        (_forecast("bull"), _forecast("flat")),
        branches=(("A", "B"), ("C", "D")),
        neutral_thresholds=(0.002, 0.002),
    )
    assert decision.branch_target_exposures == pytest.approx((0.5, 0.25))
    assert sum(decision.target_weights) == pytest.approx(0.75)
    assert decision.target_weights == pytest.approx((0.25, 0.25, 0.125, 0.125))


def test_repeated_flat_signal_stays_at_fixed_half_position() -> None:
    first = BoundaryPolicy().decide(
        (_forecast("flat"), _forecast("flat")),
        branches=(("A", "B"), ("C", "D")),
        neutral_thresholds=(0.002, 0.002),
    )
    second = BoundaryPolicy().decide(
        (_forecast("flat"), _forecast("flat")),
        branches=(("A", "B"), ("C", "D")),
        neutral_thresholds=(0.002, 0.002),
    )
    assert first.branch_target_exposures == second.branch_target_exposures == (0.25, 0.25)
    assert sum(second.target_weights) == pytest.approx(0.5)
```

Also cover bull/bull = 100%, bull/bear = 50%, flat/bear = 25%, and bear/bear = 0%.

- [ ] **Step 2: Run policy tests and verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_boundary_fk_policy.py -q
```

Expected: failures because `branch_target_exposures` does not exist and `decide` still requires `held_branches`.

- [ ] **Step 3: Implement absolute exposures**

Add to `BoundaryPolicyDecision`:

```python
branch_target_exposures: tuple[float, float] = (0.0, 0.0)
```

Remove `held_branches` from `BoundaryPolicy.decide` input. Map directions with:

```python
BRANCH_EXPOSURE = {"bull": 0.5, "flat": 0.25, "bear": 0.0}
```

For each nonzero branch exposure, assign every member `branch_exposure / len(branch_symbols)`. Keep `held_branches` in the returned decision as `(exposure > 0)` compatibility metadata. Set `open_position` from total target exposure.

- [ ] **Step 4: Run policy tests and verify pass**

Run:

```bash
PYTHONPATH=src pytest tests/test_boundary_fk_policy.py -q
```

Expected: all policy tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bnhmm/boundary_fk/policy.py tests/test_boundary_fk_policy.py
git commit -m "feat: map FK directions to fixed target exposure"
```

### Task 2: Single Pure-Path Walk-Forward Pipeline

**Files:**
- Modify: `src/bnhmm/boundary_fk/pipeline.py`
- Modify: `src/bnhmm/boundary_fk/evaluation.py`
- Test: `tests/test_boundary_fk_pipeline.py`
- Test: `tests/test_boundary_fk_evaluation.py`

- [ ] **Step 1: Write failing single-model pipeline tests**

Require the produced stage and posterior payloads to use only pure path efficiency:

```python
assert result.stages[0].drift_feature_names == ("signed_path_efficiency",)
assert {item["policy"] for item in posteriors} == {"path_efficiency_only"}
assert not (tmp_path / "boundary-policy-ablation.json").exists()
```

Require prediction payloads to include `branch_target_exposures`, and assert every `sum(target_weights)` belongs to `{0.0, 0.25, 0.5, 0.75, 1.0}` within floating tolerance.

Add evaluation coverage:

```python
assert metrics.average_exposure == pytest.approx(
    np.mean([sum(item.decision.target_weights) for item in observations])
)
```

- [ ] **Step 2: Run pipeline and evaluation tests and verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_boundary_fk_pipeline.py tests/test_boundary_fk_evaluation.py -q
```

Expected: failures because multiple policies are still fitted, the ablation artifact exists, and average exposure is absent.

- [ ] **Step 3: Simplify the pipeline**

Replace `POLICY_ABLATION_NAMES` and `DRIFT_FEATURE_SET_BY_POLICY` with:

```python
PRIMARY_POLICY_NAME = "path_efficiency_only"
PRIMARY_DRIFT_FEATURE_SET = "path_efficiency_only"
```

For each fold and branch, fit one `BayesianBranchModel` using `PRIMARY_DRIFT_FEATURE_SET`; forecast it once and apply one `BoundaryPolicy`. Remove policy-indexed posterior, observation, held-state, and weight dictionaries. Keep one `previous_weights` dictionary solely for causal turnover calculation. Do not write `boundary-policy-ablation.json`.

Add `branch_target_exposures` to `_prediction_payload`. Keep `boundary-policy-evaluation.json` as the canonical strategy result.

- [ ] **Step 4: Add average exposure to evaluation**

Add `average_exposure: float` to `BoundaryEvaluationMetrics`. In `evaluate_two_branch_forecasts`, calculate:

```python
exposure = np.asarray([sum(item.decision.target_weights) for item in observations])
average_exposure = float(np.mean(exposure))
open_rate = float(np.mean(exposure > 0.0))
```

Continue calculating turnover and transaction costs from actual target-weight changes.

- [ ] **Step 5: Run focused pipeline tests and verify pass**

Run:

```bash
PYTHONPATH=src pytest tests/test_boundary_fk_policy.py tests/test_boundary_fk_pipeline.py tests/test_boundary_fk_evaluation.py -q
```

Expected: all focused tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/bnhmm/boundary_fk/pipeline.py src/bnhmm/boundary_fk/evaluation.py tests/test_boundary_fk_pipeline.py tests/test_boundary_fk_evaluation.py
git commit -m "feat: make pure path efficiency the FK baseline"
```

### Task 3: Single-Strategy Report and Web Result

**Files:**
- Modify: `src/bnhmm/boundary_fk/reporting.py`
- Modify: `src/bnhmm/web/static/index.html`
- Modify: `src/bnhmm/web/static/app.js`
- Test: `tests/test_boundary_fk_reporting.py`
- Test: `tests/test_web_static.py`

- [ ] **Step 1: Write failing report and UI contract tests**

Require the report to include `纯路径效率`, `多 50% / 平 25% / 空 0%（每条支流）`, and `平均仓位`, while excluding `日内物理特征消融`.

Require the HTML to omit `boundary-ablation-table` and the `新旧信号消融` artifact link. Require the JavaScript to render `metrics.average_exposure` and contain no multi-signal label map.

- [ ] **Step 2: Run report and UI tests and verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_boundary_fk_reporting.py tests/test_web_static.py -q
```

Expected: failures because the ablation table and labels remain and average exposure is not rendered.

- [ ] **Step 3: Convert reporting to one strategy**

Change `write_ablation_report` to accept only stage results and write a pure-path strategy report. Its primary metrics table must include:

```text
模型 | 漂移输入 | 仓位规则 | 平均仓位 | 开仓率 | 换手 | 净收益 | 最大回撤
```

Do not render a multi-policy comparison table.

- [ ] **Step 4: Remove the Web ablation surface**

Delete the “信号消融” table from `index.html`, remove its renderer and label map from `app.js`, and remove the ablation artifact link. Add “平均仓位” to the remaining TwoBranchFK stage table. Increment the `app.js` cache version.

- [ ] **Step 5: Run focused UI tests and syntax checks**

Run:

```bash
PYTHONPATH=src pytest tests/test_boundary_fk_reporting.py tests/test_web_static.py -q
node --check src/bnhmm/web/static/app.js
git diff --check
```

Expected: all commands pass.

- [ ] **Step 6: Commit**

```bash
git add src/bnhmm/boundary_fk/reporting.py src/bnhmm/web/static/index.html src/bnhmm/web/static/app.js tests/test_boundary_fk_reporting.py tests/test_web_static.py
git commit -m "feat: present pure path FK as a single strategy"
```

### Task 4: Full Verification, 145 Experiment, and Deployment

**Files:**
- Create: `docs/path-efficiency-position-baseline-report-20260825.md`
- Modify only if a verification failure identifies a real defect.

- [ ] **Step 1: Run the complete local suite**

Run:

```bash
PYTHONPATH=src pytest -q
node --check src/bnhmm/web/static/app.js
git diff --check
```

Expected: all tests pass; existing PyTorch Transformer warnings are acceptable.

- [ ] **Step 2: Run TriadMind topology checks**

Synchronize the topology and require `triadmind_verify` to pass. Record unavailable runtime governance artifacts as an environment limitation rather than creating unrelated files.

- [ ] **Step 3: Deploy to server 145**

Synchronize changed files to `/home/wujianfeng/private/bnhmm-engine`, normalize LF line endings, run focused tests with `/home/wujianfeng/miniconda3/envs/rheos/bin/python`, restart `bnhmm-web`, and require `/api/config` to return HTTP 200.

- [ ] **Step 4: Run a same-protocol semiconductor experiment**

Copy the completed `TwoBranchFK-002` dates, semiconductor 96-stock dataset, costs, FK paths, posterior gates, half-life, and all-fold setting into a new task. Verify that the new task produces only `signed_path_efficiency`, includes branch target exposures, and has no multi-signal ablation artifact.

- [ ] **Step 5: Write the Chinese result report**

Record average exposure, turnover, net return, drawdown, drift direction accuracy, drift rank IC, and the comparison against `TwoBranchFK-002`. Explain whether the new position mapping improved risk-adjusted behavior without claiming stability from one experiment.

- [ ] **Step 6: Browser acceptance**

Open `http://192.168.1.145:8765/`, select the new task, and verify that the result page shows pure path efficiency, average exposure, and no signal-ablation table. Require no application console errors.

- [ ] **Step 7: Commit and push**

Commit the implementation and report to server `main`, push GitLab, then verify server `HEAD == origin/main`, the worktree is clean, `bnhmm-web` is active, and `/api/config` returns HTTP 200.
