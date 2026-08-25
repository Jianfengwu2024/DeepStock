# PathEfficiencyFK 60 日对流门控 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定 60 日 PathEfficiencyFK 基线上加入只减仓、不反向开仓的反向微观流量门控，并完成严格样本外对照实验。

**Architecture:** `TwoBranchDailyStateBuilder` 在构造每日支流状态时同时计算成员级反向流量占比，`TwoBranchDataset` 和版本化 NPZ 缓存负责持久化。新的 `ConvectionGate` 只根据训练截止日前的经验分位数给出支流仓位上限，`BoundaryFlowPipeline` 先调用原策略再应用上限，不修改漂移后验或方向判定。

**Tech Stack:** Python 3.11/3.13、NumPy、pytest、FastAPI/Pydantic、原生 HTML/CSS/JavaScript、DuckDB 数据缓存。

---

### Task 1: 对流日状态与缓存契约

**Files:**
- Modify: `src/bnhmm/boundary_fk/types.py`
- Modify: `src/bnhmm/boundary_fk/branch_state.py`
- Modify: `src/bnhmm/boundary_fk/cache.py`
- Test: `tests/test_two_branch_fk_state.py`
- Test: `tests/test_boundary_fk_cache.py`

- [ ] **Step 1: 写入失败测试，定义反向微观流量公式**

在固定测试日设置支流净方向为正，并给成员分钟速度安排同向与反向绝对流量，断言：

```python
assert result.convection_valid[-1, 0]
assert result.convection_strength[-1, 0] == pytest.approx(
    opposing_absolute_flow / total_absolute_flow
)
```

同时覆盖净方向为零、有效分钟不足时 `convection_valid=False`。

- [ ] **Step 2: 运行状态测试并确认因字段不存在而失败**

Run: `pytest -q tests/test_two_branch_fk_state.py`

Expected: FAIL，提示 `TwoBranchDataset` 没有 `convection_strength`。

- [ ] **Step 3: 最小实现日级对流数组**

在 `TwoBranchDataset` 增加以下字段：

```python
convection_strength: NDArray[np.float64]
convection_valid: NDArray[np.bool_]
convection_total_flow: NDArray[np.float64]
convection_observations: NDArray[np.int64]
```

在 `TwoBranchDailyStateBuilder` 先取得当日支流平均分钟速度的总和方向，再对所有有效成员分钟速度计算：

```python
total = float(np.sum(np.abs(member_velocity)))
opposing = float(np.sum(np.abs(member_velocity[member_velocity * direction < 0.0])))
strength = opposing / total
```

不把对流强度放进漂移 `feature_names`，避免模型自动把它当作方向输入。

- [ ] **Step 4: 运行状态测试并确认通过**

Run: `pytest -q tests/test_two_branch_fk_state.py`

Expected: PASS。

- [ ] **Step 5: 写入缓存往返失败测试**

保存并载入测试数据，断言四个对流数组逐项相同；把构造版本更新为 `path-efficiency-two-branch-v2-convection`，旧 manifest 必须产生 cache miss。

- [ ] **Step 6: 运行缓存测试并确认失败**

Run: `pytest -q tests/test_boundary_fk_cache.py`

Expected: FAIL，提示 NPZ 未保存对流字段或版本不匹配。

- [ ] **Step 7: 保存并载入对流字段，提升缓存版本**

在 `BoundaryDataCache.save/load` 增加四个无 pickle 数组，并更新 `CACHE_CONSTRUCTION_VERSION`。

- [ ] **Step 8: 运行状态和缓存测试**

Run: `pytest -q tests/test_two_branch_fk_state.py tests/test_boundary_fk_cache.py`

Expected: PASS。

- [ ] **Step 9: 提交数据层改动**

```bash
git add src/bnhmm/boundary_fk/types.py src/bnhmm/boundary_fk/branch_state.py src/bnhmm/boundary_fk/cache.py tests/test_two_branch_fk_state.py tests/test_boundary_fk_cache.py
git commit -m "feat: cache branch convection strength"
```

### Task 2: 因果分位数和只减仓门控

**Files:**
- Create: `src/bnhmm/boundary_fk/convection.py`
- Modify: `src/bnhmm/boundary_fk/policy.py`
- Test: `tests/test_boundary_fk_convection.py`
- Test: `tests/test_boundary_fk_policy.py`

- [ ] **Step 1: 写入失败测试，定义训练期分位数**

构造训练截止日前后两段极端不同的数据，拟合门控器并断言未来值不改变 `q50/q75/q90`：

```python
gate = ConvectionGate.fit(data, training_cutoff=cutoff, minimum_rows=30)
np.testing.assert_allclose(gate.thresholds[0], expected_training_quantiles)
```

- [ ] **Step 2: 运行新测试并确认导入失败**

Run: `pytest -q tests/test_boundary_fk_convection.py`

Expected: FAIL，提示 `bnhmm.boundary_fk.convection` 不存在。

- [ ] **Step 3: 实现独立门控器**

新增不可变对象：

```python
@dataclass(frozen=True, slots=True)
class ConvectionGate:
    thresholds: tuple[tuple[float, float, float] | None, ...]

    @classmethod
    def fit(cls, data: TwoBranchDataset, training_cutoff: np.datetime64,
            minimum_rows: int = 30) -> "ConvectionGate": ...

    def exposure_cap(self, row: int, branch: int, data: TwoBranchDataset) -> float: ...
```

仓位上限固定为 `0.5/0.375/0.25/0.0`，无效指标或样本不足返回 `0.5`。

- [ ] **Step 4: 运行门控测试并确认通过**

Run: `pytest -q tests/test_boundary_fk_convection.py`

Expected: PASS。

- [ ] **Step 5: 写入失败测试，证明门控只能减少仓位**

覆盖：基线零仓不能被恢复、75% 与 50% 上限、极强对流只清当前支流、任一 `bear` 仍全局清仓。

- [ ] **Step 6: 运行策略测试并确认失败**

Run: `pytest -q tests/test_boundary_fk_policy.py`

Expected: FAIL，提示 `decide` 尚不接受支流仓位上限。

- [ ] **Step 7: 在原决策之后应用支流上限**

给 `BoundaryPolicy.decide` 增加默认不限制的参数：

```python
branch_exposure_caps: tuple[float, float] = (0.5, 0.5)
```

对非 `bear` 状态取 `min(base_exposure, cap)`；门控清仓后的状态必须为 `BranchPositionState()`，不能保留平仓计数。`bear` 的全局清仓在门控之前执行。

- [ ] **Step 8: 运行门控与策略测试**

Run: `pytest -q tests/test_boundary_fk_convection.py tests/test_boundary_fk_policy.py`

Expected: PASS。

- [ ] **Step 9: 提交门控层改动**

```bash
git add src/bnhmm/boundary_fk/convection.py src/bnhmm/boundary_fk/policy.py tests/test_boundary_fk_convection.py tests/test_boundary_fk_policy.py
git commit -m "feat: add causal convection exposure gate"
```

### Task 3: 双模型 walk-forward 消融与诊断

**Files:**
- Modify: `src/bnhmm/boundary_fk/pipeline.py`
- Modify: `src/bnhmm/boundary_fk/evaluation.py`
- Modify: `src/bnhmm/boundary_fk/reporting.py`
- Test: `tests/test_boundary_fk_pipeline.py`
- Test: `tests/test_boundary_fk_evaluation.py`
- Test: `tests/test_boundary_fk_reporting.py`

- [ ] **Step 1: 写入失败测试，要求一次运行输出两个模型**

断言模型名严格为：

```python
assert result.model_names == (
    "PathEfficiencyFK-60D-Baseline",
    "PathEfficiencyFK-60D-ConvectionGate",
)
```

并断言两个模型共享同一折、同一预测日和同一 FK forecast，只有目标仓位不同。

- [ ] **Step 2: 运行 pipeline 测试并确认失败**

Run: `pytest -q tests/test_boundary_fk_pipeline.py`

Expected: FAIL，当前只返回一个 `PathEfficiencyFK` stage。

- [ ] **Step 3: 在每个 fold 仅拟合一次 FK，再并行形成两个策略观察序列**

每个 fold 使用相同 `posteriors` 和 `forecasts`。基线调用默认仓位上限；实验组使用该 fold 截止日拟合的 `ConvectionGate`。分别维护前一日仓位状态和权重，避免两个策略互相污染。

- [ ] **Step 4: 写入失败测试，定义对流诊断**

评估结果必须包含：四档样本数、下一日平均收益、延续率、反转率、门控次数、对流与路径效率 Rank 相关性，以及每档的收益/回撤影响。

- [ ] **Step 5: 实现最小诊断结构并写入 JSON/Markdown**

诊断只使用当前样本外观察；分档沿用当时 fold 的冻结阈值，不在全样本上重新分位。预测 JSON 每支流增加：

```python
{
    "convection_strength": value,
    "convection_band": band,
    "convection_exposure_cap": cap,
    "convection_gate_reason": reason,
}
```

- [ ] **Step 6: 运行 pipeline、评估和报告测试**

Run: `pytest -q tests/test_boundary_fk_pipeline.py tests/test_boundary_fk_evaluation.py tests/test_boundary_fk_reporting.py`

Expected: PASS。

- [ ] **Step 7: 提交实验层改动**

```bash
git add src/bnhmm/boundary_fk/pipeline.py src/bnhmm/boundary_fk/evaluation.py src/bnhmm/boundary_fk/reporting.py tests/test_boundary_fk_pipeline.py tests/test_boundary_fk_evaluation.py tests/test_boundary_fk_reporting.py
git commit -m "feat: compare 60d convection gate with baseline"
```

### Task 4: API 与 DeepStock 页面

**Files:**
- Modify: `src/bnhmm/web/jobs.py`
- Modify: `src/bnhmm/web/app.py`
- Modify: `src/bnhmm/web/static/app.js`
- Modify: `src/bnhmm/web/static/index.html`
- Test: `tests/test_web_config.py`
- Test: `tests/test_web_jobs.py`
- Test: `tests/test_web_static.py`

- [ ] **Step 1: 写入失败测试，固定 60 日并暴露实验模式**

对 `boundary-flow-fk` 请求断言 `topology_lookback_sessions` 必须为 60；页面策略说明显示“路径效率 + 对流门控”，结果标签显示“对流强度、门控档位、减仓原因”。

- [ ] **Step 2: 运行 Web 测试并确认失败**

Run: `pytest -q tests/test_web_config.py tests/test_web_jobs.py tests/test_web_static.py`

Expected: FAIL，当前页面没有对流字段且允许其他窗口直接运行该基线。

- [ ] **Step 3: 接入双模型结果和中文展示**

保持页面参数精简，不增加可调对流阈值。阈值固定由训练期分位数生成；页面只展示计算结果和门控原因。

- [ ] **Step 4: 运行 Web 测试并确认通过**

Run: `pytest -q tests/test_web_config.py tests/test_web_jobs.py tests/test_web_static.py`

Expected: PASS。

- [ ] **Step 5: 提交 Web 改动**

```bash
git add src/bnhmm/web/jobs.py src/bnhmm/web/app.py src/bnhmm/web/static/app.js src/bnhmm/web/static/index.html tests/test_web_config.py tests/test_web_jobs.py tests/test_web_static.py
git commit -m "feat: expose convection gate experiment in deepstock"
```

### Task 5: 完整验证、145 部署与真实样本外报告

**Files:**
- Create: `docs/path-efficiency-convection-gate-report.md`
- Modify only if test evidence requires it: files from Tasks 1-4

- [ ] **Step 1: 运行完整本地测试**

Run: `pytest -q`

Expected: 全部通过；仅允许已有、已记录的 PyTorch nested-tensor warning。

- [ ] **Step 2: 运行 TriadMind 拓扑验证**

对项目根运行 `triadmind_sync`、`triadmind_verify` 和 `triadmind_govern(mode="check")`，不得出现新增 ghost node、循环或契约完整性退化。

- [ ] **Step 3: 推送并部署到 145**

将当前分支提交推送到 GitLab `main`，145 的 `/home/wujianfeng/private/bnhmm-engine` 拉取同一提交，重建 60 日 v2 缓存并重启 `8765` 服务。

- [ ] **Step 4: 在 145 运行真实对照实验**

半导体数据集、相同样本外区间和 10 bps 成本下，一次运行生成 baseline 与 convection gate 两组结果。记录任务 ID、提交号、缓存版本和运行时间。

- [ ] **Step 5: 核验门控不变量**

逐日验证实验组每条支流仓位不大于基线生成的对应仓位；任何 `bear` 日两组均为零仓；实验组不存在负仓位或由零仓恢复的对流交易。

- [ ] **Step 6: 写中文实验报告**

报告列出两组成本后收益、最大回撤、平均仓位、换手、方向指标和对流四档条件统计。若没有改善，明确写“对流门控无样本外价值”，不保留为默认策略。

- [ ] **Step 7: 浏览器验收**

在 `http://192.168.1.145:8765/` 启动任务并检查桌面宽度下配置、运行进度、双模型结果和对流诊断无重叠、无错误。

- [ ] **Step 8: 提交报告并确认工作树干净**

```bash
git add docs/path-efficiency-convection-gate-report.md
git commit -m "docs: report convection gate ablation"
git status --short
```

Expected: 工作树为空。

