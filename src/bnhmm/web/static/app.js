const state = {
  config: null,
  devices: [],
  jobs: [],
  selectedId: null,
  symbols: [],
  dialogSelected: new Set(),
  resultScopeByJob: new Map(),
  symbolReportCache: new Map(),
  polling: false,
};

const numberFields = new Set([
  "epochs", "pretrain_epochs", "pretrain_mask_probability", "batch_size", "learning_rate", "kl_weight", "ordinal_weight", "forward_sessions", "strong_threshold_sigma",
  "return_standardization_lookback", "return_standardization_min_observations", "minimum_return_scale_percent",
  "minimum_train_sessions", "tuning_sessions", "test_sessions", "step_sessions",
  "max_folds", "bin_width_bps", "max_profile_bins", "normalization_lookback_sessions", "velocity_bins", "velocity_clip", "velocity_smoothing", "sequence_length", "sequence_stride",
  "minimum_sequence_length", "d_model", "latent_states", "transition_rank", "motif_dim", "heads", "layers",
  "feedforward_dim", "dropout", "mc_samples", "uncertainty_aversion",
  "transaction_cost_bps", "seed",
]);

const statusText = {
  queued: "排队中", running: "运行中", completed: "已完成", failed: "失败",
  cancelled: "已停止", interrupted: "已中断",
};

function icons() {
  if (window.lucide) window.lucide.createIcons();
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = Array.isArray(payload.detail)
      ? payload.detail.map(item => item.msg).join("；")
      : payload.detail || `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return response.json();
}

function fillForm(defaults) {
  const form = document.querySelector("#experiment-form");
  for (const [name, value] of Object.entries(defaults)) {
    const field = form.elements[name];
    if (!field) continue;
    if (name === "symbols") field.value = value.join(", ");
    else if (field.type === "checkbox") field.checked = Boolean(value);
    else field.value = value ?? "";
  }
  syncModelConstraints();
}

function syncModelConstraints() {
  const velocityField = document.querySelector("[name=velocity_bins]");
  const latentField = document.querySelector("[name=latent_states]");
  const transitionRankField = document.querySelector("[name=transition_rank]");
  const sequenceLengthField = document.querySelector("[name=sequence_length]");
  const strongThresholdField = document.querySelector("[name=strong_threshold_sigma]");
  const standardizationLookbackField = document.querySelector("[name=return_standardization_lookback]");
  const standardizationMinimumField = document.querySelector("[name=return_standardization_min_observations]");
  const widthField = document.querySelector("[name=d_model]");
  const headsField = document.querySelector("[name=heads]");
  const velocityBins = Number(velocityField.value || 3);
  const minimum = 4 + velocityBins ** 2;
  latentField.min = String(minimum);
  if (Number(latentField.value) < minimum) latentField.value = String(minimum);

  const latentStates = Number(latentField.value || minimum);
  transitionRankField.max = String(latentStates);
  if (Number(transitionRankField.value) > latentStates) {
    transitionRankField.value = String(latentStates);
  }

  const channelCount = document.querySelector("#profile-channel-count");
  if (channelCount) {
    channelCount.textContent = `4 + ${velocityBins}² = ${minimum} 通道`;
  }

  const memoryLength = Number(sequenceLengthField.value || 5);
  const memorySpec = document.querySelector("#memory-length-spec");
  if (memorySpec) memorySpec.textContent = `最近 ${memoryLength} 个交易日`;

  standardizationMinimumField.max = standardizationLookbackField.value;
  if (Number(standardizationMinimumField.value) > Number(standardizationLookbackField.value)) {
    standardizationMinimumField.value = standardizationLookbackField.value;
  }

  const strongThreshold = Number(strongThresholdField.value || 0.75);
  const strongSpec = document.querySelector("#strong-threshold-spec");
  if (strongSpec) {
    strongSpec.textContent = `-${strongThreshold.toFixed(2)}σ / 0 / +${strongThreshold.toFixed(2)}σ`;
  }

  const width = Number(widthField.value);
  const heads = Number(headsField.value);
  headsField.setCustomValidity(
    width && heads && width % heads !== 0
      ? "状态编码宽度必须能被注意力头数整除"
      : "",
  );
}

function setupDevices(devices) {
  state.devices = devices;
  const select = document.querySelector("#device-select");
  select.replaceChildren(...devices.map(device => {
    const option = document.createElement("option");
    option.value = device.id;
    option.textContent = device.id === "cpu" ? "CPU" : `${device.id} · ${device.name}`;
    option.dataset.device = JSON.stringify(device);
    return option;
  }));
  if (devices.some(item => item.id === "cuda:0")) select.value = "cuda:0";
  updateDeviceNote();
  renderGpuMonitor();
}

function updateDeviceNote() {
  const select = document.querySelector("#device-select");
  const note = document.querySelector("#device-note");
  const device = state.devices.find(item => item.id === select.value);
  if (!device) return;
  note.textContent = device.id === "cpu" ? "" : `${device.memory_used_mib} / ${device.memory_total_mib} MiB · ${device.temperature_c}°C`;
}

function updateDevices(devices) {
  state.devices = devices;
  for (const option of document.querySelectorAll("#device-select option")) {
    const device = devices.find(item => item.id === option.value);
    if (device) option.dataset.device = JSON.stringify(device);
  }
  updateDeviceNote();
  renderGpuMonitor();
}

function renderGpuMonitor() {
  const monitor = document.querySelector("#gpu-monitor");
  const gpus = state.devices.filter(item => item.id.startsWith("cuda:"));
  monitor.innerHTML = gpus.map(gpu => {
    const percent = Number.isFinite(Number(gpu.memory_percent))
      ? Number(gpu.memory_percent)
      : Number(gpu.memory_used_mib) / Math.max(Number(gpu.memory_total_mib), 1) * 100;
    const level = percent >= 90 ? "critical" : percent >= 75 ? "high" : "";
    const used = (gpu.memory_used_mib / 1024).toFixed(1);
    const total = (gpu.memory_total_mib / 1024).toFixed(0);
    const index = gpu.id.split(":")[1];
    const free = Number.isFinite(Number(gpu.memory_free_mib))
      ? Number(gpu.memory_free_mib)
      : Number(gpu.memory_total_mib) - Number(gpu.memory_used_mib);
    return `<div class="gpu-chip ${level}" title="${escapeHtml(gpu.name)} · 剩余 ${(free / 1024).toFixed(1)} GB"><div class="gpu-chip-head"><span>GPU ${index}</span><strong>${used} / ${total} GB</strong></div><div class="gpu-memory-track"><span style="width:${Math.min(percent, 100)}%"></span></div></div>`;
  }).join("");
}

function formPayload() {
  const form = document.querySelector("#experiment-form");
  const data = new FormData(form);
  const payload = {};
  for (const [name, value] of data.entries()) {
    if (name === "symbols") {
      payload.symbols = value.split(/[\s,;]+/).map(item => item.trim()).filter(Boolean);
    } else if (numberFields.has(name)) {
      payload[name] = value === "" && name === "max_folds" ? null : Number(value);
    } else {
      payload[name] = value;
    }
  }
  payload.deterministic_training = form.elements.deterministic_training.checked;
  return payload;
}

async function submitExperiment(event) {
  event.preventDefault();
  const error = document.querySelector("#form-error");
  const button = document.querySelector("#run-button");
  error.textContent = "";
  button.disabled = true;
  try {
    const job = await api("/api/jobs", { method: "POST", body: JSON.stringify(formPayload()) });
    state.selectedId = job.id;
    await refreshAll();
    document.querySelector("#config-panel").classList.remove("open");
  } catch (exception) {
    error.textContent = exception.message;
  } finally {
    button.disabled = false;
  }
}

async function refreshAll() {
  if (state.polling) return;
  state.polling = true;
  try {
    const [jobs, gpuPayload] = await Promise.all([
      api("/api/jobs"),
      api("/api/gpus").catch(() => api("/api/config").then(config => ({ devices: config.devices }))),
    ]);
    state.jobs = jobs;
    updateDevices(gpuPayload.devices);
    if (!state.jobs.some(job => job.id === state.selectedId)) {
      state.selectedId = state.jobs[0]?.id || null;
    }
    renderJobList();
    if (state.selectedId) {
      const job = await api(`/api/jobs/${encodeURIComponent(state.selectedId)}`);
      renderJob(job);
    } else {
      renderEmptyState();
    }
    renderGlobalStatus();
  } catch (exception) {
    document.querySelector("#job-subtitle").textContent = exception.message;
  } finally {
    state.polling = false;
  }
}

function renderEmptyState() {
  document.querySelector("#empty-state").classList.remove("hidden");
  document.querySelector("#job-view").classList.add("hidden");
  document.querySelector("#job-subtitle").textContent = "开始新的实验";
}

function renderJobList() {
  const list = document.querySelector("#job-list");
  document.querySelector("#run-count").textContent = state.jobs.length;
  list.replaceChildren(...state.jobs.map(job => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `job-item${job.id === state.selectedId ? " active" : ""}`;
    button.innerHTML = `<div class="job-item-head"><strong>${escapeHtml(job.request.name)}</strong><span class="status-badge ${job.status}">${statusText[job.status]}</span></div><div class="job-item-meta">${formatDate(job.created_at)} · ${job.request.symbols.length} 股</div>`;
    button.addEventListener("click", () => { state.selectedId = job.id; refreshAll(); });
    return button;
  }));
}

function renderGlobalStatus() {
  const active = state.jobs.filter(job => ["queued", "running"].includes(job.status)).length;
  const badge = document.querySelector("#global-status");
  badge.className = `status-badge ${active ? "running" : "neutral"}`;
  badge.textContent = active ? `${active} 个任务` : "空闲";
}

function renderJob(job) {
  document.querySelector("#empty-state").classList.add("hidden");
  document.querySelector("#job-view").classList.remove("hidden");
  document.querySelector("#run-title").textContent = job.request.name;
  document.querySelector("#job-subtitle").textContent = job.id;
  const badge = document.querySelector("#run-status");
  badge.className = `status-badge ${job.status}`;
  badge.textContent = statusText[job.status];
  const stateMode = job.request.state_timeframe === "mixed"
    ? "小时 + 日状态"
    : "日状态（1分钟线生成）";
  const profileMode = job.request.profile_protocol === "phase-space-joint-return-v6"
    ? `${job.request.bin_width_bps} BP 联合速度相空间 · ${job.request.normalization_lookback_sessions}日归一化`
    : job.request.profile_protocol === "phase-space-joint-v5"
      ? `${job.request.bin_width_bps} BP 旧联合速度相空间`
    : job.request.profile_protocol === "phase-space-gated-v4"
      ? `${job.request.bin_width_bps} BP 旧边际速度相空间`
    : job.request.profile_protocol === "phase-space-log-v3"
      ? `${job.request.bin_width_bps} BP 旧22通道相空间`
      : job.request.profile_protocol === "legacy-fixed-bins"
        ? "旧固定柱协议"
        : `${job.request.bin_width_bps} BP 可变日态（旧）`;
  const modelScope = job.request.training_scope === "per-symbol"
    ? "逐标的独立 BNHMM"
    : "多标的共享模型（旧）";
  const latentStates = job.request.latent_states ?? 10;
  const featureMode = ({
    "gated-full": "价格主干 + 门控成交量/联合速度",
    "price-only": "仅价格",
    "price-volume": "价格 + 成交量",
    "price-velocity": "价格 + 联合速度",
    "legacy-entangled": "旧混合特征",
  })[job.request.feature_mode] || "旧混合特征";
  const outputMode = job.request.profile_protocol === "phase-space-joint-return-v6"
    ? ((job.summary?.class_support?.length ?? 4) === 3
      ? "连续收益后验 → 空/平/多（旧）"
      : "连续收益后验 → 强空/弱空/弱多/强多")
    : "旧三分类证据头";
  const outputHead = job.summary?.model_config?.output_hidden_states
    ? `${job.summary.model_config.output_hidden_states} 隐藏节点（旧）`
    : "动态先验 + 收益/序数双头";
  const thresholds = job.summary?.regime_protocol?.thresholds || [];
  const thresholdText = thresholds.length === 2
    ? `平状态 ${formatPercent(Math.abs(Number(thresholds[1])))}`
    : job.request.strong_threshold_sigma == null
      ? `强弱分界 ±${Number(job.request.strong_threshold_percent ?? 1.5).toFixed(2)}%（旧）`
      : `强弱分界 ±${Number(job.request.strong_threshold_sigma).toFixed(2)}σ · ${job.request.return_standardization_lookback}日标准化`;
  document.querySelector("#run-meta").textContent = `${job.request.start} → ${job.request.end} · ${job.request.symbols.join(", ")} · ${modelScope} · ${featureMode} · ${latentStates} 隐状态 → ${outputHead} · ${outputMode} · ${thresholdText} · ${stateMode} · ${profileMode} · ${job.request.device}`;
  renderJobGpuMemory(job);
  const active = ["queued", "running"].includes(job.status);
  document.querySelector("#cancel-button").classList.toggle("hidden", !active);
  document.querySelector("#delete-button").classList.toggle("hidden", active);
  document.querySelector("#progress-band").classList.toggle("hidden", !active);
  document.querySelector("#progress-label").textContent = job.status === "queued" ? "等待 GPU 任务" : "Walk-forward 运行中";
  document.querySelector("#elapsed-time").textContent = job.started_at ? elapsed(job.started_at, job.finished_at) : "";
  document.querySelector("#job-log").textContent = (job.log || []).join("\n") || "暂无日志";
  renderResultScope(job);
  renderSelectedResult(job);
  icons();
}

function resultScope(job) {
  const requested = state.resultScopeByJob.get(job.id) || "portfolio";
  return requested === "portfolio" || job.symbol_artifacts?.[requested]
    ? requested
    : "portfolio";
}

function renderResultScope(job) {
  const section = document.querySelector("#result-scope");
  const symbols = Object.keys(job.symbol_artifacts || {});
  const independent = job.request.training_scope === "per-symbol" && symbols.length > 0;
  section.classList.toggle("hidden", !independent);
  if (!independent) return;

  const selected = resultScope(job);
  document.querySelector("#result-scope-note").textContent = selected === "portfolio"
    ? "当前显示各独立模型的组合汇总"
    : `当前只显示 ${selected} 的独立模型结果`;
  document.querySelector("#result-scope-badge").textContent = selected === "portfolio"
    ? "组合汇总"
    : "单股独立模型";

  const choices = [
    { value: "portfolio", label: "组合总览" },
    ...symbols.map(symbol => ({ value: symbol, label: symbol })),
  ];
  const tabs = document.querySelector("#scope-tabs");
  tabs.replaceChildren(...choices.map(choice => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `scope-tab${choice.value === selected ? " active" : ""}`;
    button.textContent = choice.label;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(choice.value === selected));
    button.addEventListener("click", () => {
      state.resultScopeByJob.set(job.id, choice.value);
      renderResultScope(job);
      renderSelectedResult(job);
    });
    return button;
  }));
}

function renderSelectedResult(job) {
  const scope = resultScope(job);
  if (scope === "portfolio") {
    document.querySelector("#insight-label").textContent = job.summary?.model_scope === "per-symbol"
      ? "独立模型组合结论"
      : "本次结论";
    renderInsight(job.summary);
    renderMetrics(job.summary);
    renderSymbolMetrics(job.summary?.per_symbol || []);
    renderFolds(job.summary?.folds || []);
    renderArtifacts(job, job.artifacts || {});
    return;
  }

  const row = job.summary?.per_symbol?.find(item => item.symbol === scope) || {};
  const fallback = {
    ...row,
    symbol: scope,
    model_scope: "single-symbol",
    regime_protocol: job.summary?.regime_protocol,
  };
  const key = `${job.id}:${scope}`;
  const cached = state.symbolReportCache.get(key);
  const summary = cached ? {
    ...cached,
    symbol: scope,
    model_scope: "single-symbol",
    regime_protocol: job.summary?.regime_protocol,
    oos_predictions: cached.observation_count,
  } : fallback;
  renderSymbolResult(job, scope, summary);

  const evaluationUrl = job.symbol_artifacts?.[scope]?.["evaluation.json"];
  if (!evaluationUrl || state.symbolReportCache.has(key)) return;
  state.symbolReportCache.set(key, null);
  const activeButton = document.querySelector(`.scope-tab.active`);
  activeButton?.classList.add("loading");
  api(evaluationUrl).then(report => {
    state.symbolReportCache.set(key, report);
    if (state.selectedId !== job.id || resultScope(job) !== scope) return;
    renderSymbolResult(job, scope, {
      ...report,
      symbol: scope,
      model_scope: "single-symbol",
      regime_protocol: job.summary?.regime_protocol,
      oos_predictions: report.observation_count,
    });
    renderResultScope(job);
  }).catch(() => {
    state.symbolReportCache.delete(key);
    activeButton?.classList.remove("loading");
  });
}

function renderSymbolResult(job, symbol, summary) {
  document.querySelector("#insight-label").textContent = `${symbol} 单股结论`;
  renderInsight(summary);
  renderMetrics(summary);
  const row = job.summary?.per_symbol?.find(item => item.symbol === symbol);
  renderSymbolMetrics(row ? [row] : []);
  renderFolds((job.summary?.folds || []).filter(fold => fold.symbol === symbol));
  renderArtifacts(job, job.symbol_artifacts?.[symbol] || {});
}

function renderJobGpuMemory(job) {
  const element = document.querySelector("#run-gpu-memory");
  const isGpu = String(job.request.device).startsWith("cuda:");
  const trackingAvailable = Object.prototype.hasOwnProperty.call(job, "peak_gpu_memory_mib");
  if (isGpu && job.status === "running" && !trackingAvailable) {
    element.classList.remove("hidden");
    element.textContent = "本任务显存：该任务启动早于监控启用，下一任务开始记录";
    return;
  }
  const hasSample = Number(job.peak_gpu_memory_mib) > 0;
  if (!isGpu || (!hasSample && job.status !== "running")) {
    element.classList.add("hidden");
    return;
  }
  element.classList.remove("hidden");
  element.textContent = job.status === "running"
    ? `本任务显存：当前 ${formatMemory(job.current_gpu_memory_mib)} · 峰值 ${formatMemory(job.peak_gpu_memory_mib)}`
    : `本任务峰值显存：${formatMemory(job.peak_gpu_memory_mib)}`;
}

function classificationBaseline(summary) {
  const reportedCount = Array.isArray(summary.class_support) ? summary.class_support.length : 0;
  const protocolCount = Array.isArray(summary.regime_protocol?.states)
    ? summary.regime_protocol.states.length
    : 0;
  const count = reportedCount >= 2 ? reportedCount : (protocolCount >= 2 ? protocolCount : 5);
  const label = ({ 3: "三", 4: "四", 5: "五" })[count] || String(count);
  return {
    count,
    label,
    accuracy: 1 / count,
    brier: 1 - 1 / count,
    logLoss: Math.log(count),
  };
}

function renderInsight(summary) {
  const panel = document.querySelector("#result-insight");
  if (!summary) {
    panel.classList.add("hidden");
    return;
  }
  const accuracy = Number(summary.accuracy);
  const brier = Number(summary.brier_score);
  const logLoss = Number(summary.log_loss);
  const baseline = classificationBaseline(summary);
  const majority = Number(summary.majority_baseline);
  const balanced = Number(summary.balanced_accuracy);
  const comparisons = summary.baselines || {};
  const priorBrier = Number(comparisons.class_prior_brier);
  const priorLogLoss = Number(comparisons.class_prior_log_loss);
  const correlation = Number(summary.return_residual?.correlation);
  let level = "weak";
  let title = "只有微弱信号，暂不足以用于实盘";
  let detail = `命中率必须超过多数类基线，均衡命中率必须显著超过 ${formatPercent(baseline.accuracy)}，才能说明模型具有分类能力。`;
  if (accuracy <= majority || balanced <= baseline.accuracy + 0.01 || correlation <= 0 || logLoss >= priorLogLoss || brier >= priorBrier) {
    level = "poor";
    title = accuracy <= majority ? "模型未超过多数类基线" : "模型没有稳定的方向区分能力";
    detail = `本次命中率 ${formatPercent(accuracy)}，多数类基线 ${formatPercent(majority)}，均衡命中率 ${formatPercent(balanced)}，收益相关性 ${formatNumber(correlation, 4)}。较高的普通命中率可能只是类别不均衡造成的。`;
  } else if (
    accuracy >= majority + 0.03
    && balanced >= baseline.accuracy + 0.05
    && Number.isFinite(priorBrier) && brier <= priorBrier - 0.02
    && Number.isFinite(priorLogLoss) && logLoss <= priorLogLoss - 0.03
  ) {
    level = "promising";
    title = "模型出现初步区分能力，仍需扩大样本验证";
    detail = `命中率和均衡命中率均超过基线，概率误差也优于按真实类别比例预测的基线，但仍需更多标的和滚动窗口确认稳定性。`;
  }
  const gate = summary.validation_gate;
  if (gate && !gate.passed) {
    const labels = {
      beats_majority_accuracy: "命中率未超过多数类",
      balanced_accuracy_above_random: "均衡命中率优势不足",
      positive_log_information_gain: "Log Loss 未优于训练期先验",
      beats_prior_brier: "Brier 未优于训练期先验",
      beats_dynamic_prior_return_rmse: "收益误差未优于滚动均值",
      beats_zero_return_rmse: "收益误差未优于零收益预测",
      positive_return_correlation: "收益相关性非正",
      minimum_oos_observations: "样本外数量不足 200",
    };
    const failed = Object.entries(gate.checks || {})
      .filter(([, passed]) => !passed)
      .map(([name]) => labels[name] || name);
    level = "poor";
    title = "未通过样本外发布门槛";
    detail = `当前结果仅供研究，不能发布为可交易模型。未通过：${failed.join("、")}。`;
  }
  const backtest = summary.backtest || {};
  const inactive = Number.isFinite(Number(backtest.average_abs_position))
    ? Number(backtest.average_abs_position) < 0.02
    : Math.abs(Number(backtest.total_return || 0)) < 0.001 && Math.abs(Number(backtest.max_drawdown || 0)) < 0.002;
  if (inactive) detail += " 蓝色净值几乎不动，表示不确定性过滤后仓位接近零。";
  if (summary.model_scope === "per-symbol") detail += " 顶部指标是各标的独立模型的组合汇总，单股结论请查看滚动验证页。";
  if (summary.model_scope === "single-symbol") detail += ` 当前页面只统计 ${summary.symbol}，没有混入其他股票。`;
  panel.className = `result-insight ${level}`;
  document.querySelector("#insight-title").textContent = title;
  document.querySelector("#insight-detail").textContent = detail;
}

function renderMetrics(summary) {
  const grid = document.querySelector("#metrics-grid");
  if (!summary) {
    grid.replaceChildren();
    return;
  }
  const baseline = classificationBaseline(summary);
  const comparisons = summary.baselines || {};
  const values = [
    ["样本外信号", formatInteger(summary.oos_predictions), "未参与训练的预测数量"],
    [`${baseline.label}状态命中率`, formatPercent(summary.accuracy), `随机约 ${formatPercent(baseline.accuracy)}，越高越好`],
    ["均衡命中率", formatPercent(summary.balanced_accuracy), `${baseline.label}种状态分别计算召回率后取平均`],
    ["训练先验基线", formatPercent(summary.majority_baseline), "每折只用训练期多数状态，不读取样本外标签"],
    ["状态持续基线", comparisons.persistence_accuracy == null ? "—" : formatPercent(comparisons.persistence_accuracy), "直接沿用上一交易日状态"],
    ["状态反转基线", comparisons.reversal_accuracy == null ? "—" : formatPercent(comparisons.reversal_accuracy), "强空↔强多、弱空↔弱多"],
    ["概率误差 Brier", formatNumber(summary.brier_score, 4), `类别比例基线 ${formatNumber(comparisons.class_prior_brier, 4)}`],
    ["概率损失 Log Loss", formatNumber(summary.log_loss, 4), `类别比例基线 ${formatNumber(comparisons.class_prior_log_loss, 4)}`],
    ["对数信息增益", formatNumber(comparisons.log_information_gain, 4), "大于 0 才说明当前状态提供了超越历史先验的信息"],
    ["有序概率误差 RPS", formatNumber(summary.ranked_probability_score, 4), "越低越好，距离更远的错判惩罚更大"],
    ["收益相关性", formatNumber(summary.return_residual?.correlation, 4), "预测收益与实际收益的样本外相关性"],
    ["收益 RMSE", formatPercent(summary.return_residual?.rmse), `零收益 ${formatPercent(comparisons.zero_return_rmse)} · 滚动均值 ${formatPercent(comparisons.dynamic_prior_return_rmse)}`],
    ["发布验证", summary.validation_gate ? (summary.validation_gate.passed ? "通过" : "未通过") : "未评估", "必须同时通过分类、收益误差、相关性和样本量门槛"],
    ["模型策略收益", formatPercent(summary.backtest?.total_return), "已扣除设定的交易成本"],
    ["最大回撤", formatPercent(Math.abs(Number(summary.backtest?.max_drawdown))), "历史峰值后的最大跌幅"],
  ];
  grid.innerHTML = values.map(([label, value, help]) => `<div class="metric-card"><span>${label}</span><strong>${value}</strong><small>${help}</small></div>`).join("");
}

function renderFolds(folds) {
  const table = document.querySelector("#fold-table");
  table.innerHTML = folds.map(fold => `<tr><td>${escapeHtml(fold.symbol || "共享（旧）")}</td><td>${fold.fold + 1}</td><td>${fold.tuning_start}</td><td>${fold.test_start} → ${fold.test_end}</td><td>${fold.selected_epochs}</td><td>${fold.pretest_sequences}</td><td>${fold.outer_test_predictions}</td><td>${formatNumber(fold.tuning_best_validation_loss, 4)}</td></tr>`).join("");
}

function renderSymbolMetrics(rows) {
  const section = document.querySelector("#symbol-results");
  const table = document.querySelector("#symbol-metrics-table");
  section.classList.toggle("hidden", !rows.length);
  table.innerHTML = rows.map(row => `<tr><td>${escapeHtml(row.symbol)}</td><td>${formatInteger(row.oos_predictions)}</td><td>${formatPercent(row.accuracy)}</td><td>${formatPercent(row.balanced_accuracy)}</td><td>${formatPercent(row.majority_baseline)}</td><td>${formatNumber(row.brier_score, 4)}</td><td>${formatNumber(row.log_loss, 4)}</td><td>${formatPercent(row.backtest?.total_return)}</td><td>${formatPercent(Math.abs(Number(row.backtest?.max_drawdown)))}</td></tr>`).join("");
}

function renderArtifacts(job, artifacts) {
  const cache = `?v=${encodeURIComponent(job.finished_at || Date.now())}`;
  setImage("equity-image", artifacts["equity_drawdown.png"], cache);
  setImage("residual-image", artifacts["residuals.png"], cache);
  setImage("calibration-image", artifacts["calibration_acf.png"], cache);
  const row = document.querySelector("#download-row");
  const names = { "evaluation.json": "评估数据", "predictions.json": "OOS 预测", "run-summary.json": "运行摘要" };
  row.innerHTML = Object.entries(names).filter(([name]) => artifacts[name]).map(([name, label]) => `<a href="${artifacts[name]}" download>${label}</a>`).join("");
}

function setImage(id, source, cache) {
  const image = document.querySelector(`#${id}`);
  const wrapper = image.closest(".report-image-wrap");
  wrapper.classList.toggle("hidden", !source);
  if (source) image.src = source + cache;
  else image.removeAttribute("src");
}

async function openSymbolDialog() {
  const dialog = document.querySelector("#symbol-dialog");
  if (!state.symbols.length) {
    const path = document.querySelector("[name=database_path]").value;
    try {
      state.symbols = (await api(`/api/symbols?limit=200&database_path=${encodeURIComponent(path)}`)).symbols;
    } catch (exception) {
      document.querySelector("#form-error").textContent = exception.message;
      return;
    }
  }
  state.dialogSelected = selectedSymbols();
  renderSymbolOptions();
  dialog.showModal();
  icons();
}

function selectedSymbols() {
  return new Set(document.querySelector("[name=symbols]").value.split(/[\s,;]+/).filter(Boolean));
}

function renderSymbolOptions(filter = "") {
  const values = state.symbols.filter(symbol => symbol.includes(filter.trim()));
  const list = document.querySelector("#symbol-list");
  list.replaceChildren(...values.map(symbol => {
    const label = document.createElement("label");
    label.className = "symbol-option";
    label.innerHTML = `<input type="checkbox" value="${escapeHtml(symbol)}" ${state.dialogSelected.has(symbol) ? "checked" : ""}><span>${escapeHtml(symbol)}</span>`;
    label.querySelector("input").addEventListener("change", event => {
      if (event.target.checked) state.dialogSelected.add(symbol);
      else state.dialogSelected.delete(symbol);
      updateSelectedCount();
    });
    return label;
  }));
  updateSelectedCount();
}

function updateSelectedCount() {
  const count = state.dialogSelected.size;
  document.querySelector("#selected-count").textContent = `已选 ${count}`;
}

function applySymbols() {
  const values = [...state.dialogSelected];
  document.querySelector("[name=symbols]").value = values.join(", ");
  document.querySelector("#symbol-dialog").close();
}

function setupEvents() {
  document.querySelector("#experiment-form").addEventListener("submit", submitExperiment);
  document.querySelector("#device-select").addEventListener("change", updateDeviceNote);
  for (const name of ["velocity_bins", "latent_states", "transition_rank", "sequence_length", "strong_threshold_sigma", "return_standardization_lookback", "return_standardization_min_observations", "d_model", "heads"]) {
    document.querySelector(`[name=${name}]`).addEventListener("input", syncModelConstraints);
  }
  document.querySelector("#refresh-button").addEventListener("click", refreshAll);
  document.querySelector("#choose-symbols").addEventListener("click", openSymbolDialog);
  document.querySelector("#apply-symbols").addEventListener("click", applySymbols);
  document.querySelector("#symbol-search").addEventListener("input", event => renderSymbolOptions(event.target.value));
  document.querySelector("#select-all").addEventListener("click", () => { document.querySelectorAll("#symbol-list input").forEach(item => { item.checked = true; state.dialogSelected.add(item.value); }); updateSelectedCount(); });
  document.querySelector("#clear-symbols").addEventListener("click", () => { state.dialogSelected.clear(); document.querySelectorAll("#symbol-list input").forEach(item => item.checked = false); updateSelectedCount(); });
  document.querySelector("#cancel-button").addEventListener("click", async () => { if (state.selectedId) { await api(`/api/jobs/${state.selectedId}/cancel`, { method: "POST" }); refreshAll(); } });
  document.querySelector("#delete-button").addEventListener("click", deleteSelectedJob);
  document.querySelector("#open-config").addEventListener("click", () => document.querySelector("#config-panel").classList.add("open"));
  document.querySelector("#close-config").addEventListener("click", () => document.querySelector("#config-panel").classList.remove("open"));
  document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(item => item.classList.toggle("active", item === tab));
    document.querySelectorAll(".tab-panel").forEach(panel => panel.classList.toggle("hidden", panel.dataset.panel !== tab.dataset.tab));
  }));
}

async function deleteSelectedJob() {
  if (!state.selectedId) return;
  const job = state.jobs.find(item => item.id === state.selectedId);
  const name = job?.request?.name || state.selectedId;
  if (!window.confirm(`确定删除“${name}”及其全部检查点和报告吗？此操作不可撤销。`)) return;
  const identifier = state.selectedId;
  const button = document.querySelector("#delete-button");
  button.disabled = true;
  try {
    await api(`/api/jobs/${encodeURIComponent(identifier)}`, { method: "DELETE" });
    state.jobs = state.jobs.filter(item => item.id !== identifier);
    state.selectedId = state.jobs[0]?.id || null;
    await refreshAll();
  } catch (exception) {
    window.alert(`删除失败：${exception.message}`);
  } finally {
    button.disabled = false;
  }
}

function formatDate(value) { return new Date(value).toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }); }
function formatNumber(value, digits = 2) { return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—"; }
function formatInteger(value) { return Number.isFinite(Number(value)) ? Number(value).toLocaleString("zh-CN") : "—"; }
function formatPercent(value) { return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "—"; }
function formatMemory(value) { const mib = Number(value || 0); return mib >= 1024 ? `${(mib / 1024).toFixed(2)} GB` : `${Math.round(mib)} MiB`; }
function elapsed(start, end) { const seconds = Math.max(0, (new Date(end || Date.now()) - new Date(start)) / 1000); return seconds < 60 ? `${Math.floor(seconds)} 秒` : `${Math.floor(seconds / 60)} 分 ${Math.floor(seconds % 60)} 秒`; }
function escapeHtml(value) { return String(value).replace(/[&<>"]/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[character])); }

async function initialize() {
  setupEvents();
  state.config = await api("/api/config");
  fillForm(state.config.defaults);
  setupDevices(state.config.devices);
  await refreshAll();
  icons();
  setInterval(refreshAll, 2500);
}

initialize().catch(exception => {
  document.querySelector("#job-subtitle").textContent = exception.message;
});
