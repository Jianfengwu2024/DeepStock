const state = {
  config: null,
  devices: [],
  jobs: [],
  selectedId: null,
  symbols: [],
  dialogSelected: new Set(),
  polling: false,
};

const numberFields = new Set([
  "epochs", "batch_size", "learning_rate", "forward_sessions",
  "minimum_train_sessions", "tuning_sessions", "test_sessions", "step_sessions",
  "max_folds", "profile_bins", "sequence_length", "sequence_stride",
  "minimum_sequence_length", "d_model", "motif_dim", "heads", "layers",
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
  payload.allow_short = form.elements.allow_short.checked;
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
    if (!state.selectedId && state.jobs.length) state.selectedId = state.jobs[0].id;
    renderJobList();
    if (state.selectedId) {
      const job = await api(`/api/jobs/${encodeURIComponent(state.selectedId)}`);
      renderJob(job);
    }
    renderGlobalStatus();
  } catch (exception) {
    document.querySelector("#job-subtitle").textContent = exception.message;
  } finally {
    state.polling = false;
  }
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
  document.querySelector("#run-meta").textContent = `${job.request.start} → ${job.request.end} · ${job.request.symbols.join(", ")} · ${job.request.device}`;
  renderJobGpuMemory(job);
  const active = ["queued", "running"].includes(job.status);
  document.querySelector("#cancel-button").classList.toggle("hidden", !active);
  document.querySelector("#progress-band").classList.toggle("hidden", !active);
  document.querySelector("#progress-label").textContent = job.status === "queued" ? "等待 GPU 任务" : "Walk-forward 运行中";
  document.querySelector("#elapsed-time").textContent = job.started_at ? elapsed(job.started_at, job.finished_at) : "";
  document.querySelector("#job-log").textContent = (job.log || []).join("\n") || "暂无日志";
  renderInsight(job.summary);
  renderMetrics(job.summary);
  renderFolds(job.summary?.folds || []);
  renderArtifacts(job);
  icons();
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

function renderInsight(summary) {
  const panel = document.querySelector("#result-insight");
  if (!summary) {
    panel.classList.add("hidden");
    return;
  }
  const accuracy = Number(summary.accuracy);
  const brier = Number(summary.brier_score);
  const logLoss = Number(summary.log_loss);
  let level = "weak";
  let title = "只有微弱信号，暂不足以用于实盘";
  let detail = `五分类随机命中率是 20%。本次命中率 ${formatPercent(accuracy)}，虽略高于随机，但概率误差仍较大。`;
  if (brier >= 0.79 || logLoss >= 1.59 || accuracy <= 0.21) {
    level = "poor";
    title = "模型输出接近随机猜测";
    detail = `五分类随机基准为：命中率 20%、Brier 约 0.80、Log Loss 约 1.61。本次结果与随机基准非常接近，当前模型尚未学到稳定规律。`;
  } else if (accuracy >= 0.28 && brier <= 0.74 && logLoss <= 1.50) {
    level = "promising";
    title = "模型出现初步区分能力，仍需扩大样本验证";
    detail = `本次三个分类指标均明显优于随机基准，但仍需更多股票和更多滚动窗口确认稳定性。`;
  }
  const backtest = summary.backtest || {};
  const inactive = Number.isFinite(Number(backtest.average_abs_position))
    ? Number(backtest.average_abs_position) < 0.02
    : Math.abs(Number(backtest.total_return || 0)) < 0.001 && Math.abs(Number(backtest.max_drawdown || 0)) < 0.002;
  if (inactive) detail += " 蓝色净值几乎不动，表示不确定性过滤后仓位接近零。";
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
  const values = [
    ["样本外信号", formatInteger(summary.oos_predictions), "未参与训练的预测数量"],
    ["五状态命中率", formatPercent(summary.accuracy), "随机约 20%，越高越好"],
    ["概率误差 Brier", formatNumber(summary.brier_score, 4), "0 最好，随机约 0.80"],
    ["概率损失 Log Loss", formatNumber(summary.log_loss, 4), "0 最好，随机约 1.61"],
    ["模型策略收益", formatPercent(summary.backtest?.total_return), "已扣除设定的交易成本"],
    ["最大回撤", formatPercent(Math.abs(Number(summary.backtest?.max_drawdown))), "历史峰值后的最大跌幅"],
  ];
  grid.innerHTML = values.map(([label, value, help]) => `<div class="metric-card"><span>${label}</span><strong>${value}</strong><small>${help}</small></div>`).join("");
}

function renderFolds(folds) {
  const table = document.querySelector("#fold-table");
  table.innerHTML = folds.map(fold => `<tr><td>${fold.fold + 1}</td><td>${fold.tuning_start}</td><td>${fold.test_start} → ${fold.test_end}</td><td>${fold.selected_epochs}</td><td>${fold.pretest_sequences}</td><td>${fold.outer_test_predictions}</td><td>${formatNumber(fold.tuning_best_validation_loss, 4)}</td></tr>`).join("");
}

function renderArtifacts(job) {
  const cache = `?v=${encodeURIComponent(job.finished_at || Date.now())}`;
  setImage("equity-image", job.artifacts["equity_drawdown.png"], cache);
  setImage("residual-image", job.artifacts["residuals.png"], cache);
  setImage("calibration-image", job.artifacts["calibration_acf.png"], cache);
  const row = document.querySelector("#download-row");
  const names = { "evaluation.json": "评估数据", "predictions.json": "OOS 预测", "run-summary.json": "运行摘要" };
  row.innerHTML = Object.entries(names).filter(([name]) => job.artifacts[name]).map(([name, label]) => `<a href="${job.artifacts[name]}" download>${label}</a>`).join("");
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
  document.querySelector("#refresh-button").addEventListener("click", refreshAll);
  document.querySelector("#choose-symbols").addEventListener("click", openSymbolDialog);
  document.querySelector("#apply-symbols").addEventListener("click", applySymbols);
  document.querySelector("#symbol-search").addEventListener("input", event => renderSymbolOptions(event.target.value));
  document.querySelector("#select-all").addEventListener("click", () => { document.querySelectorAll("#symbol-list input").forEach(item => { item.checked = true; state.dialogSelected.add(item.value); }); updateSelectedCount(); });
  document.querySelector("#clear-symbols").addEventListener("click", () => { state.dialogSelected.clear(); document.querySelectorAll("#symbol-list input").forEach(item => item.checked = false); updateSelectedCount(); });
  document.querySelector("#cancel-button").addEventListener("click", async () => { if (state.selectedId) { await api(`/api/jobs/${state.selectedId}/cancel`, { method: "POST" }); refreshAll(); } });
  document.querySelector("#open-config").addEventListener("click", () => document.querySelector("#config-panel").classList.add("open"));
  document.querySelector("#close-config").addEventListener("click", () => document.querySelector("#config-panel").classList.remove("open"));
  document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(item => item.classList.toggle("active", item === tab));
    document.querySelectorAll(".tab-panel").forEach(panel => panel.classList.toggle("hidden", panel.dataset.panel !== tab.dataset.tab));
  }));
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
