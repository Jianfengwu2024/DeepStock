const state = {
  config: null,
  jobs: [],
  selectedId: null,
  selected: new Set(),
  editing: false,
  page: "data",
  pools: [],
  quantAiPools: [],
  quantWatchlist: null,
  selectedPool: null,
  verificationToken: null,
  verificationUser: null,
  deepstockAuthorized: false,
  permissionStatus: "unknown",
  boundaryReady: false,
  pendingDeleteId: null,
};
const pageNames = new Set(["data", "strategy", "training", "pools", "ai", "verification"]);
let symbolSearchTimer = null;

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
  });
  if (!response.ok) {
    const body = await response.text();
    try {
      const parsed = JSON.parse(body);
      throw new Error(parsed.detail || parsed.message || ("HTTP " + response.status));
    } catch (error) {
      if (error instanceof SyntaxError) throw new Error(body || ("HTTP " + response.status));
      throw error;
    }
  }
  return response.status === 204 ? null : response.json();
}

function strategy() {
  return $('input[name="strategy"]:checked').value;
}

function isFk() {
  return strategy() === "boundary-flow-fk";
}

function namedFormControls(form, name) {
  return Array.from(form.elements).filter((item) => item.name === name);
}

function preferredFormControl(form, name) {
  return namedFormControls(form, name).find((item) => !item.disabled);
}

function fillForm(values) {
  const form = $("#experiment-form");
  Object.entries(values).forEach(([name, value]) => {
    const controls = namedFormControls(form, name);
    if (!controls.length || value == null) return;
    if (controls[0].type === "radio") {
      controls.forEach((item) => { item.checked = item.value === String(value); });
    } else if (controls[0].type === "checkbox") {
      controls[0].checked = Boolean(value);
    } else if (name === "symbols") {
      controls[0].value = Array.isArray(value) ? value.join(",") : value;
    } else {
      controls.forEach((item) => { item.value = String(value); });
    }
  });
}

function syncStrategyControlAvailability() {
  const form = $("#experiment-form");
  form.querySelectorAll("[name]").forEach((element) => {
    if (element.name === "strategy") return;
    const hiddenSection = element.closest(".hidden");
    element.disabled = Boolean(hiddenSection && form.contains(hiddenSection));
  });
}

function nextName(kind) {
  const prefix = kind === "bnhmm" ? "BNHMM" : "FK-Bayesian";
  const pattern = new RegExp("^" + prefix + "-(\\d+)$", "i");
  const last = state.jobs.reduce((maximum, job) => {
    const match = job.request.name.match(pattern);
    return match ? Math.max(maximum, Number(match[1])) : maximum;
  }, 0);
  return prefix + "-" + String(last + 1).padStart(3, "0");
}

function syncStrategy() {
  const kind = strategy();
  const currentName = $('[name="name"]').value;
  fillForm(state.config.strategy_defaults[kind]);
  $('input[name="strategy"][value="' + kind + '"]').checked = true;
  $('[name="name"]').value = currentName && !/^(BNHMM|FK-Bayesian)-\d+$/i.test(currentName)
    ? currentName : nextName(kind);
  $("#bnhmm-fields").classList.toggle("hidden", kind !== "bnhmm");
  $("#boundary-fields").classList.toggle("hidden", kind !== "boundary-flow-fk");
  $("#dataset-field").classList.remove("hidden");
  syncStrategyControlAvailability();
  applyDataset();
}

function renderDatasets() {
  const select = $("#dataset-select");
  const previous = select.value;
  const datasets = (state.config.market_datasets || []).filter((item) =>
    ["ready", "partial"].includes(item.status) && Number(item.row_counts?.["1m"] || 0) > 0);
  select.innerHTML = '<option value="">选择已就绪的本地行情数据集</option>' + datasets.map((item) =>
    '<option value="' + escapeHtml(item.dataset_id) + '">' + escapeHtml(item.name) + " · " + item.symbols.length + " 只 · " + item.status + "</option>"
  ).join("");
  const selected = datasets.find((item) => item.dataset_id === previous) ||
    datasets.find((item) => item.dataset_id === state.config.defaults.market_dataset_id) || datasets[0];
  if (selected) select.value = selected.dataset_id;
  select.onchange = applyDataset;
  if (selected && selected.dataset_id !== previous) applyDataset();
  const warning = $("#strategy-data-warning");
  warning.classList.toggle("hidden", Boolean(selected));
  $("#strategy-data-warning-text").textContent = selected ? "" : "尚无已就绪的本地分钟数据集。请先在数据服务页从股票池创建数据集并完成历史回补。";
  $("#run-button").disabled = !selected || (isFk() && !state.boundaryReady);
  if (!selected) {
    $('[name="database_path"]').value = "";
    $("#dataset-summary").textContent = "";
  }
}

function applyDataset() {
  const dataset = (state.config.market_datasets || []).find((item) => item.dataset_id === $("#dataset-select").value);
  $('[name="database_path"]').readOnly = true;
  if (!dataset) {
    $('[name="database_path"]').value = "";
    return;
  }
  const start = dataset.start_date;
  const endDay = dataset.last_refresh_date || dataset.start_date;
  const end = new Date(Date.parse(endDay + "T00:00:00Z") + 86400000).toISOString().slice(0, 10);
  $('[name="database_path"]').value = dataset.database_path;
  $('[name="database_path"]').readOnly = true;
  $('[name="symbols"]').value = dataset.symbols.join(",");
  $('[name="market_dataset_id"]').value = dataset.dataset_id;
  $("#dataset-summary").textContent = dataset.symbols.length + " 只股票 · " + dataset.intervals.join(", ") +
    " · " + start + " 至 " + endDay + " · " + (dataset.status === "ready" ? "已就绪" : "部分可用");
  if (!$('[name="start"]').value || $('[name="start"]').value < start) $('[name="start"]').value = start;
  if (!$('[name="end"]').value || $('[name="end"]').value > end || $('[name="end"]').value <= $('[name="start"]').value) $('[name="end"]').value = end;
  if ($('[name="start"]').value >= end) $('[name="start"]').value = start;
  checkBoundaryData();
}

function renderLocalMarketDatasets(datasets) {
  state.config.market_datasets = datasets;
  $("#market-dataset-count").textContent = String(datasets.length);
  const selectedPool = $("#market-dataset-pool").value;
  $("#market-dataset-pool").innerHTML = '<option value="">选择本地股票池</option>' + state.pools.map((pool) =>
    '<option value="' + escapeHtml(pool.id) + '">' + escapeHtml(pool.name) + " · " + pool.symbols.length + " 只</option>").join("");
  $("#market-dataset-pool").value = selectedPool;
  $("#market-datasets").innerHTML = datasets.length ? datasets.map((dataset) => {
    const summary = dataset.intervals.map((interval) => interval + ": " + Number(dataset.row_counts?.[interval] || 0).toLocaleString()).join(" · ");
    const error = dataset.error ? '<pre class="dataset-error">' + escapeHtml(dataset.error) + '</pre>' : "";
    const freshness = dataset.last_attempt_at ? " · 最近更新尝试 " + new Date(dataset.last_attempt_at).toLocaleString() : "";
    const actions = dataset.status === "queued" || dataset.status === "updating" ? '<span class="status-badge">更新中</span>' :
      dataset.source === "local-materialized" ? '<span class="status-badge">本地已物化</span>' :
      '<button class="secondary-button" type="button" data-refresh-dataset="' + escapeHtml(dataset.dataset_id) + '">立即增量更新</button>';
    const sourceLabel = dataset.purpose === "cloud-strategy" ? "云端策略临时数据 · 保留 7 天" + freshness : dataset.source === "local-materialized" ? "独立本地数据库 · 不自动更新" : "每日收盘自动增量" + freshness;
    const deleteId = "market:" + dataset.dataset_id;
    const deleteLabel = state.pendingDeleteId === deleteId ? "再次点击确认" : "删除";
    return '<article class="dataset-row"><div class="market-dataset-info"><strong>' + escapeHtml(dataset.name) + '</strong><span>' + dataset.symbols.length + " 只 · " + dataset.intervals.join(", ") + " · " + dataset.start_date + " 至 " + (dataset.last_refresh_date || "等待首次更新") + '</span><small>' + escapeHtml(summary) + " · " + dataset.status + '</small><small>' + escapeHtml(sourceLabel) + '</small>' + error + '</div><div class="button-row">' + actions + '<button class="danger-button" type="button" data-delete-dataset="' + escapeHtml(dataset.dataset_id) + '" ' + (["queued", "updating"].includes(dataset.status) ? "disabled" : "") + '>' + deleteLabel + '</button></div></article>';
  }).join("") : '<div class="empty-state">先从股票池创建数据集；行情、训练样本和模型文件都会保存在本机</div>';
  $$('[data-refresh-dataset]').forEach((button) => button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await api("/api/market-datasets/" + encodeURIComponent(button.dataset.refreshDataset) + "/refresh", {method: "POST"});
      await refreshMarketDatasets();
    } catch (error) { $("#market-dataset-form-status").textContent = error.message; }
    finally { button.disabled = false; }
  }));
  $$('[data-delete-dataset]').forEach((button) => button.addEventListener("click", async () => {
    const dataset = datasets.find((item) => item.dataset_id === button.dataset.deleteDataset);
    const deleteId = "market:" + button.dataset.deleteDataset;
    if (state.pendingDeleteId !== deleteId) {
      state.pendingDeleteId = deleteId;
      renderLocalMarketDatasets(datasets);
      return;
    }
    state.pendingDeleteId = null;
    try {
      await api("/api/market-datasets/" + encodeURIComponent(button.dataset.deleteDataset), {method: "DELETE"});
      await refreshMarketDatasets();
    } catch (error) { $("#market-dataset-form-status").textContent = error.message; }
  }));
  renderDatasets();
}

async function refreshMarketDatasets() {
  const datasets = await api("/api/market-datasets");
  renderLocalMarketDatasets(datasets);
  await loadStrategyDatasets();
}

async function createMarketDataset(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const intervals = Array.from(form.querySelectorAll('[name="intervals"]:checked')).map((item) => item.value);
  const payload = {name: form.elements.namedItem("name").value, stock_pool_id: form.elements.namedItem("stock_pool_id").value, start_date: form.elements.namedItem("start_date").value, intervals};
  const status = $("#market-dataset-form-status");
  status.textContent = "正在建立本地数据集…";
  try {
    const dataset = await api("/api/market-datasets", {method: "POST", body: JSON.stringify(payload)});
    status.textContent = "已加入队列：" + dataset.name;
    form.elements.namedItem("name").value = "";
    await refreshMarketDatasets();
  } catch (error) { status.textContent = error.message; }
}

function serializeForm() {
  const form = $("#experiment-form");
  const payload = {...state.config.strategy_defaults[strategy()]};
  const data = new FormData(form);
  for (const [name, raw] of data.entries()) {
    if (name === "symbols") {
      payload.symbols = raw.split(/[\s,]+/).filter(Boolean);
    } else {
      const control = preferredFormControl(form, name);
      payload[name] = control && control.type === "number"
        ? (raw === "" ? null : Number(raw))
        : raw;
    }
  }
  payload.strategy = strategy();
  payload.training_scope = isFk() ? "sector-pooled" : "per-symbol";
  payload.market_dataset_id = $("#dataset-select").value || null;
  payload.dataset_id = isFk() ? $("#dataset-select").value : null;
  payload.deterministic_training = $('[name="deterministic_training"]').checked;
  if (payload.max_folds == null) delete payload.max_folds;
  return payload;
}

async function checkBoundaryData() {
  if (!isFk()) return;
  state.boundaryReady = false;
  $("#run-button").disabled = true;
  const datasetId = $("#dataset-select").value;
  if (!datasetId) return;
  const start = $('[name="start"]').value, end = $('[name="end"]').value;
  const query = start && end ? "?" + new URLSearchParams({start,end}) : "";
  try {
    const result = await api("/api/boundary-data/status/" + encodeURIComponent(datasetId) + query);
    if (!isFk() || $("#dataset-select").value !== datasetId || $('[name="start"]').value !== start || $('[name="end"]').value !== end) return;
    state.boundaryReady = result.status === "ready";
    $("#boundary-data-status").textContent = state.boundaryReady
      ? "已就绪（滚动60日锚点）" : (result.reason || "未准备，请先构建并保存");
    $("#run-button").disabled = !state.boundaryReady;
  } catch {
    $("#boundary-data-status").textContent = "状态读取失败";
  }
}

async function buildBoundaryData() {
  const dataset = (state.config.market_datasets || []).find((item) => item.dataset_id === $("#dataset-select").value);
  if (!dataset) return;
  $("#boundary-data-status").textContent = "正在构建并保存";
  try { await api("/api/boundary-data/build", {
    method: "POST",
    body: JSON.stringify({
      dataset_id: dataset.dataset_id,
      database_path: dataset.database_path,
      symbols: dataset.symbols,
      start: $('[name="start"]').value,
      end: $('[name="end"]').value,
    }),
  });
    await checkBoundaryData();
    await loadStrategyDatasets();
  } catch(error) { $("#boundary-data-status").textContent = error.message; }
}

function statusText(status) {
  return ({queued: "排队", running: "运行中", completed: "完成", failed: "失败",
    cancelled: "已终止", interrupted: "已中断"})[status] || status;
}

function updateSelection() {
  $("#history-selected-count").textContent = "已选 " + state.selected.size;
  $("#batch-delete-button").classList.toggle("hidden", !state.editing || !state.selected.size);
  $("#history-batch-actions").classList.toggle("hidden", !state.editing);
  $("#history-select-all").checked = state.jobs.length > 0 &&
    state.selected.size === state.jobs.length;
}

function renderJobs() {
  $("#run-count").textContent = state.jobs.length;
  $("#job-list").innerHTML = state.jobs.map((job) => {
    const check = state.editing
      ? '<input class="job-check" type="checkbox" ' +
        (state.selected.has(job.id) ? "checked" : "") + ">"
      : "";
    return '<button class="job-item ' + (job.id === state.selectedId ? "active" : "") +
      '" data-id="' + job.id + '">' + check + "<span><strong>" +
      escapeHtml(job.request.name) + "</strong><small>" +
      (job.request.strategy === "bnhmm" ? "BNHMM" : "FK-贝叶斯") + " · " +
      statusText(job.status) + " · " +
      (job.summary?.symbols ?? job.request.symbols.length) + " 只</small></span></button>";
  }).join("");
  $$(".job-item").forEach((button) => button.addEventListener("click", (event) => {
    const id = button.dataset.id;
    if (state.editing) {
      event.preventDefault();
      state.selected.has(id) ? state.selected.delete(id) : state.selected.add(id);
      renderJobs();
    } else {
      selectJob(id);
    }
  }));
  updateSelection();
}

async function refreshJobs() {
  state.jobs = await api("/api/jobs");
  renderJobs();
  $("#training-status").textContent = state.jobs.some((item) => item.status === "running")
    ? "运行中" : "空闲";
  if (state.selectedId) await selectJob(state.selectedId, false);
  else if (state.jobs.length && !state.editing) await selectJob(state.jobs[0].id);
}

function setPage(page) {
  if (!pageNames.has(page)) page = "data";
  state.page = page;
  $$("[data-page-panel]").forEach((panel) =>
    panel.classList.toggle("hidden", panel.dataset.pagePanel !== page));
  $$(".primary-nav button").forEach((button) =>
    button.classList.toggle("active", button.dataset.page === page));
  if (page === "data") loadDataService();
  if (page === "data") loadCloudCatalog();
  if (page === "pools") loadPools();
  if (page === "pools" && state.verificationToken) loadQuantAiPools();
}

async function loadDataService() {
  try {
    const result = await api("/api/data-service");
    $("#service-source").textContent = result.source_type === "broker" ? "本机只读行情 broker" : "本地 DuckDB";
    $("#data-source-value").textContent = result.source || "未配置";
    $("#data-source-status").textContent = result.status === "connected" ? "已连接" : result.status;
    $("#data-checked-at").textContent = result.checked_at?.replace("T", " ").slice(0, 19) || "-";
    $("#data-dataset-count").textContent = result.local_market_dataset_count + " 个";
    $("#service-strip").classList.toggle("service-error", result.status !== "connected");
    $("#service-strip strong").textContent = result.status === "connected" ? "行情连接正常" : "行情连接异常";
    $("#device-list").innerHTML = (result.devices || []).map((device) =>
      '<div class="device-card"><strong>' + escapeHtml(device.name || device.id) + '</strong><span>' +
      (device.memory_total_mib ? device.memory_used_mib + " / " + device.memory_total_mib + " MiB" : device.id) +
      '</span></div>').join("") || '<span class="muted">未检测到设备</span>';
    const datasets = state.config?.datasets || [];
    $("#sector-dataset-count").textContent = String(datasets.length);
    $("#sector-datasets").innerHTML = datasets.length ? datasets.map((item) =>
      '<article class="dataset-row"><div><strong>' + escapeHtml(item.name || item.dataset_id) + '</strong><span>' +
      escapeHtml(item.description || "") + '</span></div><small>' + item.available_member_count + " / " + item.member_count + " 只 · " +
      (item.data_start || "-") + " 至 " + (item.data_end || "-") + '</small></article>').join("") :
      '<div class="empty-state">尚未发现本地板块数据集</div>';
    await loadStrategyDatasets();
  } catch (error) {
    $("#service-strip strong").textContent = error.message;
  }
}

async function loadCloudCatalog() {
  if (!state.verificationToken) return;
  const target = $("#cloud-datasets");
  target.innerHTML = '<span class="muted">正在读取云端目录…</span>';
  try {
    const result = await api("/api/verification/catalog", {method: "POST", body: JSON.stringify({access_token: state.verificationToken})});
    $("#cloud-dataset-count").textContent = result.total + " 个";
    target.innerHTML = result.datasets.length ? result.datasets.map((dataset) =>
      '<article class="dataset-row"><div><strong>' + escapeHtml(dataset.category + " · " + dataset.name) + '</strong><span>' + escapeHtml(dataset.description) + '</span></div><small>' + dataset.sample_count.toLocaleString() + " 样本 · " + dataset.input_intervals.join(", ") + '</small></article>').join("") :
      '<div class="empty-state">云端目录中还没有已发布数据集</div>';
  } catch (error) {
    target.innerHTML = '<div class="empty-state">' + escapeHtml(error.message) + '</div>';
    $("#cloud-dataset-count").textContent = "不可用";
  }
}

async function loadStrategyDatasets() {
  try {
    const datasets = await api("/api/strategy-datasets");
    $("#strategy-dataset-count").textContent = String(datasets.length);
    $("#strategy-datasets").innerHTML = datasets.length ? datasets.map((item) => {
      const ready = item.status === "ready";
      const p = item.progress || {};
      const progressText = item.status === "running" ? '<small class="strategy-build-progress">' + escapeHtml(p.message || "等待计算资源") +
        (typeof p.fraction === "number" ? " · " + (p.fraction*100).toFixed(1) + "%" : "") + " · 已用 " + window.DeepStockProgress.duration(p.elapsed_seconds) + " · 本阶段预计剩余 " + window.DeepStockProgress.duration(p.eta_seconds) + '</small>' : "";
      const deleteId = "strategy:" + item.dataset_id;
      const deleteLabel = state.pendingDeleteId === deleteId ? "再次点击确认" : "删除";
      return '<article class="dataset-row"><div class="market-dataset-info"><strong>' + escapeHtml(item.name) + '</strong><span>' +
        item.symbols + " 只 · 滚动 " + item.lookback_sessions + " 日 · " + (item.rows || 0).toLocaleString() + " 个交易日状态" + '</span><small>' +
        (item.start || "待构建") + " 至 " + (item.end || "待构建") + "（结束为读取上界） · " + (ready ? "已保存，可直接运行" : escapeHtml(item.reason || "未就绪")) + '</small><small>版本 ' + escapeHtml(item.construction_version || "—") + " · " + escapeHtml(item.available_at || "—") + " · 历史版本 " + (item.history?.length || 0) + '</small>' + progressText + '</div><div class="button-row"><span class="status-badge">' + (ready ? "ready" : item.status) + '</span><button type="button" class="secondary-button" data-prepare-strategy="' + escapeHtml(item.dataset_id) + '"' + (item.status === "running" ? " disabled" : "") + '>' + (ready ? "验证并复用" : "准备并保存") + '</button><button type="button" class="danger-button" data-delete-strategy="' + escapeHtml(item.dataset_id) + '"' + (item.status === "running" ? " disabled" : "") + '>' + deleteLabel + '</button></div></article>';
    }).join("") : '<div class="empty-state">还没有策略预计算数据集；从策略页构建并保存后，运行任务只会复用它。</div>';
    $$('[data-prepare-strategy]').forEach(button => button.addEventListener('click', async () => {
      const item = datasets.find(row => row.dataset_id === button.dataset.prepareStrategy);
      const market = state.config.market_datasets.find(row => row.dataset_id === item.dataset_id);
      if (!market) return;
      const dataEnd = market.last_refresh_date || market.start_date;
      const addDays = (value,count) => new Date(Date.parse(value+'T00:00:00Z')+count*86400000).toISOString().slice(0,10);
      const start = item.start || [addDays(market.start_date,450),addDays(dataEnd,-365)].sort().at(-1);
      const end = item.end || addDays(dataEnd,1);
      button.disabled = true;
      try { await api('/api/boundary-data/build',{method:'POST',body:JSON.stringify({dataset_id:item.dataset_id,start,end})}); await loadStrategyDatasets(); }
      catch(error) { button.disabled=false; button.textContent=error.message; }
    }));
    $$('[data-delete-strategy]').forEach(button => button.addEventListener('click', async () => {
      const item = datasets.find(row => row.dataset_id === button.dataset.deleteStrategy);
      if (!item || item.status === "running") return;
      const deleteId = "strategy:" + item.dataset_id;
      if (state.pendingDeleteId !== deleteId) {
        state.pendingDeleteId = deleteId;
        void loadStrategyDatasets();
        return;
      }
      state.pendingDeleteId = null;
      button.disabled = true;
      try { await api('/api/strategy-datasets/' + encodeURIComponent(item.dataset_id), {method:'DELETE'}); await refreshMarketDatasets(); }
      catch(error) { button.disabled=false; button.textContent=error.message; }
    }));
  } catch (error) {
    $("#strategy-dataset-count").textContent = "读取失败";
    $("#strategy-datasets").innerHTML = '<div class="empty-state">' + escapeHtml(error.message) + '</div>';
  }
}

async function loadPools() {
  state.pools = await api("/api/stock-pools");
  $("#pool-list").innerHTML = state.pools.length ? state.pools.map((pool) =>
    '<button class="pool-item ' + (pool.id === state.selectedPool ? "active" : "") + '" data-pool-id="' + pool.id + '"><strong>' +
    escapeHtml(pool.name) + '</strong><small>' + (pool.source === "quant-ai" ? "Quant AI · " : "") + pool.symbols.length + " 只股票</small></button>").join("") :
    '<div class="muted pool-empty">还没有股票池</div>';
  $$(".pool-item").forEach((button) => button.addEventListener("click", () => selectPool(button.dataset.poolId)));
  $("#strategy-pool").innerHTML = '<option value="">选择股票池</option>' + state.pools.map((pool) =>
    '<option value="' + pool.id + '">' + escapeHtml(pool.name) + " · " + pool.symbols.length + " 只</option>").join("");
  if (state.selectedPool && state.pools.some((pool) => pool.id === state.selectedPool)) selectPool(state.selectedPool);
  else if (state.pools[0]) selectPool(state.pools[0].id);
  else renderPool(null);
}

async function loadQuantAiPools() {
  const list = $("#quant-pool-list");
  if (!state.verificationToken) return;
  $("#quant-pool-refresh").disabled = false;
  $("#quant-pool-count").textContent = "正在读取";
  list.innerHTML = '<div class="empty-state">正在读取线上 AI 选股结果…</div>';
  try {
    const result = await api("/api/verification/stock-pools", {
      method: "POST", body: JSON.stringify({access_token: state.verificationToken}),
    });
    state.quantAiPools = result.pools || [];
    $("#quant-pool-count").textContent = result.total + " 个线上股票池";
    $("#quant-pool-sync-all").disabled = !state.quantAiPools.length;
    renderQuantAiPools();
  } catch (error) {
    $("#quant-pool-count").textContent = "读取失败";
    list.innerHTML = '<div class="empty-state">' + escapeHtml(error.message) + '</div>';
  }
}

function renderVerifiedUser() {
  const user = state.verificationUser;
  const loginFields = $("#verification-login-fields");
  loginFields.classList.toggle("hidden", Boolean(user));
  loginFields.querySelectorAll("input,button").forEach((control) => { control.disabled = Boolean(user); });
  $("#verification-clear").disabled = !user;
  $("#verification-result").innerHTML = user
    ? '<span class="status-light"></span><strong>' + escapeHtml(user.display_name || user.username || "Quant Platform 用户") +
      '</strong><small>在线</small>'
    : '<span class="status-light status-neutral"></span><span>未登录</span>';
  $("#verification-permissions").classList.toggle("hidden", !user);
  $("#verification-permissions").innerHTML = user
    ? '<span class="permission-item">账号在线</span><span class="permission-item ' + (state.deepstockAuthorized ? "permission-granted" : state.permissionStatus === "unavailable" ? "permission-pending" : "permission-denied") + '">DeepStock ' + (state.deepstockAuthorized ? "有权访问" : state.permissionStatus === "unavailable" ? "权限暂不可验证" : "无 fast_screener 权限") + '</span>'
    : "";
  $("#quant-watchlist-sync").disabled = !user;
}

async function applyVerifiedSession(result, token) {
  state.verificationToken = token;
  state.verificationUser = result.user;
  state.deepstockAuthorized = Boolean(result.authorized);
  state.permissionStatus = result.permission_status || (result.authorized ? "granted" : "denied");
  sessionStorage.setItem("deepstock.quant.access-token", token);
  $("#verification-status").textContent = "";
  $("#cloud-catalog-refresh").disabled = !state.deepstockAuthorized;
  renderVerifiedUser();
  await Promise.all([loadQuantAiPools(), loadQuantWatchlist()]);
  if (state.deepstockAuthorized) {
    loadCloudCatalog();
    $("#cloud-fk-worker-status").textContent = "正在连接 Quant Platform 在线策略 worker…";
    try {
      const worker = await api("/api/cloud-fk/connect", {
        method: "POST", body: JSON.stringify({access_token: token}),
      });
      $("#cloud-fk-worker-status").textContent = worker.connected
        ? "本机策略 worker 已连接；分钟行情与算法数据仅保存在本机。"
        : "本机策略 worker 尚未连接。";
    } catch (error) {
      $("#cloud-fk-worker-status").textContent = "在线策略 worker 连接失败：" + error.message;
    }
  } else {
    $("#cloud-fk-worker-status").textContent = "当前账号无 DeepStock 在线策略权限。";
  }
}

async function restoreVerifiedSession() {
  const token = sessionStorage.getItem("deepstock.quant.access-token");
  if (!token) return false;
  try {
    const result = await api("/api/verification/session", {
      method: "POST", body: JSON.stringify({access_token: token}),
    });
    await applyVerifiedSession(result, token);
    return true;
  } catch {
    sessionStorage.removeItem("deepstock.quant.access-token");
    state.verificationToken = null;
    state.verificationUser = null;
    state.deepstockAuthorized = false;
    renderVerifiedUser();
    return false;
  }
}

async function loadQuantWatchlist() {
  if (!state.verificationToken) return;
  $("#quant-watchlist-count").textContent = "正在读取";
  $("#quant-watchlist-status").textContent = "";
  try {
    const result = await api("/api/verification/watchlist", {
      method: "POST", body: JSON.stringify({access_token: state.verificationToken}),
    });
    state.quantWatchlist = result.items?.[0] || null;
    $("#quant-watchlist-count").textContent = (state.quantWatchlist?.symbols.length || 0) + " 只";
    $("#quant-watchlist-status").textContent = result.total
      ? "已同步查看 Quant Platform 自选股。点击按钮可将当前列表更新到本机。"
      : "Quant Platform 自选股为空";
  } catch (error) {
    state.quantWatchlist = null;
    $("#quant-watchlist-count").textContent = "读取失败";
    $("#quant-watchlist-status").textContent = error.message;
  }
}

async function syncQuantWatchlist() {
  if (!state.verificationToken) return;
  $("#quant-watchlist-sync").disabled = true;
  $("#quant-watchlist-status").textContent = "正在读取并同步用户自选股…";
  try {
    const remote = await api("/api/verification/watchlist", {
      method: "POST", body: JSON.stringify({access_token: state.verificationToken}),
    });
    if (!remote.total || !remote.items?.length) {
      state.quantWatchlist = null;
      $("#quant-watchlist-count").textContent = "0 只";
      $("#quant-watchlist-status").textContent = "Quant Platform 自选股为空，没有更改本机股票池。";
      return;
    }
    state.quantWatchlist = remote.items[0];
    const result = await api("/api/stock-pools/sync-quant-watchlist", {
      method: "POST",
      body: JSON.stringify({access_token: state.verificationToken, pool_id: state.quantWatchlist.pool_id}),
    });
    state.selectedPool = result.pool.id;
    await loadPools();
    $("#quant-watchlist-count").textContent = result.pool.symbols.length + " 只";
    $("#quant-watchlist-status").textContent = "已同步用户自选股到本机「" + result.pool.name + "」，共 " + result.pool.symbols.length + " 只。再次同步会更新该股票池。";
  } catch (error) {
    $("#quant-watchlist-status").textContent = error.message;
  } finally {
    $("#quant-watchlist-sync").disabled = !state.verificationUser;
  }
}

function renderQuantAiPools() {
  const list = $("#quant-pool-list");
  list.innerHTML = state.quantAiPools.length ? state.quantAiPools.map((pool) => {
    const local = state.pools.find((item) => item.cloud_pool_id === pool.pool_id);
    return '<article class="quant-pool-row"><div class="quant-pool-info"><div class="quant-pool-title"><strong>' +
      escapeHtml(pool.name) + '</strong><span>' + (pool.source === "fast" ? "快速选股" : "AI 选股") + '</span></div><p>' +
      escapeHtml(pool.description || pool.query || "Quant Platform AI 选股结果") + '</p><small>' + pool.symbols.length + " 只 · " +
      escapeHtml(pool.created_at || "") + (pool.run_id ? " · 运行 " + escapeHtml(pool.run_id) : "") + '</small></div><button class="secondary-button" data-quant-sync="' +
      escapeHtml(pool.pool_id) + '">' + (local ? "更新本地股票池" : "同步到本机") + '</button></article>';
  }).join("") : '<div class="empty-state">Quant Platform 中还没有已保存的 AI 选股结果</div>';
  $$('[data-quant-sync]').forEach((button) => button.addEventListener("click", () => syncQuantAiPool(button.dataset.quantSync, button)));
}

async function syncQuantAiPool(poolId, button) {
  if (!state.verificationToken) return;
  button.disabled = true;
  button.textContent = "同步中…";
  try {
    const result = await api("/api/stock-pools/sync-quant-ai", {
      method: "POST",
      body: JSON.stringify({access_token: state.verificationToken, pool_id: poolId}),
    });
    state.selectedPool = result.pool.id;
    await loadPools();
    renderQuantAiPools();
    $("#pool-error").textContent = "已同步「" + result.pool.name + "」至本机，共 " + result.pool.symbols.length + " 只";
  } catch (error) {
    $("#pool-error").textContent = error.message;
    button.disabled = false;
    button.textContent = "重试同步";
  }
}

function renderPool(pool) {
  state.selectedPool = pool?.id || null;
  $("#pool-title").textContent = pool?.name || "选择股票池";
  $("#pool-count").textContent = (pool?.symbols.length || 0) + " 只";
  $("#pool-origin").textContent = pool?.source === "quant-ai"
    ? "来源：Quant Platform AI 选股 · 同步 " + (pool.synced_at || "")
    : "手工维护";
  $("#pool-description").textContent = pool?.description || "";
  $("#pool-description").classList.toggle("hidden", !pool?.description);
  $("#pool-delete").disabled = !pool;
  $("#pool-symbol-form button").disabled = !pool;
  $("#pool-symbols").innerHTML = pool ? (pool.symbols.length ? pool.symbols.map((symbol) =>
    '<span class="symbol-chip"><strong>' + escapeHtml(symbol) + '</strong><button type="button" data-symbol="' + symbol + '" title="移除 ' + symbol + '">×</button></span>').join("") :
    '<div class="empty-state">股票池为空，手动添加证券代码</div>') : '<div class="empty-state">选择或新建一个股票池</div>';
  $$(".symbol-chip button").forEach((button) => button.addEventListener("click", async () => {
    await api("/api/stock-pools/" + state.selectedPool + "/symbols/" + encodeURIComponent(button.dataset.symbol), {method: "DELETE"});
    await loadPools();
  }));
}

function searchPoolSymbols(query) {
  window.clearTimeout(symbolSearchTimer);
  const prefix = query.trim().toUpperCase();
  if (prefix.length < 2) {
    $("#stock-symbol-options").replaceChildren();
    return;
  }
  symbolSearchTimer = window.setTimeout(async () => {
    try {
      const result = await api("/api/symbols?" + new URLSearchParams({q: prefix, limit: "25"}));
      $("#stock-symbol-options").innerHTML = result.symbols.map((symbol) =>
        '<option value="' + escapeHtml(symbol) + '"></option>').join("");
    } catch { $("#stock-symbol-options").replaceChildren(); }
  }, 250);
}

function selectPool(id) {
  renderPool(state.pools.find((pool) => pool.id === id) || null);
  $$(".pool-item").forEach((button) => button.classList.toggle("active", button.dataset.poolId === id));
}

async function testAiService() {
  const form = $("#ai-form");
  const data = Object.fromEntries(new FormData(form));
  $("#ai-status").textContent = "正在读取模型列表...";
  try {
    const result = await api("/api/ai/models", {method: "POST", body: JSON.stringify(data)});
    $("#ai-model-options").innerHTML = result.models.map((model) => '<option value="' + escapeHtml(model) + '"></option>').join("");
    $("#ai-status").textContent = "已连接，发现 " + result.count + " 个模型。模型名称可继续手动输入。";
  } catch (error) { $("#ai-status").textContent = error.message; }
}

function loadAiConfig() {
  try {
    const saved = JSON.parse(localStorage.getItem("deepstock.ai.config.v1") || "{}");
    for (const name of ["provider", "base_url", "model"]) {
      if (typeof saved[name] === "string") $("#ai-form").elements.namedItem(name).value = saved[name];
    }
    $("#ai-form").elements.namedItem("api_key").value = sessionStorage.getItem("deepstock.ai.key") || "";
  } catch { $("#ai-status").textContent = "浏览器未开放会话存储"; }
}

function saveAiConfig() {
  const form = $("#ai-form");
  const config = Object.fromEntries(["provider", "base_url", "model"].map((name) => [name, form.elements.namedItem(name).value.trim()]));
  try {
    localStorage.setItem("deepstock.ai.config.v1", JSON.stringify(config));
    sessionStorage.setItem("deepstock.ai.key", form.elements.namedItem("api_key").value);
    $("#ai-status").textContent = "配置已保存；密钥只保存在当前浏览器会话。";
  } catch { $("#ai-status").textContent = "无法保存配置，请检查浏览器存储权限。"; }
}

async function verifyAccount(event) {
  event.preventDefault();
  const status = $("#verification-status");
  state.verificationToken = null;
  state.verificationUser = null;
  state.quantAiPools = [];
  $("#quant-pool-count").textContent = "请先验证账号";
  $("#quant-pool-list").innerHTML = '<div class="empty-state">在“用户验证”页面登录并授权后查看自己的线上 AI 选股结果</div>';
  $("#quant-pool-refresh").disabled = true;
  $("#quant-pool-sync-all").disabled = true;
  $("#cloud-dataset-count").textContent = "需要验证";
  $("#cloud-datasets").innerHTML = '<div class="empty-state">验证 Quant Platform 账号和访问资格后查看</div>';
  $("#cloud-catalog-refresh").disabled = true;
  renderVerifiedUser();
  status.textContent = "正在验证...";
  try {
    const credentials = Object.fromEntries(new FormData(event.target));
    const result = await api("/api/verification/login", {method: "POST", body: JSON.stringify(credentials)});
    event.target.elements.namedItem("password").value = "";
    await applyVerifiedSession(result, result.access_token);
    status.textContent = "";
  } catch (error) { event.target.elements.namedItem("password").value = ""; state.verificationToken = null; state.verificationUser = null; state.deepstockAuthorized = false; state.quantAiPools = []; sessionStorage.removeItem("deepstock.quant.access-token"); renderVerifiedUser(); status.textContent = error.message; }
}

function setTab(name) {
  $$("#result-tabs button").forEach((button) =>
    button.classList.toggle("active", button.dataset.tab === name));
  $$(".result-panel").forEach((panel) =>
    panel.classList.toggle("hidden", panel.dataset.panel !== name));
}

async function artifactJson(job, names) {
  const name = names.find((item) => job.artifacts?.[item]);
  return name ? api(job.artifacts[name]) : null;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"]/g, (char) =>
    ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[char]));
}

function jsonBlock(value) {
  return value == null
    ? '<div class="empty-state">暂无结果</div>'
    : "<pre>" + escapeHtml(JSON.stringify(value, null, 2)) + "</pre>";
}

function imageBlock(url, alt) {
  return url
    ? '<img class="result-image" src="' + url + '" alt="' + alt + '">'
    : '<div class="empty-state">暂无结果</div>';
}

function isJobFk(job) {
  return job.request.strategy === "boundary-flow-fk";
}

let renderedArtifactKey = null;
async function renderArtifacts(job) {
  const key = JSON.stringify([job.id, job.status, job.finished_at, job.artifacts]);
  if (key === renderedArtifactKey && !["queued", "running"].includes(job.status)) return;
  const [evaluation, predictions, posterior, summary, residual] = await Promise.all([
    artifactJson(job, isJobFk(job) ? ["sector-direction-evaluation.json", "boundary-policy-evaluation.json"] : ["evaluation.json"]),
    artifactJson(job, isJobFk(job) ? ["feynman-kac-predictions.json"] : ["predictions.json"]),
    artifactJson(job, isJobFk(job) ? ["boundary-model-posterior.json"] : ["fk-diagnostics.json"]),
    artifactJson(job, ["run-summary.json"]), artifactJson(job, ["boundary-residual-analysis.json"]),
  ]);
  if (state.selectedId !== job.id) return;
  if (window.DeepStockDashboard) {
    window.DeepStockDashboard.render(job, {evaluation, predictions, posterior, summary, residual});
    renderedArtifactKey = key;
    return;
  }
  $("#evaluation-content").innerHTML = jsonBlock(evaluation);
  $("#predictions-content").innerHTML = jsonBlock(predictions);
  $("#posterior-content").innerHTML = jsonBlock(posterior);
  $("#summary-content").innerHTML = jsonBlock(summary);
  $("#residuals-content").innerHTML = residual
    ? jsonBlock(residual) : imageBlock(job.artifacts["residuals.png"], "残差图");
  $("#returns-content").innerHTML = job.artifacts["equity_drawdown.png"]
    ? imageBlock(job.artifacts["equity_drawdown.png"], "净值与回撤")
    : jsonBlock(await artifactJson(job, ["boundary-policy-evaluation.json"]));
  renderedArtifactKey = key;
}

async function selectJob(id, rerender = true) {
  state.selectedId = id;
  const job = await api("/api/jobs/" + encodeURIComponent(id));
  if (state.selectedId !== id) return;
  $("#empty-state").classList.add("hidden");
  $("#job-view").classList.remove("hidden");
  $("#run-title").textContent = job.request.name;
  $("#run-meta").textContent = (isJobFk(job) ? "FK-贝叶斯" : "BNHMM") + " · " +
    job.request.start + " 至 " + job.request.end + " · " + statusText(job.status);
  $("#job-subtitle").textContent = job.error || job.output_dir;
  $("#job-log").textContent = job.log || "";
  const active = ["queued", "running"].includes(job.status);
  $("#cancel-button").classList.toggle("hidden", !active);
  const publish = $("#publish-cloud-button");
  publish.classList.toggle("hidden", !isJobFk(job) || job.status !== "completed");
  if (publish.dataset.jobId !== job.id) {
    publish.dataset.jobId = job.id;
    $("#publish-cloud-status").textContent = "";
  }
  publish.onclick = async () => {
    const status = $("#publish-cloud-status");
    if (!state.verificationToken) { status.textContent = "请先在用户验证页面登录 Quant Platform，再发布训练结果。"; return; }
    const dataset = state.config.market_datasets.find(item => item.dataset_id === (job.request.market_dataset_id || job.request.dataset_id));
    const sector = [["半导体","semiconductor"],["人工智能","artificial_intelligence"],["新能源","new_energy"]].find(([name]) => dataset?.name.includes(name))?.[1];
    if (!sector) { status.textContent = "无法识别任务板块，请确认对应本地数据集。"; return; }
    publish.disabled = true; status.textContent = "正在发布已保存训练结果…";
    try {
      await api("/api/cloud-fk/publish/" + encodeURIComponent(job.id), {method:"POST", body:JSON.stringify({sector,access_token:state.verificationToken})});
      status.textContent = "已发布，可在 Quant Platform → 云端策略 → FK-贝叶斯查看。再次发布会复用同一条记录。";
    } catch(error) { status.textContent = error.message; }
    finally { publish.disabled = false; }
  };
  $("#cancel-button").onclick = async () => {
    await api("/api/jobs/" + encodeURIComponent(job.id) + "/cancel", {method: "POST"});
    await refreshJobs();
  };
  window.DeepStockProgress.render(job);
  $("#gpu-monitor").textContent = job.peak_gpu_memory_mib
    ? "显存 " + job.current_gpu_memory_mib + " / 峰值 " +
      job.peak_gpu_memory_mib + " MiB" : "";
  await renderArtifacts(job);
  if (rerender) renderJobs();
}

async function submit(event) {
  event.preventDefault();
  $("#form-error").textContent = "";
  try {
    if (!$("#dataset-select").value) throw new Error("请先加载一个本地分钟行情数据集");
    if (isFk()) { await checkBoundaryData(); if (!state.boundaryReady) throw new Error("策略数据集未就绪，请在数据服务页先准备并保存。"); }
    const job = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify(serializeForm()),
    });
    state.selectedId = job.id;
    await refreshJobs();
    $('[name="name"]').value = nextName(strategy());
  } catch (error) {
    $("#form-error").textContent = error.message;
  }
}

async function init() {
  state.config = await api("/api/config");
  $$(".primary-nav button").forEach((button) => button.addEventListener("click", () => {
    const target = "#" + button.dataset.page;
    if (location.hash === target) setPage(button.dataset.page);
    else location.hash = button.dataset.page;
  }));
  window.addEventListener("hashchange", () => setPage(location.hash.slice(1)));
  $("#service-check").addEventListener("click", loadDataService);
  $("#data-refresh").addEventListener("click", loadDataService);
  $("#strategy-data-open").addEventListener("click", () => { location.hash = "data"; });
  $("#market-dataset-form").addEventListener("submit", createMarketDataset);
  $("#pool-create").addEventListener("click", () => $("#pool-dialog").showModal());
  $("#pool-create-cancel").addEventListener("click", () => $("#pool-dialog").close());
  $("#pool-create-form").addEventListener("submit", async (event) => { event.preventDefault(); try { const pool = await api("/api/stock-pools", {method: "POST", body: JSON.stringify(Object.fromEntries(new FormData(event.target)))}); $("#pool-dialog").close(); $("#pool-create-error").textContent = ""; event.target.reset(); state.selectedPool = pool.id; await loadPools(); } catch (error) { $("#pool-create-error").textContent = error.message; } });
  $("#pool-symbol-form").addEventListener("submit", async (event) => { event.preventDefault(); if (!state.selectedPool) return; try { await api("/api/stock-pools/" + state.selectedPool + "/symbols", {method: "POST", body: JSON.stringify({symbol: $("#pool-symbol-input").value})}); $("#pool-symbol-input").value = ""; await loadPools(); } catch (error) { $("#pool-error").textContent = error.message; } });
  $("#pool-symbol-input").addEventListener("input", (event) => searchPoolSymbols(event.target.value));
  $("#pool-delete").addEventListener("click", async () => { if (state.selectedPool && confirm("删除当前股票池？")) { await api("/api/stock-pools/" + state.selectedPool, {method: "DELETE"}); state.selectedPool = null; await loadPools(); } });
  $("#ai-test").addEventListener("click", testAiService);
  $("#ai-save").addEventListener("click", saveAiConfig);
  $("#verification-form").addEventListener("submit", verifyAccount);
  $("#verification-clear").addEventListener("click", () => {
    void api("/api/cloud-fk/disconnect", {method: "POST"});
    sessionStorage.removeItem("deepstock.quant.access-token");
    state.verificationToken = null;
    state.verificationUser = null;
    state.deepstockAuthorized = false;
    $("#cloud-fk-worker-status").textContent = "本机策略 worker 未连接";
    state.quantAiPools = [];
    state.quantWatchlist = null;
    $("#verification-form").reset();
    $("#verification-status").textContent = "";
    $("#cloud-catalog-refresh").disabled = true;
    $("#cloud-dataset-count").textContent = "需要验证";
    $("#cloud-datasets").innerHTML = '<div class="empty-state">验证 Quant Platform 账号和访问资格后查看</div>';
    $("#quant-pool-count").textContent = "请先验证账号";
    $("#quant-pool-list").innerHTML = '<div class="empty-state">验证账号后查看自己的线上 AI 选股结果</div>';
    $("#quant-pool-refresh").disabled = true;
    $("#quant-pool-sync-all").disabled = true;
    $("#quant-watchlist-count").textContent = "请先验证账号";
    $("#quant-watchlist-status").textContent = "";
    renderVerifiedUser();
  });
  $("#cloud-catalog-refresh").addEventListener("click", loadCloudCatalog);
  $("#quant-pool-refresh").addEventListener("click", loadQuantAiPools);
  $("#quant-watchlist-sync").addEventListener("click", syncQuantWatchlist);
  $("#quant-pool-sync-all").addEventListener("click", async () => {
    for (const button of $$('[data-quant-sync]')) {
      await syncQuantAiPool(button.dataset.quantSync, button);
    }
  });
  $("#open-training").addEventListener("click", () => { location.hash = "training"; setPage("training"); });
  $("#strategy-pool-apply").addEventListener("click", () => { const pool = state.pools.find((item) => item.id === $("#strategy-pool").value); if (pool) { $("[name=\"symbols\"]").value = pool.symbols.join(","); $("#form-error").textContent = "已加载 " + pool.symbols.length + " 只标的"; } });
  renderDatasets();
  $("#device-select").innerHTML = (state.config.devices || []).map((item) =>
    '<option value="' + item.id + '">' + item.name + "</option>").join("");
  fillForm(state.config.defaults);
  loadAiConfig();
  renderVerifiedUser();
  await restoreVerifiedSession();
  await loadPools();
  await refreshMarketDatasets();
  await refreshJobs();
  syncStrategy();
  $$('input[name="strategy"]').forEach((input) =>
    input.addEventListener("change", syncStrategy));
  $("#experiment-form").addEventListener("submit", submit);
  $("#refresh-button").addEventListener("click", refreshJobs);
  $("#boundary-data-build").addEventListener("click", buildBoundaryData);
  $$('[name="start"],[name="end"]').forEach(input => input.addEventListener('change',checkBoundaryData));
  $("#edit-history-button").addEventListener("click", () => {
    state.editing = !state.editing;
    state.selected.clear();
    $("#edit-history-button").textContent = state.editing ? "完成" : "编辑";
    renderJobs();
  });
  $("#history-select-all").addEventListener("change", (event) => {
    state.selected = event.target.checked
      ? new Set(state.jobs.map((item) => item.id)) : new Set();
    renderJobs();
  });
  $("#batch-delete-button").addEventListener("click", async () => {
    for (const id of state.selected) {
      await api("/api/jobs/" + encodeURIComponent(id), {method: "DELETE"});
    }
    state.selected.clear();
    state.selectedId = null;
    await refreshJobs();
  });
  $$("#result-tabs button").forEach((button) =>
    button.addEventListener("click", () => setTab(button.dataset.tab)));
  window.setInterval(refreshJobs, 3000);
  window.setInterval(refreshMarketDatasets, 15000);
  window.setInterval(() => { if(isFk()) void checkBoundaryData(); if(state.page==='data') void loadStrategyDatasets(); }, 3000);
  setPage(location.hash.slice(1) || "data");
}

init().catch((error) => {
  $("#form-error").textContent = error.message;
});
