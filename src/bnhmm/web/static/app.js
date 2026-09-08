const state = {
  config: null,
  jobs: [],
  selectedId: null,
  selected: new Set(),
  editing: false,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
  });
  if (!response.ok) {
    throw new Error((await response.text()) || ("HTTP " + response.status));
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
    element.disabled = Boolean(element.closest(".hidden"));
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
  $("#dataset-field").classList.toggle("hidden", kind === "bnhmm");
  syncStrategyControlAvailability();
  if (kind === "boundary-flow-fk") checkBoundaryData();
}

function renderDatasets() {
  const select = $("#dataset-select");
  select.innerHTML = state.config.datasets.map((item) =>
    '<option value="' + item.dataset_id + '">' + (item.name || item.dataset_id) + "</option>"
  ).join("");
  select.addEventListener("change", applyDataset);
  applyDataset();
}

function applyDataset() {
  const dataset = state.config.datasets.find((item) => item.dataset_id === $("#dataset-select").value);
  if (!dataset) return;
  $('[name="database_path"]').value = dataset.database_path;
  $('[name="symbols"]').value = dataset.symbols.join(",");
  $("#dataset-summary").textContent = dataset.symbols.length + " 只股票 · " +
    (dataset.start || "") + " 至 " + (dataset.end || "");
  checkBoundaryData();
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
  payload.dataset_id = isFk() ? $("#dataset-select").value : null;
  payload.deterministic_training = $('[name="deterministic_training"]').checked;
  if (payload.max_folds == null) delete payload.max_folds;
  return payload;
}

async function checkBoundaryData() {
  if (!isFk() || !$("#dataset-select").value) return;
  try {
    const result = await api("/api/boundary-data/status/" + encodeURIComponent($("#dataset-select").value));
    const variant = (result.variants || []).find((item) => item.lookback_sessions === 60);
    $("#boundary-data-status").textContent = variant && variant.status === "ready"
      ? "已就绪（滚动60日锚点）" : "需要重建";
  } catch {
    $("#boundary-data-status").textContent = "状态读取失败";
  }
}

async function buildBoundaryData() {
  const dataset = state.config.datasets.find((item) => item.dataset_id === $("#dataset-select").value);
  if (!dataset) return;
  $("#boundary-data-status").textContent = "正在重建";
  await api("/api/boundary-data/build", {
    method: "POST",
    body: JSON.stringify({
      dataset_id: dataset.dataset_id,
      database_path: dataset.database_path,
      symbols: dataset.symbols,
      start: dataset.start,
      end: dataset.end,
    }),
  });
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
  $("#global-status").textContent = state.jobs.some((item) => item.status === "running")
    ? "运行中" : "空闲";
  if (state.selectedId) await selectJob(state.selectedId, false);
}

function setTab(name) {
  $$("#result-tabs button").forEach((button) =>
    button.classList.toggle("active", button.dataset.tab === name));
  $$(".result-panel").forEach((panel) =>
    panel.classList.toggle("hidden", panel.dataset.panel !== name));
}

async function artifactJson(job, names) {
  const name = names.find((item) => job.artifacts[item]);
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

async function renderArtifacts(job) {
  const evaluation = await artifactJson(job, isJobFk(job)
    ? ["sector-direction-evaluation.json", "boundary-policy-evaluation.json"]
    : ["evaluation.json"]);
  const predictions = await artifactJson(job, isJobFk(job)
    ? ["feynman-kac-predictions.json"] : ["predictions.json"]);
  const posterior = await artifactJson(job, isJobFk(job)
    ? ["boundary-model-posterior.json"] : ["fk-diagnostics.json"]);
  const summary = await artifactJson(job, ["run-summary.json"]);
  const residual = await artifactJson(job, ["boundary-residual-analysis.json"]);
  $("#evaluation-content").innerHTML = jsonBlock(evaluation);
  $("#predictions-content").innerHTML = jsonBlock(predictions);
  $("#posterior-content").innerHTML = jsonBlock(posterior);
  $("#summary-content").innerHTML = jsonBlock(summary);
  $("#residuals-content").innerHTML = residual
    ? jsonBlock(residual) : imageBlock(job.artifacts["residuals.png"], "残差图");
  $("#returns-content").innerHTML = job.artifacts["equity_drawdown.png"]
    ? imageBlock(job.artifacts["equity_drawdown.png"], "净值与回撤")
    : jsonBlock(await artifactJson(job, ["boundary-policy-evaluation.json"]));
}

async function selectJob(id, rerender = true) {
  state.selectedId = id;
  const job = await api("/api/jobs/" + encodeURIComponent(id));
  $("#empty-state").classList.add("hidden");
  $("#job-view").classList.remove("hidden");
  $("#run-title").textContent = job.request.name;
  $("#run-meta").textContent = (isJobFk(job) ? "FK-贝叶斯" : "BNHMM") + " · " +
    job.request.start + " 至 " + job.request.end + " · " + statusText(job.status);
  $("#job-subtitle").textContent = job.error || job.output_dir;
  $("#job-log").textContent = job.log || "";
  const active = ["queued", "running"].includes(job.status);
  $("#cancel-button").classList.toggle("hidden", !active);
  $("#cancel-button").onclick = async () => {
    await api("/api/jobs/" + encodeURIComponent(job.id) + "/cancel", {method: "POST"});
    await refreshJobs();
  };
  $("#progress-band").classList.toggle("hidden", !active);
  const progress = job.progress || {};
  $("#progress-label").textContent = progress.message || statusText(job.status);
  $("#progress-value").value = Math.round((progress.fraction || 0) * 100);
  $("#elapsed-time").textContent = progress.elapsed_seconds
    ? Math.round(progress.elapsed_seconds) + " 秒" : "";
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
  renderDatasets();
  $("#device-select").innerHTML = (state.config.devices || []).map((item) =>
    '<option value="' + item.id + '">' + item.name + "</option>").join("");
  fillForm(state.config.defaults);
  await refreshJobs();
  syncStrategy();
  $$('input[name="strategy"]').forEach((input) =>
    input.addEventListener("change", syncStrategy));
  $("#experiment-form").addEventListener("submit", submit);
  $("#refresh-button").addEventListener("click", refreshJobs);
  $("#boundary-data-build").addEventListener("click", buildBoundaryData);
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
}

init().catch((error) => {
  $("#form-error").textContent = error.message;
});
