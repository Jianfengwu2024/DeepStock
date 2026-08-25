const state = {
  config: null,
  devices: [],
  jobs: [],
  comparisons: [],
  selectedId: null,
  selectedComparisonId: null,
  symbols: [],
  datasets: [],
  dialogSelected: new Set(),
  resultScopeByJob: new Map(),
  symbolReportCache: new Map(),
  boundaryResultCache: new Map(),
  boundaryTabByJob: new Map(),
  strategyFormValues: new Map(),
  activeStrategy: "bnhmm",
  historyEditing: false,
  historySelected: new Set(),
  historyDeleting: false,
  polling: false,
};

const numberFields = new Set([
  "epochs", "pretrain_epochs", "pretrain_mask_probability", "batch_size", "learning_rate", "kl_weight", "ordinal_weight", "symmetry_weight", "auxiliary_gate_floor", "auxiliary_gate_initial", "auxiliary_warmup_epochs", "forward_sessions", "strong_threshold_sigma",
  "return_standardization_lookback", "return_standardization_min_observations", "minimum_return_scale_percent",
  "minimum_train_sessions", "tuning_sessions", "test_sessions", "step_sessions",
  "max_folds", "bin_width_bps", "max_profile_bins", "normalization_lookback_sessions", "velocity_bins", "velocity_aggregation_minutes", "velocity_clip", "velocity_smoothing", "sequence_length", "sequence_stride",
  "minimum_sequence_length", "d_model", "latent_states", "transition_rank", "motif_dim", "heads", "layers",
  "feedforward_dim", "dropout", "mc_samples", "uncertainty_aversion",
  "transaction_cost_bps", "seed", "conservation_weight",
  "conservation_grid_points", "conservation_ridge", "minimum_token_bins",
  "maximum_tokens", "boundary_threshold",
  "holographic_volume_bins", "holographic_smoothing",
  "normalization_minimum_observations", "maximum_lag_minutes",
  "minimum_attribution_rows", "factor_process_variance",
  "sector_flow_ranking_window", "sector_flow_evaluation_horizon_minutes",
  "sector_flow_top_k", "sector_flow_monte_carlo_samples",
  "regime_context_sessions", "regime_hidden_channels", "regime_latent_states",
  "regime_neutral_sigma", "regime_volatility_lookback_sessions",
  "regime_volatility_minimum_observations", "regime_confidence_threshold",
  "regime_maximum_bull_variance", "regime_epochs", "regime_batch_size",
  "regime_learning_rate", "regime_kl_weight", "regime_state_loss_weight",
  "regime_mc_samples",
  "topology_lookback_sessions", "boundary_fk_paths",
  "boundary_drift_half_life_sessions", "boundary_minimum_direction_probability",
  "boundary_minimum_signal_to_noise", "boundary_minimum_survival_probability",
]);

const strategyParameterFields = {
  bnhmm: [
    "pretrain_epochs", "pretrain_mask_probability", "kl_weight", "ordinal_weight",
    "symmetry_weight", "deterministic_training", "feature_mode", "auxiliary_gate_floor",
    "auxiliary_gate_initial", "auxiliary_warmup_epochs", "velocity_bins", "velocity_aggregation_minutes", "velocity_clip",
    "velocity_smoothing", "sequence_length", "d_model", "motif_dim", "heads", "layers",
    "feedforward_dim", "dropout", "latent_states", "transition_rank", "mc_samples", "device",
  ],
  "holo-bnhmm": [
    "pretrain_epochs", "pretrain_mask_probability", "kl_weight", "ordinal_weight",
    "symmetry_weight", "deterministic_training", "velocity_aggregation_minutes",
    "velocity_clip", "holographic_volume_bins", "holographic_smoothing",
    "sequence_length", "d_model", "heads", "layers", "feedforward_dim", "dropout",
    "latent_states", "transition_rank", "mc_samples", "device",
  ],
  "holo-bnhmm-lite": [
    "pretrain_epochs", "pretrain_mask_probability", "kl_weight", "ordinal_weight",
    "symmetry_weight", "deterministic_training", "velocity_aggregation_minutes",
    "velocity_clip", "holographic_volume_bins", "holographic_smoothing",
    "sequence_length", "d_model", "heads", "layers", "feedforward_dim", "dropout",
    "latent_states", "transition_rank", "mc_samples", "device",
  ],
  "holo-repair": [
    "repair_variant", "pretrain_epochs", "pretrain_mask_probability", "kl_weight",
    "ordinal_weight", "symmetry_weight", "deterministic_training",
    "velocity_aggregation_minutes", "velocity_clip", "holographic_volume_bins",
    "holographic_smoothing", "sequence_length", "d_model", "heads", "layers",
    "feedforward_dim", "dropout", "latent_states", "transition_rank", "mc_samples", "device",
  ],
  "stock-transformer": [
    "conservation_weight", "conservation_grid_points", "conservation_ridge",
    "minimum_token_bins", "maximum_tokens", "boundary_threshold", "sequence_length",
    "d_model", "heads", "layers", "feedforward_dim", "dropout", "mc_samples", "device",
  ],
  "fk-stock": [],
  "sector-flow-bayes": [
    "minimum_train_sessions", "normalization_lookback_sessions",
    "sector_flow_path_windows", "sector_flow_top_k", "sector_index_symbol",
    "regime_context_sessions", "regime_hidden_channels", "regime_latent_states",
    "regime_neutral_sigma", "regime_volatility_lookback_sessions",
    "regime_volatility_minimum_observations", "regime_confidence_threshold",
    "regime_maximum_bull_variance", "regime_epochs", "regime_batch_size",
    "regime_learning_rate", "regime_kl_weight", "regime_state_loss_weight",
    "regime_mc_samples", "device",
  ],
  "boundary-flow-fk": [
    "minimum_train_sessions", "max_folds",
    "regime_neutral_sigma", "boundary_fk_paths", "boundary_drift_half_life_sessions",
    "boundary_minimum_direction_probability", "boundary_minimum_signal_to_noise",
    "boundary_minimum_survival_probability", "transaction_cost_bps", "seed",
  ],
};

const strategyNamePrefixes = {
  bnhmm: "BNHMM",
  "holo-bnhmm": "HoloBNHMM",
  "holo-bnhmm-lite": "HoloLite",
  "holo-repair": "HoloRepair",
  "stock-transformer": "StockTransformer",
  "fk-stock": "FKStock",
  "sector-flow-bayes": "SectorFlow",
  "boundary-flow-fk": "PathEfficiencyFK-60D",
};

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
    for (const field of namedFormControls(form, name)) {
      if (name === "symbols") field.value = value.join(", ");
      else if (field.type === "radio") field.checked = field.value === value;
      else if (field.type === "checkbox") field.checked = Boolean(value);
      else field.value = value ?? "";
    }
  }
  state.activeStrategy = currentStrategy();
  state.strategyFormValues.set(
    state.activeStrategy,
    readFormFields(strategyParameterFields[state.activeStrategy]),
  );
  syncStrategySelection();
  syncModelConstraints();
  syncDatasetSelection();
  if (currentStrategy() === "boundary-flow-fk") refreshBoundaryDataStatus();
}

function currentStrategy() {
  return document.querySelector("[name=strategy]:checked")?.value || "bnhmm";
}

function strategyPrefix(strategy = currentStrategy()) {
  if (strategy === "holo-repair") {
    return document.querySelector("#repair-variant")?.value || "HoloRepair";
  }
  return strategyNamePrefixes[strategy] || strategy;
}

function nextExperimentName(strategy = currentStrategy()) {
  const prefix = strategyPrefix(strategy);
  const pattern = new RegExp(`^${prefix}-(\\d+)$`, "i");
  const maximum = state.jobs.reduce((value, job) => {
    if (job.request.strategy !== strategy) return value;
    if (strategy === "holo-repair" && job.request.repair_variant !== prefix) return value;
    const match = String(job.request.name || "").match(pattern);
    return match ? Math.max(value, Number(match[1])) : value;
  }, 0);
  return `${prefix}-${String(maximum + 1).padStart(3, "0")}`;
}

function setNextExperimentName(strategy = currentStrategy()) {
  document.querySelector('[name="name"]').value = nextExperimentName(strategy);
}

function namedFormControls(form, name) {
  return [...form.elements].filter(field => field.name === name);
}

function preferredFormControl(form, name) {
  const controls = namedFormControls(form, name);
  return controls.find(field => !field.disabled)
    || controls.find(field => !field.closest(".hidden"))
    || controls[0];
}

function readFormFields(names) {
  const form = document.querySelector("#experiment-form");
  return Object.fromEntries(names.map(name => {
    const field = preferredFormControl(form, name);
    return [name, field?.type === "checkbox" ? field.checked : field?.value];
  }));
}

function writeFormFields(values) {
  const form = document.querySelector("#experiment-form");
  for (const [name, value] of Object.entries(values || {})) {
    for (const field of namedFormControls(form, name)) {
      if (field.type === "checkbox") field.checked = Boolean(value);
      else field.value = value ?? "";
    }
  }
}

function changeStrategy() {
  const strategy = currentStrategy();
  if (strategy !== state.activeStrategy) {
    state.strategyFormValues.set(
      state.activeStrategy,
      readFormFields(strategyParameterFields[state.activeStrategy]),
    );
    const saved = state.strategyFormValues.get(strategy);
    const preset = state.config?.strategy_defaults?.[strategy] || {};
    const fields = strategyParameterFields[strategy];
    writeFormFields(saved || Object.fromEntries(fields.map(name => [name, preset[name]])));
    state.activeStrategy = strategy;
    setNextExperimentName(strategy);
  }
  syncStrategySelection();
  syncModelConstraints();
  syncDatasetSelection();
  if (strategy === "boundary-flow-fk") refreshBoundaryDataStatus();
  icons();
}

function syncStrategySelection() {
  const strategy = currentStrategy();
  const stock = strategy === "stock-transformer";
  const fk = strategy === "fk-stock";
  const sectorFlow = strategy === "sector-flow-bayes";
  const boundaryFlow = strategy === "boundary-flow-fk";
  const holoLite = strategy === "holo-bnhmm-lite";
  const repair = strategy === "holo-repair";
  const repairName = document.querySelector("#repair-variant")?.value || "HoloRepair";
  const holo = strategy === "holo-bnhmm" || holoLite || repair;
  const bayesianHmm = strategy === "bnhmm" || holo;
  const neural = !fk && !sectorFlow && !boundaryFlow;
  document.querySelectorAll(".bnhmm-only").forEach(element => element.classList.toggle("hidden", strategy !== "bnhmm"));
  document.querySelectorAll(".holo-only").forEach(element => element.classList.toggle("hidden", !holo));
  document.querySelectorAll(".repair-only").forEach(element => element.classList.toggle("hidden", !repair));
  document.querySelectorAll(".bayesian-hmm-only").forEach(element => element.classList.toggle("hidden", !bayesianHmm));
  document.querySelectorAll(".flow-only").forEach(element => element.classList.toggle("hidden", !bayesianHmm));
  document.querySelectorAll(".stock-only").forEach(element => element.classList.toggle("hidden", !stock));
  document.querySelectorAll(".fk-only").forEach(element => element.classList.toggle("hidden", !fk));
  document.querySelectorAll(".sector-flow-only").forEach(element => element.classList.toggle("hidden", !sectorFlow));
  document.querySelectorAll(".boundary-flow-only").forEach(element => element.classList.toggle("hidden", !boundaryFlow));
  document.querySelectorAll(".non-boundary-flow-only").forEach(element => element.classList.toggle("hidden", boundaryFlow));
  document.querySelectorAll(".non-sector-flow-only").forEach(element => element.classList.toggle("hidden", sectorFlow || boundaryFlow));
  document.querySelectorAll(".legacy-walk-only").forEach(element => element.classList.toggle("hidden", sectorFlow || boundaryFlow));
  document.querySelectorAll(".neural-only").forEach(element => element.classList.toggle("hidden", !neural));
  document.querySelectorAll(".compute-only").forEach(element => element.classList.toggle("hidden", fk || boundaryFlow));
  const scope = document.querySelector("#training-scope");
  const perSymbol = scope.querySelector('option[value="per-symbol"]');
  perSymbol.disabled = stock || fk || holo || sectorFlow || boundaryFlow;
  if (stock || fk || holo || sectorFlow || boundaryFlow) scope.value = "sector-pooled";
  document.querySelector('[name="profile_protocol"]').value = holo
    ? (holoLite
      ? "price-volume-holographic-lite-v1"
      : "price-volume-holographic-flow-v1")
    : "phase-space-10m-local-flow-v8";
  document.querySelector("#brand-subtitle").textContent = boundaryFlow
    ? "PathEfficiencyFK · Bayesian drift and potential"
    : sectorFlow
    ? "SectorFlow · Bayesian sector dynamics"
    : fk
    ? "FKStock · Bayesian Feynman-Kac model"
    : stock ? "StockTransformer · conservation language model"
      : holo ? `${repair ? repairName : holoLite ? "HoloLite" : "HoloBNHMM"} · 2D Bayesian flow field`
        : "BNHMM · Bayesian hidden-state model";
  document.querySelector("#policy-section-title").textContent = sectorFlow || boundaryFlow
    ? "评估与成本"
    : "仓位策略";
  document.querySelector("#state-section-title").textContent = stock || fk
    ? "每日价格-成交量密度态"
    : holo ? "每日二维全息态" : "每日相空间态";
  if (neural) {
    document.querySelector("#model-section-title").textContent = stock
      ? "第三层 · StockTransformer"
      : "短期记忆与贝叶斯模型";
    document.querySelector("#model-backbone-spec").textContent = stock
      ? "守恒 Token Transformer"
      : holo ? "二维局部场编码器 + Transformer" : "确定性 Transformer";
  }
  const noDataset = (stock || fk || holo || sectorFlow || boundaryFlow) && state.datasets.length === 0;
  document.querySelector("#run-button").disabled = noDataset;
  document.querySelector("#form-error").textContent = noDataset
    ? `${boundaryFlow ? "路径效率 FK" : sectorFlow ? "SectorFlow" : fk ? "FKStock" : stock ? "StockTransformer" : holoLite ? "HoloLite" : "HoloBNHMM"} 需要可用的概念板块数据集`
    : "";
  syncStrategyControlAvailability();
}

function syncStrategyControlAvailability() {
  const form = document.querySelector("#experiment-form");
  form.querySelectorAll("[name]").forEach(element => {
    element.disabled = Boolean(element.closest(".hidden"));
  });
}

function setupDatasets(datasets) {
  state.datasets = datasets || [];
  const select = document.querySelector("#dataset-select");
  select.replaceChildren(...state.datasets.map(dataset => {
    const option = document.createElement("option");
    option.value = dataset.dataset_id;
    option.textContent = dataset.name;
    return option;
  }));
}

function setupRepairVariants(variants) {
  const select = document.querySelector("#repair-variant");
  select.replaceChildren(...(variants || []).map(variant => {
    const option = document.createElement("option");
    option.value = variant.name;
    option.textContent = variant.implemented
      ? `${variant.name} · 阶段 ${variant.stage}`
      : `${variant.name} · 待前序验证`;
    option.title = variant.description || "";
    option.disabled = !variant.implemented;
    return option;
  }));
}

function syncDatasetSelection() {
  const form = document.querySelector("#experiment-form");
  const sectorMode = form.elements.training_scope.value === "sector-pooled";
  const datasetFields = document.querySelector("#sector-dataset-fields");
  const datasetSelect = document.querySelector("#dataset-select");
  const databaseField = form.elements.database_path;
  const symbolsField = form.elements.symbols;
  const chooseButton = document.querySelector("#choose-symbols");
  datasetFields.classList.toggle("hidden", !sectorMode);
  datasetSelect.disabled = !sectorMode;
  databaseField.readOnly = sectorMode;
  symbolsField.readOnly = sectorMode;
  chooseButton.disabled = sectorMode;
  if (!sectorMode) {
    document.querySelector("#dataset-summary").textContent = "";
    return;
  }
  const dataset = state.datasets.find(item => item.dataset_id === datasetSelect.value)
    || state.datasets[0];
  if (!dataset) {
    document.querySelector("#dataset-summary").textContent = "尚未发现板块数据集";
    return;
  }
  datasetSelect.value = dataset.dataset_id;
  databaseField.value = dataset.database_path;
  symbolsField.value = dataset.symbols.join(", ");
  state.symbols = [...dataset.symbols];
  document.querySelector("#dataset-summary").innerHTML = `
    <strong>${escapeHtml(dataset.name)}</strong>
    <span>${dataset.available_member_count} / ${dataset.member_count} 只股票可用</span>
    <span>${escapeHtml(dataset.data_start || "-")} 至 ${escapeHtml(dataset.data_end || "-")}</span>
    <span>${escapeHtml(dataset.source)} · ${escapeHtml(dataset.source_code)}</span>`;
}

async function refreshBoundaryDataStatus() {
  const element = document.querySelector("#boundary-data-status");
  const button = document.querySelector("#boundary-data-build");
  const datasetId = document.querySelector("#dataset-select")?.value;
  if (!datasetId || currentStrategy() !== "boundary-flow-fk") return;
  try {
    const payload = await api(`/api/boundary-data/status/${encodeURIComponent(datasetId)}`);
    const variants = payload.cache?.variants || [];
    const ready = variants.filter(item => item.status === "ready").length;
    const expected = state.config?.boundary_cache_lookbacks?.length || variants.length || 7;
    button.disabled = payload.status === "running";
    if (payload.status === "running") element.textContent = `构造中 · 已就绪 ${ready}/${expected}`;
    else if (payload.status === "failed") element.textContent = `构造失败 · ${payload.error || "请查看日志"}`;
    else if (ready === expected) element.textContent = `${expected} 套数据已就绪 · ${variants[0]?.start || ""} 至 ${variants[0]?.end || ""}`;
    else element.textContent = `已就绪 ${ready}/${expected}，请先构造`;
  } catch (exception) {
    element.textContent = exception.message;
    button.disabled = false;
  }
}

async function buildBoundaryData() {
  const form = document.querySelector("#experiment-form");
  const button = document.querySelector("#boundary-data-build");
  const status = document.querySelector("#boundary-data-status");
  button.disabled = true;
  status.textContent = "正在启动构造任务";
  try {
    await api("/api/boundary-data/build", {
      method: "POST",
      body: JSON.stringify({
        dataset_id: form.elements.dataset_id.value,
        start: form.elements.start.value,
        end: form.elements.end.value,
      }),
    });
    await refreshBoundaryDataStatus();
  } catch (exception) {
    status.textContent = exception.message;
    button.disabled = false;
  }
}

function syncModelConstraints() {
  const strategy = currentStrategy();
  const fk = strategy === "fk-stock";
  const holoLite = strategy === "holo-bnhmm-lite";
  const holo = strategy === "holo-bnhmm" || holoLite || strategy === "holo-repair";
  const velocityField = document.querySelector("[name=velocity_bins]");
  const latentField = document.querySelector("[name=latent_states]");
  const transitionRankField = document.querySelector("[name=transition_rank]");
  const sequenceLengthField = document.querySelector("[name=sequence_length]");
  const strongThresholdField = document.querySelector("[name=strong_threshold_sigma]");
  const standardizationLookbackField = document.querySelector("[name=return_standardization_lookback]");
  const standardizationMinimumField = document.querySelector("[name=return_standardization_min_observations]");
  const widthField = document.querySelector("[name=d_model]");
  const headsField = document.querySelector("[name=heads]");
  const auxiliaryFloorField = document.querySelector("[name=auxiliary_gate_floor]");
  const auxiliaryInitialField = document.querySelector("[name=auxiliary_gate_initial]");
  const velocityBins = Number(velocityField.value || 3);
  const holographicVolumeBins = Number(document.querySelector("[name=holographic_volume_bins]").value || 5);
  const holographicFieldCount = holoLite ? 5 : 8;
  const minimum = holo
    ? holographicVolumeBins * holographicFieldCount
    : 4 + velocityBins ** 2;
  latentField.min = String(minimum);
  if (Number(latentField.value) < minimum) latentField.value = String(minimum);

  const latentStates = Number(latentField.value || minimum);
  transitionRankField.max = String(latentStates);
  if (Number(transitionRankField.value) > latentStates) {
    transitionRankField.value = String(latentStates);
  }

  const channelCount = document.querySelector("#profile-channel-count");
  if (channelCount) {
    channelCount.textContent = `4 + ${velocityBins}² = ${4 + velocityBins ** 2} 通道`;
  }
  const holographicChannelCount = document.querySelector("#holographic-channel-count");
  if (holographicChannelCount) {
    holographicChannelCount.textContent = `${holographicVolumeBins} × ${holographicFieldCount} = ${holographicVolumeBins * holographicFieldCount} 通道`;
  }
  const holographicFieldSpec = document.querySelector("#holographic-field-spec");
  if (holographicFieldSpec) {
    holographicFieldSpec.textContent = holoLite
      ? "密度、价格流、成交量流、散度、旋度"
      : "密度、双流、温度、协方差、散度、旋度";
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
    !fk && width && heads && width % heads !== 0
      ? "状态编码宽度必须能被注意力头数整除"
      : "",
  );
  auxiliaryInitialField.setCustomValidity(
    !fk && Number(auxiliaryInitialField.value) < Number(auxiliaryFloorField.value)
      ? "辅助初始权重不能低于辅助权重下限"
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
  syncStrategyControlAvailability();
  const data = new FormData();
  for (const element of form.querySelectorAll("[name]")) {
    if (element.disabled) continue;
    if (["checkbox", "radio"].includes(element.type) && !element.checked) continue;
    data.append(element.name, element.value);
  }
  const payload = {};
  for (const [name, value] of data.entries()) {
    if (name === "symbols") {
      payload.symbols = value.split(/[\s,;]+/).map(item => item.trim()).filter(Boolean);
    } else if (name === "sector_flow_path_windows") {
      payload[name] = value.split(",").map(item => Number(item.trim())).filter(Number.isFinite);
    } else if (numberFields.has(name)) {
      payload[name] = value === "" && ["max_folds", "factor_process_variance"].includes(name) ? null : Number(value);
    } else {
      payload[name] = value;
    }
  }
  payload.deterministic_training = form.elements.deterministic_training.checked;
  payload.holographic_fields = currentStrategy() === "holo-bnhmm-lite"
    ? ["density", "price_current", "volume_current", "divergence", "vorticity"]
    : ["density", "price_current", "volume_current", "price_temperature", "volume_temperature", "covariance_flux", "divergence", "vorticity"];
  payload.holographic_conv_layers = currentStrategy() === "holo-bnhmm-lite" ? 1 : 2;
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
    state.selectedComparisonId = null;
    await refreshAll();
    setNextExperimentName();
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
    const [jobs, comparisons, gpuPayload] = await Promise.all([
      api("/api/jobs"),
      api("/api/comparisons"),
      api("/api/gpus").catch(() => api("/api/config").then(config => ({ devices: config.devices }))),
    ]);
    state.jobs = jobs;
    state.comparisons = comparisons;
    updateDevices(gpuPayload.devices);
    if (state.selectedComparisonId && !state.comparisons.some(item => item.id === state.selectedComparisonId)) {
      state.selectedComparisonId = null;
    }
    if (!state.selectedComparisonId && !state.jobs.some(job => job.id === state.selectedId)) {
      state.selectedId = state.jobs[0]?.id || null;
    }
    renderJobList();
    renderComparisonList();
    if (state.selectedComparisonId) {
      const comparison = await api(`/api/comparisons/${encodeURIComponent(state.selectedComparisonId)}`);
      renderComparison(comparison);
    } else if (state.selectedId) {
      const job = await api(`/api/jobs/${encodeURIComponent(state.selectedId)}`);
      renderJob(job);
    } else {
      renderEmptyState();
    }
    renderGlobalStatus();
    if (currentStrategy() === "boundary-flow-fk") await refreshBoundaryDataStatus();
  } catch (exception) {
    document.querySelector("#job-subtitle").textContent = exception.message;
  } finally {
    state.polling = false;
  }
}

function renderEmptyState() {
  document.querySelector("#empty-state").classList.remove("hidden");
  document.querySelector("#job-view").classList.add("hidden");
  document.querySelector("#comparison-view").classList.add("hidden");
  document.querySelector("#job-subtitle").textContent = "开始新的实验";
}

function renderJobList() {
  const list = document.querySelector("#job-list");
  document.querySelector("#run-count").textContent = state.jobs.length;
  const activeIds = new Set(
    state.jobs
      .filter(job => ["queued", "running"].includes(job.status))
      .map(job => job.id),
  );
  const existingIds = new Set(state.jobs.map(job => job.id));
  for (const identifier of state.historySelected) {
    if (!existingIds.has(identifier) || activeIds.has(identifier)) {
      state.historySelected.delete(identifier);
    }
  }
  renderHistoryEditor(activeIds);
  list.replaceChildren(...state.jobs.map(job => {
    const strategy = job.request.strategy === "holo-repair"
      ? job.request.repair_variant
      : ({ bnhmm: "BNHMM", "holo-bnhmm": "HoloBNHMM", "holo-bnhmm-lite": "HoloLite", "stock-transformer": "StockTransformer", "fk-stock": "FKStock", "sector-flow-bayes": "SectorFlow", "boundary-flow-fk": "路径效率 FK" })[job.request.strategy] || job.request.strategy;
    const content = `<div class="job-item-head"><strong>${escapeHtml(job.request.name)}</strong><span class="status-badge ${job.status}">${statusText[job.status]}</span></div><div class="job-item-meta">${strategy} · ${formatDate(job.created_at)} · ${job.request.symbols.length} 股</div>`;
    if (state.historyEditing) {
      const label = document.createElement("label");
      const disabled = activeIds.has(job.id) || state.historyDeleting;
      label.className = `job-item selection-mode${state.historySelected.has(job.id) ? " selected" : ""}${disabled ? " disabled" : ""}`;
      label.title = activeIds.has(job.id) ? "运行中的任务不能删除" : "";
      label.innerHTML = `<input class="history-job-checkbox" type="checkbox" ${state.historySelected.has(job.id) ? "checked" : ""} ${disabled ? "disabled" : ""}><div class="job-item-content">${content}</div>`;
      label.querySelector("input").addEventListener("change", event => {
        if (event.target.checked) state.historySelected.add(job.id);
        else state.historySelected.delete(job.id);
        renderJobList();
      });
      return label;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = `job-item${job.id === state.selectedId ? " active" : ""}`;
    button.innerHTML = content;
    button.addEventListener("click", () => {
      state.selectedId = job.id;
      state.selectedComparisonId = null;
      refreshAll();
    });
    return button;
  }));
  icons();
}

function renderComparisonList() {
  document.querySelector("#comparison-count").textContent = state.comparisons.length;
  const list = document.querySelector("#comparison-list");
  list.replaceChildren(...state.comparisons.map(comparison => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `job-item comparison-item${comparison.id === state.selectedComparisonId ? " active" : ""}`;
    const label = comparison.status === "completed"
      ? "已完成"
      : comparison.status === "running" ? "运行中" : comparison.status === "queued" ? "排队中" : "失败";
    button.innerHTML = `<div class="job-item-head"><strong>${escapeHtml(comparison.name)}</strong><span class="status-badge ${comparison.status}">${label}</span></div><div class="job-item-meta">${comparison.completed_count}/${comparison.windows.length} 完成 · ${formatDate(comparison.created_at)}</div>`;
    button.addEventListener("click", () => {
      state.selectedComparisonId = comparison.id;
      state.selectedId = null;
      refreshAll();
    });
    return button;
  }));
}

function renderHistoryEditor(activeIds) {
  const editButton = document.querySelector("#edit-history-button");
  editButton.innerHTML = state.historyEditing
    ? '<i data-lucide="check"></i>'
    : '<i data-lucide="pencil"></i>';
  const editLabel = state.historyEditing ? "完成编辑" : "编辑历史测试";
  editButton.title = editLabel;
  editButton.setAttribute("aria-label", editLabel);
  editButton.disabled = state.historyDeleting;
  const eligible = state.jobs.filter(job => !activeIds.has(job.id));
  const selectedEligible = eligible.filter(job => state.historySelected.has(job.id));
  const deleteButton = document.querySelector("#batch-delete-button");
  deleteButton.disabled = !state.historyEditing || selectedEligible.length === 0 || state.historyDeleting;
  deleteButton.title = state.historyDeleting ? "删除中" : "删除选中的测试";
  deleteButton.setAttribute("aria-label", deleteButton.title);
  const actions = document.querySelector("#history-batch-actions");
  actions.classList.toggle("hidden", !state.historyEditing);
  if (!state.historyEditing) return;
  const selectAll = document.querySelector("#history-select-all");
  selectAll.checked = eligible.length > 0 && selectedEligible.length === eligible.length;
  selectAll.indeterminate = selectedEligible.length > 0 && selectedEligible.length < eligible.length;
  selectAll.disabled = eligible.length === 0 || state.historyDeleting;
  document.querySelector("#history-selected-count").textContent = `已选 ${selectedEligible.length}`;
}

function nextComparisonName() {
  const pattern = /^PathEfficiencyFK-Compare-(\d+)$/i;
  const maximum = state.comparisons.reduce((value, item) => {
    const match = String(item.name || "").match(pattern);
    return match ? Math.max(value, Number(match[1])) : value;
  }, 0);
  return `PathEfficiencyFK-Compare-${String(maximum + 1).padStart(3, "0")}`;
}

async function submitWindowComparison() {
  const error = document.querySelector("#form-error");
  const button = document.querySelector("#run-window-comparison");
  error.textContent = "";
  button.disabled = true;
  try {
    const comparison = await api("/api/comparisons", {
      method: "POST",
      body: JSON.stringify({ name: nextComparisonName(), request: formPayload() }),
    });
    state.selectedComparisonId = comparison.id;
    state.selectedId = null;
    await refreshAll();
    document.querySelector("#config-panel").classList.remove("open");
  } catch (exception) {
    error.textContent = exception.message;
  } finally {
    button.disabled = false;
  }
}

function comparisonStatusLabel(status) {
  return ({ completed: "已完成", running: "运行中", queued: "排队中", failed: "失败", cancelled: "已停止", interrupted: "已中断" })[status] || status;
}

function renderComparison(comparison) {
  document.querySelector("#empty-state").classList.add("hidden");
  document.querySelector("#job-view").classList.add("hidden");
  document.querySelector("#comparison-view").classList.remove("hidden");
  document.querySelector("#comparison-title").textContent = comparison.name;
  document.querySelector("#job-subtitle").textContent = comparison.id;
  const badge = document.querySelector("#comparison-status");
  badge.className = `status-badge ${comparison.status}`;
  badge.textContent = comparisonStatusLabel(comparison.status);
  const windowCount = comparison.windows.length;
  const windowLabels = comparison.windows.map(item => item.window_sessions).join(" / ");
  document.querySelector("#comparison-meta").textContent = `PathEfficiencyFK · ${windowLabels} 日 · ${formatDate(comparison.created_at)}`;
  document.querySelector("#comparison-progress").innerHTML = `<strong>${comparison.completed_count}/${windowCount} 个窗口完成</strong><span>${comparison.finished_count}/${windowCount} 个窗口已经结束</span>`;

  const completed = comparison.windows.filter(item => item.status === "completed" && Object.keys(item.metrics || {}).length);
  const byMetric = (field, direction) => [...completed].sort((left, right) => direction * (Number(right.metrics[field]) - Number(left.metrics[field])))[0];
  const bestReturn = byMetric("net_total_return", 1);
  const lowestDrawdown = byMetric("maximum_drawdown", -1);
  const bestDirection = byMetric("drift_direction_accuracy", 1);
  const bestBalanced = byMetric("balanced_accuracy", 1);
  const conclusion = document.querySelector("#comparison-conclusion");
  if (!completed.length) {
    conclusion.className = "boundary-evaluation-summary caution";
    conclusion.innerHTML = `<span>当前进度</span><strong>正在等待窗口结果</strong><p>六个窗口按顺序运行，完成一个就会立即出现在下方表格和净值图中。</p>`;
  } else {
    const positive = Number(bestReturn.metrics.net_total_return) > 0;
    const weakPrediction = Number(bestBalanced.metrics.balanced_accuracy) < 0.36
      || Number(bestDirection.metrics.drift_direction_accuracy) < 0.55;
    conclusion.className = `boundary-evaluation-summary ${positive && weakPrediction ? "caution" : positive ? "positive" : "negative"}`;
    const title = positive && weakPrediction
      ? `${bestReturn.window_sessions} 日收益较好，但预测优势仍弱`
      : `${bestReturn.window_sessions} 日窗口的成本后收益最高`;
    conclusion.innerHTML = `<span>当前结论</span><strong>${title}</strong><p>已完成 ${completed.length} 个窗口；最高收益 ${formatPercent(bestReturn.metrics.net_total_return)}，最高均衡正确率 ${formatPercent(bestBalanced.metrics.balanced_accuracy)}。均衡正确率接近 33% 时，不能只凭收益认定模型已经稳定。</p>`;
  }
  document.querySelector("#comparison-summary-metrics").innerHTML = [
    boundaryMetric("最高成本后收益", bestReturn ? `${bestReturn.window_sessions} 日 · ${formatPercent(bestReturn.metrics.net_total_return)}` : "等待结果"),
    boundaryMetric("最低最大回撤", lowestDrawdown ? `${lowestDrawdown.window_sessions} 日 · ${formatPercent(lowestDrawdown.metrics.maximum_drawdown)}` : "等待结果"),
    boundaryMetric("最高方向正确率", bestDirection ? `${bestDirection.window_sessions} 日 · ${formatPercent(bestDirection.metrics.drift_direction_accuracy)}` : "等待结果"),
    boundaryMetric("最高均衡正确率", bestBalanced ? `${bestBalanced.window_sessions} 日 · ${formatPercent(bestBalanced.metrics.balanced_accuracy)}` : "等待结果"),
  ].join("");

  document.querySelector("#comparison-window-table").innerHTML = comparison.windows.map(item => {
    const metrics = item.metrics || {};
    const action = item.status === "completed"
      ? `<button class="mini-button" type="button" data-comparison-job="${escapeHtml(item.job_id)}">查看完整结果</button>`
      : item.error ? `<span class="negative-text">${escapeHtml(item.error)}</span>` : "—";
    return `<tr><td><strong>${item.window_sessions} 日</strong></td><td><span class="status-badge ${item.status}">${comparisonStatusLabel(item.status)}</span></td><td>${formatPercent(metrics.net_total_return)}</td><td>${formatPercent(metrics.maximum_drawdown)}</td><td>${formatPercent(metrics.drift_direction_accuracy)}</td><td>${formatPercent(metrics.balanced_accuracy)}</td><td>${formatPercent(metrics.average_exposure)}</td><td>${formatPercent(metrics.average_turnover)}</td><td>${action}</td></tr>`;
  }).join("");
  document.querySelectorAll("[data-comparison-job]").forEach(button => {
    button.addEventListener("click", () => openComparisonChild(button.dataset.comparisonJob));
  });
  drawComparisonEquity(comparison.windows);
  icons();
}

async function openComparisonChild(jobId) {
  state.selectedId = jobId;
  state.selectedComparisonId = null;
  const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
  renderJobList();
  renderComparisonList();
  renderJob(job);
}

async function deleteSelectedComparison() {
  if (!state.selectedComparisonId) return;
  if (!window.confirm("确定删除这组窗口对比吗？所有原始回测结果会继续保留。")) return;
  await api(`/api/comparisons/${encodeURIComponent(state.selectedComparisonId)}`, { method: "DELETE" });
  state.selectedComparisonId = null;
  await refreshAll();
}

function drawComparisonEquity(windows) {
  const canvas = document.querySelector("#comparison-equity-canvas");
  const context = canvas.getContext("2d");
  const colors = { 120: "#8c3f62", 90: "#176b87", 60: "#2f7d4f", 40: "#a96f16", 30: "#7a5aa6", 20: "#b64b45", 10: "#52636a" };
  const series = windows
    .filter(item => item.equity_curve?.length)
    .map(item => ({ window: item.window_sessions, values: item.equity_curve.map(point => Number(point.value)) }));
  context.clearRect(0, 0, canvas.width, canvas.height);
  document.querySelector("#comparison-equity-legend").innerHTML = series
    .map(item => `<span><i style="background:${colors[item.window]}"></i>${item.window} 日</span>`)
    .join("");
  if (!series.length) {
    context.fillStyle = "#6d787d";
    context.font = "20px sans-serif";
    context.fillText("等待窗口净值结果", 40, 60);
    return;
  }
  const padding = { left: 62, right: 24, top: 30, bottom: 38 };
  const allValues = series.flatMap(item => item.values);
  const minimum = Math.min(1, ...allValues);
  const maximum = Math.max(1, ...allValues);
  const span = Math.max(maximum - minimum, 1e-9);
  const y = value => padding.top + (maximum - value) * (canvas.height - padding.top - padding.bottom) / span;
  context.strokeStyle = "#d9dfe1";
  context.lineWidth = 1;
  for (let row = 0; row <= 4; row += 1) {
    const py = padding.top + row * (canvas.height - padding.top - padding.bottom) / 4;
    context.beginPath(); context.moveTo(padding.left, py); context.lineTo(canvas.width - padding.right, py); context.stroke();
  }
  for (const item of series) {
    const x = index => padding.left + index * (canvas.width - padding.left - padding.right) / Math.max(item.values.length - 1, 1);
    context.strokeStyle = colors[item.window];
    context.lineWidth = 3;
    context.beginPath();
    item.values.forEach((value, index) => index ? context.lineTo(x(index), y(value)) : context.moveTo(x(index), y(value)));
    context.stroke();
  }
  context.fillStyle = "#34444a";
  context.font = "16px sans-serif";
  context.fillText(maximum.toFixed(3), 8, padding.top + 5);
  context.fillText(minimum.toFixed(3), 8, canvas.height - padding.bottom + 5);
}

function renderGlobalStatus() {
  const active = state.jobs.filter(job => ["queued", "running"].includes(job.status)).length;
  const badge = document.querySelector("#global-status");
  badge.className = `status-badge ${active ? "running" : "neutral"}`;
  badge.textContent = active ? `${active} 个任务` : "空闲";
}

function renderJob(job) {
  document.querySelector("#empty-state").classList.add("hidden");
  document.querySelector("#comparison-view").classList.add("hidden");
  document.querySelector("#job-view").classList.remove("hidden");
  document.querySelector("#run-title").textContent = job.request.name;
  document.querySelector("#job-subtitle").textContent = job.id;
  const badge = document.querySelector("#run-status");
  badge.className = `status-badge ${job.status}`;
  badge.textContent = statusText[job.status];
  const stock = job.request.strategy === "stock-transformer";
  const fk = job.request.strategy === "fk-stock";
  const sectorFlow = job.request.strategy === "sector-flow-bayes";
  const boundaryFlow = job.request.strategy === "boundary-flow-fk";
  const holoLite = job.request.strategy === "holo-bnhmm-lite";
  const repair = job.request.strategy === "holo-repair";
  const holo = job.request.strategy === "holo-bnhmm" || holoLite || repair;
  const dataset = state.datasets.find(
    item => item.dataset_id === job.request.dataset_id,
  );
  if (boundaryFlow) {
    document.querySelector("#run-meta").textContent = `${job.request.start} → ${job.request.end} · ${dataset?.name || job.request.dataset_id || "板块"} · 路径效率 FK · 两支流月度冻结 · ${job.request.minimum_train_sessions} 个成熟训练日 · ${formatInteger(job.request.boundary_fk_paths)} 条 FK 路径 · CPU`;
  } else if (sectorFlow) {
    const water = job.summary?.latest_water_state || {};
    document.querySelector("#run-meta").textContent = `${job.request.start} → ${job.request.end} · ${dataset?.name || job.request.dataset_id || "板块"} · 1分钟同步 · 最大滞后 ${job.request.maximum_lag_minutes} 分钟 · 水势 ${formatNumber(water.flow_strength, 3)} · ${job.request.normalization_lookback_sessions}日因果归一化`;
  } else if (fk) {
    const fields = job.summary?.folds?.at(-1)?.outer_field_diagnostics;
    const fieldText = fields
      ? `μ=${formatNumber(fields.effective_drift_mean, 4)} · σ²=${formatNumber(fields.diffusion_variance_mean, 4)}`
      : "单特征贝叶斯生成元";
    document.querySelector("#run-meta").textContent = `${job.request.start} → ${job.request.end} · ${dataset?.name || job.request.dataset_id || "板块"} · 滚动扩散 → 贝叶斯后验 · ${fieldText} · 1 个有效输入`;
  } else if (stock) {
    const law = job.summary?.folds?.at(-1)?.conservation_law;
    const tokens = job.summary?.folds?.at(-1)?.token_diagnostics;
    const lawText = law
      ? `D=${formatNumber(law.diffusion, 4)} · v=${formatNumber(law.drift, 4)} · noise=${formatNumber(law.noise_scale, 4)}`
      : `${job.request.conservation_grid_points}点守恒网格`;
    const tokenText = tokens
      ? `日均 ${formatNumber(tokens.mean_tokens_per_day, 1)} Token · ${formatNumber(tokens.mean_sentences_per_day, 1)} 句`
      : `最多 ${job.request.maximum_tokens} Token/日`;
    document.querySelector("#run-meta").textContent = `${job.request.start} → ${job.request.end} · ${dataset?.name || job.request.dataset_id || "板块"} · 随机连续性方程 → 守恒分词/断句 → StockTransformer · ${lawText} · ${tokenText} · ${job.request.sequence_length}日记忆 · ${job.request.device}`;
  } else {
  const stateMode = job.request.state_timeframe === "mixed"
    ? "小时 + 日状态"
    : "日状态（1分钟线生成）";
  const profileMode = ["price-volume-holographic-flow-v1", "price-volume-holographic-lite-v1"].includes(job.request.profile_protocol)
    ? `${job.request.bin_width_bps} BP × ${job.request.holographic_volume_bins} 档成交活跃度 · 二维散度/旋度场`
    : job.request.profile_protocol === "phase-space-10m-local-flow-v8"
    ? `${job.request.bin_width_bps} BP 局部联合速度场 · ${job.request.normalization_lookback_sessions}日归一化`
    : ["phase-space-10m-joint-return-v7", "phase-space-joint-return-v6"].includes(job.request.profile_protocol)
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
  const modelName = repair
    ? job.request.repair_variant
    : holo ? (holoLite ? "HoloLite" : "HoloBNHMM") : "BNHMM";
  const modelScope = ({
    "per-symbol": `逐标的独立 ${modelName}`,
    "sector-pooled": `${dataset?.name || job.request.dataset_id || "板块"}联合 ${modelName}`,
    "shared-legacy": "多标的共享模型（旧）",
  })[job.request.training_scope] || job.request.training_scope;
  const latentStates = job.request.latent_states ?? 10;
  const featureMode = holo ? (holoLite ? "二维核心场轻量模型" : "二维价格-成交活跃度全息流场") : ({
    "gated-full": "价格主干 + 门控成交量/联合速度",
    "price-only": "仅价格",
    "price-volume": "价格 + 成交量",
    "price-velocity": "价格 + Z2 流态（推荐）",
    "legacy-entangled": "旧混合特征",
  })[job.request.feature_mode] || "旧混合特征";
  const auxiliaryMode = holo
    ? `${job.request.holographic_fields?.length ?? (holoLite ? 5 : 8)} 个局部场 · 平滑 ${Number(job.request.holographic_smoothing ?? 0.75).toFixed(2)}`
    : job.request.feature_mode === "price-only"
    ? "辅助分支关闭"
    : `辅助门 ${Number(job.request.auxiliary_gate_initial ?? 0.05).toFixed(2)} 初始 / ${Number(job.request.auxiliary_gate_floor ?? 0).toFixed(2)} 下限 / ${job.request.auxiliary_warmup_epochs ?? 0} 轮全权重`;
  const outputMode = ["price-volume-holographic-flow-v1", "price-volume-holographic-lite-v1", "phase-space-10m-local-flow-v8", "phase-space-10m-joint-return-v7", "phase-space-joint-return-v6"].includes(job.request.profile_protocol)
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
  document.querySelector("#run-meta").textContent = `${job.request.start} → ${job.request.end} · ${job.request.symbols.join(", ")} · ${modelScope} · ${featureMode} · ${auxiliaryMode} · ${latentStates} 隐状态 → ${outputHead} · ${outputMode} · ${thresholdText} · ${stateMode} · ${profileMode} · ${job.request.device}`;
  }
  renderJobGpuMemory(job);
  const active = ["queued", "running"].includes(job.status);
  document.querySelector("#cancel-button").classList.toggle("hidden", !active);
  document.querySelector("#progress-band").classList.toggle("hidden", !active);
  document.querySelector("#progress-label").textContent = job.status === "queued"
    ? (boundaryFlow ? "等待路径效率 FK 任务" : sectorFlow ? "等待 SectorFlow 任务" : fk ? "等待 FKStock 任务" : "等待 GPU 任务")
    : boundaryFlow ? "路径效率 FK 月度 Walk-forward 运行中" : sectorFlow ? "SectorFlow 月度 Walk-forward 运行中" : fk ? "FKStock 最小模型 Walk-forward 运行中" : stock ? "StockTransformer 三层 Walk-forward 运行中" : holo ? `${holoLite ? "HoloLite" : "HoloBNHMM"} 二维流场 Walk-forward 运行中` : "BNHMM Walk-forward 运行中";
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
  const sectorFlow = job.request.strategy === "sector-flow-bayes";
  const boundaryFlow = job.request.strategy === "boundary-flow-fk";
  document.querySelector("#sector-flow-results").classList.toggle("hidden", !sectorFlow);
  document.querySelector("#boundary-flow-results").classList.toggle("hidden", !boundaryFlow);
  const activePanel = document.querySelector(".tab.active")?.dataset.tab || "overview";
  document.querySelectorAll(".standard-result-only").forEach(element => {
    if (sectorFlow || boundaryFlow) element.classList.add("hidden");
    else if (element.matches(".tab-panel")) element.classList.toggle("hidden", element.dataset.panel !== activePanel);
    else element.classList.remove("hidden");
  });
  if (boundaryFlow) {
    renderBoundaryFlowResult(job);
    return;
  }
  if (sectorFlow) {
    renderSectorFlowResult(job);
    return;
  }
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

function renderBoundaryFlowResult(job) {
  const status = document.querySelector("#boundary-flow-result-status");
  const evaluationUrl = job.artifacts?.["boundary-policy-evaluation.json"];
  const predictionsUrl = job.artifacts?.["feynman-kac-predictions.json"];
  const posteriorUrl = job.artifacts?.["boundary-model-posterior.json"];
  const residualUrl = job.artifacts?.["boundary-residual-analysis.json"];
  const convectionUrl = job.artifacts?.["convection-potential-evaluation.json"];
  const downloads = {
    "path-efficiency-fk-report.md": "中文报告",
    "path-efficiency-convection-potential-report.md": "一日反向流量外势报告",
    "boundary-policy-evaluation.json": "策略评估",
    "convection-potential-evaluation.json": "一日反向流量外势统计",
    "feynman-kac-predictions.json": "样本外预测",
    "boundary-model-posterior.json": "模型后验",
    "boundary-residual-analysis.json": "残差分析",
    "boundary-daily-states.json": "每日支流状态",
    "branch-memberships.json": "月度支流成分",
    "run-summary.json": "运行摘要",
  };
  document.querySelector("#boundary-flow-downloads").innerHTML = Object.entries(downloads)
    .filter(([name]) => job.artifacts?.[name])
    .map(([name, label]) => `<a href="${job.artifacts[name]}" download>${label}</a>`)
    .join("");

  const cached = state.boundaryResultCache.get(job.id);
  if (cached?.error) {
    status.textContent = `结果读取失败：${cached.error}`;
    return;
  }
  if (cached?.stages && cached?.predictions) {
    paintBoundaryFlowResult(job, cached);
    return;
  }
  if (!evaluationUrl || !predictionsUrl) {
    status.textContent = job.status === "failed"
      ? "运行失败，请查看日志"
      : ["queued", "running"].includes(job.status) ? "路径效率 FK 样本外计算中" : "暂无可读取的结果制品";
    document.querySelector("#boundary-probabilities").replaceChildren();
    document.querySelector("#boundary-dynamics").replaceChildren();
    document.querySelector("#boundary-prediction-table").replaceChildren();
    document.querySelector("#boundary-gate-reason").textContent = "";
    return;
  }
  if (cached?.loading) {
    status.textContent = "正在读取样本外结果";
    return;
  }

  state.boundaryResultCache.set(job.id, { loading: true });
  status.textContent = "正在读取样本外结果";
  Promise.all([
    api(evaluationUrl),
    api(predictionsUrl),
    posteriorUrl ? api(posteriorUrl) : Promise.resolve([]),
    residualUrl ? api(residualUrl) : Promise.resolve({}),
    convectionUrl ? api(convectionUrl) : Promise.resolve({}),
  ]).then(([stages, predictions, posterior, residual, convection]) => {
    const payload = {
      stages: Array.isArray(stages) ? stages : [],
      predictions: Array.isArray(predictions) ? predictions : [],
      posterior: Array.isArray(posterior) ? posterior : [],
      residual: residual || {},
      convection: convection || {},
    };
    state.boundaryResultCache.set(job.id, payload);
    if (state.selectedId === job.id) paintBoundaryFlowResult(job, payload);
  }).catch(exception => {
    state.boundaryResultCache.set(job.id, { error: exception.message });
    if (state.selectedId === job.id) status.textContent = `结果读取失败：${exception.message}`;
  });
}

function switchBoundaryResultTab(name, jobId = state.selectedId) {
  if (jobId) state.boundaryTabByJob.set(jobId, name);
  document.querySelectorAll("[data-boundary-tab]").forEach(button => {
    const selected = button.dataset.boundaryTab === name;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  document.querySelectorAll("[data-boundary-panel]").forEach(panel => {
    panel.classList.toggle("hidden", panel.dataset.boundaryPanel !== name);
  });
}

function boundaryMetric(label, value, note = "") {
  return `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong>${note ? `<small>${escapeHtml(note)}</small>` : ""}</div>`;
}

function boundaryImprovementVerdict(metrics = {}, baselineMetrics = {}) {
  const strategyReturn = Number(metrics.net_total_return);
  const baselineReturn = Number(baselineMetrics.net_total_return);
  const strategyDrawdown = Math.abs(Number(metrics.maximum_drawdown));
  const baselineDrawdown = Math.abs(Number(baselineMetrics.maximum_drawdown));
  const returnDelta = strategyReturn - baselineReturn;
  const drawdownImprovement = baselineDrawdown - strategyDrawdown;

  if (![strategyReturn, baselineReturn, strategyDrawdown, baselineDrawdown].every(Number.isFinite)) {
    return {
      title: "基线数据不完整，暂时无法判断是否改进",
      level: "caution",
      returnDelta,
      drawdownImprovement,
      passed: false,
    };
  }
  if (returnDelta > 0) {
    return {
      title: "对流势能版本超过纯路径效率基线",
      level: "positive",
      returnDelta,
      drawdownImprovement,
      passed: true,
    };
  }
  return {
    title: drawdownImprovement > 0
      ? "对流势能降低了回撤，但没有提高收益，未通过改进标准"
      : "对流势能未超过纯路径效率基线，未通过改进标准",
    level: drawdownImprovement > 0 ? "caution" : "negative",
    returnDelta,
    drawdownImprovement,
    passed: false,
  };
}

function paintBoundaryFlowResult(job, payload) {
  const { stages, predictions: allPredictions, posterior, residual, convection } = payload;
  const oneDayPotentialModel = "PathEfficiencyFK-60D-OneDayPotential";
  const baselineModel = "PathEfficiencyFK-60D-Baseline";
  const stage = stages.find(item => item.model_name === oneDayPotentialModel)
    || stages.find(item => item.model_name !== baselineModel)
    || {};
  const potentialModel = stage.model_name || oneDayPotentialModel;
  const isOneDayPotential = potentialModel === oneDayPotentialModel;
  const predictions = allPredictions.filter(item => item.model_version === potentialModel);
  const baselinePredictions = allPredictions.filter(item => item.model_version === baselineModel);
  const baselineStage = stages.find(item => item.model_name === baselineModel) || {};
  document.querySelector("#boundary-result-title").textContent = isOneDayPotential
    ? "60 日路径效率 + 一日反向流量外势 FK"
    : "60 日路径效率 + 对流势能 FK（旧实验）";
  const metrics = stage.metrics || {};
  const baselineMetrics = baselineStage.metrics || {};
  const latest = predictions.at(-1);
  document.querySelector("#boundary-flow-result-status").textContent =
    `${formatInteger(predictions.length)} 个样本外交易日 · ${isOneDayPotential ? "一日反向流量 C1 外势" : "历史外势实验"}`;

  const verdict = boundaryImprovementVerdict(metrics, baselineMetrics);
  document.querySelector("#boundary-evaluation-summary").innerHTML = `
    <span>本次结论</span><strong>${verdict.title}</strong>
    <p>基于 ${formatInteger(metrics.observations)} 个真实样本外交易日。相对基线收益 ${formatPercent(verdict.returnDelta)}，回撤改善 ${formatPercent(verdict.drawdownImprovement)}；${verdict.passed ? "通过" : "未通过"}改进标准。</p>`;
  document.querySelector("#boundary-evaluation-summary").className = `boundary-evaluation-summary ${verdict.level}`;
  document.querySelector("#boundary-evaluation-metrics").innerHTML = [
    boundaryMetric("基线成本后收益", formatPercent(baselineMetrics.net_total_return), "60 日纯路径效率 FK"),
    boundaryMetric("成本后总收益", formatPercent(metrics.net_total_return), "扣除换手成本"),
    boundaryMetric(isOneDayPotential ? "平均一日外势" : "平均历史外势", formatNumber(convection?.mean_effective_potential, 4), isOneDayPotential ? "β0 背景势 + β1·标准化 C1" : "旧实验结果"),
    boundaryMetric("平均趋势存活率", formatPercent(convection?.mean_trend_survival_probability), "FK 对漂移的保留比例"),
    boundaryMetric("最大回撤", formatPercent(metrics.maximum_drawdown), "历史高点后的最大跌幅"),
    boundaryMetric("方向判断正确率", formatPercent(metrics.drift_direction_accuracy), "上涨或下跌方向"),
    boundaryMetric("平均实际仓位", formatPercent(metrics.average_exposure), "整个测试期平均"),
  ].join("");

  if (!latest) {
    document.querySelector("#boundary-probabilities").innerHTML = '<p class="boundary-empty">没有样本外预测。</p>';
    document.querySelector("#boundary-dynamics").replaceChildren();
    document.querySelector("#boundary-gate-reason").textContent = "没有产生可展示的支流预测。";
    return;
  }

  const states = ["空", "平", "多"];
  const branches = latest.branches || [];
  document.querySelector("#boundary-probabilities").innerHTML = branches.map(branch => {
    const probabilities = branch.state_probabilities || [];
    return `<div class="boundary-probability ${branch.direction === "bear" ? "bear" : branch.direction === "bull" ? "bull" : "neutral"}">
      <strong>支流 ${branch.branch} · ${{ bear: "空", flat: "平", bull: "多" }[branch.direction] || "—"}</strong>
      ${states.map((label, index) => `<span>${label} ${formatPercent(probabilities[index])}</span>`).join("")}
    </div>`;
  }).join("");

  const dynamics = branches.flatMap(branch => {
    const features = branch.convection_potential_features || {};
    const values = [
    [`支流 ${branch.branch} 预期涨跌`, formatPercent(branch.expected_drift)],
    [`支流 ${branch.branch} 一日外势`, formatNumber(branch.expected_effective_potential, 4)],
    [`支流 ${branch.branch} 趋势延续可信度`, formatPercent(branch.survival_confidence)],
    [`支流 ${branch.branch} 看多概率`, formatPercent(branch.bullish_drift_probability)],
    [`支流 ${branch.branch} 趋势强度`, formatNumber(branch.drift_signal_to_noise, 3)],
    [`支流 ${branch.branch} C1 一日反向流量`, formatPercent(features.reverse_flow_1d)],
    ];
    return values;
  });
  document.querySelector("#boundary-dynamics").innerHTML = dynamics
    .map(([label, value]) => `<div><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");

  const branchTargetExposures = latest.branch_target_exposures || [0, 0];
  const branchFlatSignalCounts = latest.branch_flat_signal_counts || [0, 0];
  const totalExposure = branchTargetExposures.reduce((total, value) => total + value, 0);
  const holdings = latest.selected_symbols?.length
    ? `当前持有 ${latest.selected_symbols.length} 只股票 · 目标仓位 ${formatPercent(totalExposure)} · 两支流仓位 ${branchTargetExposures.map(formatPercent).join(" / ")}`
    : `当前空仓 · 连续看平 ${branchFlatSignalCounts.join(" / ")}`;
  const trace = `信号 ${latest.signal_time || "—"} · 训练截止 ${latest.training_cutoff || "—"} · 下一交易日 ${latest.target_time || "—"}`;
  const convectionTrace = branches.map(branch =>
    `支流${branch.branch} 势能 ${formatNumber(branch.expected_effective_potential, 4)} / 存活率 ${formatPercent(branch.survival_confidence)}`
  ).join(" · ");
  document.querySelector("#boundary-gate-reason").innerHTML = `<strong>${escapeHtml(holdings)}</strong><span>${escapeHtml(trace)}</span><span>${escapeHtml(convectionTrace)}</span>`;

  document.querySelector("#boundary-prediction-table").innerHTML = predictions.slice(-60).reverse().map(item => {
    const itemBranches = item.branches || [];
    const exposure = (item.branch_target_exposures || []).reduce((sum, value) => sum + Number(value || 0), 0);
    const label = branch => ({ bear: "空", flat: "平", bull: "多" })[branch?.direction] || "—";
    const convectionText = itemBranches.map(branch => {
      const features = branch.convection_potential_features || {};
      return formatPercent(features.reverse_flow_1d);
    }).join(" · ");
    return `<tr><td>${escapeHtml(String(item.signal_time || "—").slice(0, 10))}</td><td>${label(itemBranches[0])}</td><td>${label(itemBranches[1])}</td><td>${escapeHtml(convectionText)}</td><td>${formatPercent(exposure)}</td><td>${formatInteger(item.selected_symbols?.length || 0)}</td><td class="${Number(item.gross_strategy_return) >= 0 ? "positive-text" : "negative-text"}">${formatPercent(item.gross_strategy_return)}</td></tr>`;
  }).join("");

  document.querySelector("#boundary-model-quality").innerHTML = [
    boundaryMetric("概率整体误差", formatNumber(metrics.brier_score, 4), "越小越好"),
    boundaryMetric("概率校准偏差", formatPercent(metrics.ece), "预测概率与实际频率的差"),
    boundaryMetric("三状态均衡正确率", formatPercent(metrics.balanced_accuracy), "避免多数状态掩盖结果"),
    boundaryMetric("训练成熟样本", formatInteger(stage.minimum_fit_rows), "每支流用于后验估计"),
  ].join("");

  paintBoundaryReturnComparison(job, metrics, predictions, baselineMetrics, baselinePredictions, isOneDayPotential);
  const modelResidual = Array.isArray(residual)
    ? residual.find(item => item.model_name === potentialModel) || {}
    : residual;
  paintBoundaryResidualAnalysis(metrics, modelResidual, predictions);
  document.querySelector("#boundary-run-summary").innerHTML = [
    boundaryMetric("数据集", job.request.dataset_id || "—"),
    boundaryMetric("测试区间", `${job.request.start} 至 ${job.request.end}`),
    boundaryMetric("股票数量", `${formatInteger(job.request.symbols?.length)} 只`),
    boundaryMetric("支流历史", `${job.request.topology_lookback_sessions} 日`),
    boundaryMetric("外势输入", isOneDayPotential ? "一日反向流量 C1" : "历史实验输入"),
    boundaryMetric("FK 模拟路径", formatInteger(job.request.boundary_fk_paths)),
    boundaryMetric("模型版本", latest.model_version || posterior?.[0]?.model_name || "PathEfficiencyFK"),
    boundaryMetric("训练截止", latest.training_cutoff || "—"),
    boundaryMetric("运行耗时", `${formatNumber(job.summary?.elapsed_seconds, 2)} 秒`),
  ].join("");
  switchBoundaryResultTab(state.boundaryTabByJob.get(job.id) || "evaluation", job.id);
}

function boundaryEquitySeries(job, predictions) {
  const costRate = Number(job.request.transaction_cost_bps || 0) / 10000;
  let equity = 1;
  return predictions.map(item => {
    equity *= 1 + Number(item.gross_strategy_return || 0) - costRate * Number(item.turnover || 0);
    return equity;
  });
}

function paintBoundaryReturnComparison(job, metrics, predictions, baselineMetrics, baselinePredictions, isOneDayPotential) {
  const strategyEquity = boundaryEquitySeries(job, predictions);
  const baselineEquity = boundaryEquitySeries(job, baselinePredictions);
  const wins = predictions.filter(item => Number(item.gross_strategy_return) > 0).length;
  document.querySelector("#boundary-return-metrics").innerHTML = [
    boundaryMetric(isOneDayPotential ? "一日反向流量外势 FK 收益" : "历史外势 FK 收益", formatPercent(metrics.net_total_return)),
    boundaryMetric("纯路径效率基线", formatPercent(baselineMetrics.net_total_return)),
    boundaryMetric("相对基线", formatPercent(Number(metrics.net_total_return || 0) - Number(baselineMetrics.net_total_return || 0))),
    boundaryMetric("最大回撤", formatPercent(metrics.maximum_drawdown)),
    boundaryMetric("盈利交易日", formatPercent(predictions.length ? wins / predictions.length : 0)),
    boundaryMetric("平均换手", formatPercent(metrics.average_turnover)),
  ].join("");
  drawBoundaryComparison(
    document.querySelector("#boundary-return-canvas"),
    [
      { values: strategyEquity, color: "#176b87" },
      { values: baselineEquity, color: "#69767c" },
    ],
    1,
  );
}

function drawBoundaryComparison(canvas, series, baseline) {
  const allValues = series.flatMap(item => item.values).filter(Number.isFinite);
  if (!canvas || !allValues.length) return;
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padding = { left: 58, right: 22, top: 22, bottom: 35 };
  const minimum = Math.min(...allValues, baseline);
  const maximum = Math.max(...allValues, baseline);
  const span = Math.max(maximum - minimum, 1e-9);
  const y = value => padding.top + (maximum - value) * (height - padding.top - padding.bottom) / span;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#d9dfe1";
  context.lineWidth = 1;
  for (let row = 0; row <= 4; row += 1) {
    const py = padding.top + row * (height - padding.top - padding.bottom) / 4;
    context.beginPath(); context.moveTo(padding.left, py); context.lineTo(width - padding.right, py); context.stroke();
  }
  context.strokeStyle = "#aab2b6";
  context.beginPath(); context.moveTo(padding.left, y(baseline)); context.lineTo(width - padding.right, y(baseline)); context.stroke();
  series.forEach(item => {
    if (!item.values.length) return;
    const x = index => padding.left + index * (width - padding.left - padding.right) / Math.max(item.values.length - 1, 1);
    context.strokeStyle = item.color;
    context.lineWidth = 3;
    context.beginPath();
    item.values.forEach((value, index) => index ? context.lineTo(x(index), y(value)) : context.moveTo(x(index), y(value)));
    context.stroke();
  });
  context.fillStyle = "#34444a";
  context.font = "16px sans-serif";
  context.fillText(maximum.toFixed(3), 4, padding.top + 5);
  context.fillText(minimum.toFixed(3), 4, height - padding.bottom + 5);
}

function paintBoundaryReturnAnalysis(job, metrics, predictions) {
  const costRate = Number(job.request.transaction_cost_bps || 0) / 10000;
  let equity = 1;
  let peak = 1;
  const series = predictions.map(item => {
    equity *= 1 + Number(item.gross_strategy_return || 0) - costRate * Number(item.turnover || 0);
    peak = Math.max(peak, equity);
    return { value: equity, drawdown: equity / peak - 1 };
  });
  const wins = predictions.filter(item => Number(item.gross_strategy_return) > 0).length;
  document.querySelector("#boundary-return-metrics").innerHTML = [
    boundaryMetric("成本后总收益", formatPercent(metrics.net_total_return)),
    boundaryMetric("最大回撤", formatPercent(metrics.maximum_drawdown)),
    boundaryMetric("盈利交易日", formatPercent(predictions.length ? wins / predictions.length : 0)),
    boundaryMetric("平均换手", formatPercent(metrics.average_turnover)),
  ].join("");
  drawBoundaryLine(document.querySelector("#boundary-return-canvas"), series.map(item => item.value), "策略净值", "#176b87", 1);
}

function paintBoundaryResidualAnalysis(metrics, residual, predictions) {
  const values = predictions.map(item => {
    const branches = item.branches || [];
    const errors = branches.map(branch => Number(branch.actual_return) - Number(branch.expected_drift)).filter(Number.isFinite);
    return errors.length ? errors.reduce((sum, value) => sum + value, 0) / errors.length : 0;
  });
  const mean = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
  const mae = values.length ? values.reduce((sum, value) => sum + Math.abs(value), 0) / values.length : 0;
  document.querySelector("#boundary-residual-metrics").innerHTML = [
    boundaryMetric("平均预测偏差", formatPercent(mean), "应接近 0"),
    boundaryMetric("平均绝对误差", formatPercent(mae), "越小越好"),
    boundaryMetric("误差连续性", formatNumber(residual.probability_residual_autocorrelation ?? metrics.probability_residual_autocorrelation, 3), "接近 0 表示没有连续偏错"),
    boundaryMetric("样本数量", formatInteger(values.length)),
  ].join("");
  drawBoundaryLine(document.querySelector("#boundary-residual-canvas"), values, "实际收益 - 预测收益", "#b64b45", 0);
}

function drawBoundaryLine(canvas, values, label, color, baseline) {
  if (!canvas || !values.length) return;
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padding = { left: 58, right: 22, top: 35, bottom: 35 };
  const minimum = Math.min(...values, baseline);
  const maximum = Math.max(...values, baseline);
  const span = Math.max(maximum - minimum, 1e-9);
  const x = index => padding.left + index * (width - padding.left - padding.right) / Math.max(values.length - 1, 1);
  const y = value => padding.top + (maximum - value) * (height - padding.top - padding.bottom) / span;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#d9dfe1";
  context.lineWidth = 1;
  for (let row = 0; row <= 4; row += 1) {
    const py = padding.top + row * (height - padding.top - padding.bottom) / 4;
    context.beginPath(); context.moveTo(padding.left, py); context.lineTo(width - padding.right, py); context.stroke();
  }
  context.strokeStyle = "#889499";
  context.beginPath(); context.moveTo(padding.left, y(baseline)); context.lineTo(width - padding.right, y(baseline)); context.stroke();
  context.strokeStyle = color;
  context.lineWidth = 3;
  context.beginPath();
  values.forEach((value, index) => index ? context.lineTo(x(index), y(value)) : context.moveTo(x(index), y(value)));
  context.stroke();
  context.fillStyle = "#34444a";
  context.font = "20px sans-serif";
  context.fillText(label, padding.left, 24);
  context.font = "16px sans-serif";
  context.fillText(maximum.toFixed(3), 4, padding.top + 5);
  context.fillText(minimum.toFixed(3), 4, height - padding.bottom + 5);
}

function renderSectorFlowResult(job) {
  const summary = job.summary || {};
  const regimeMode = summary.protocol === "river-regime-v3";
  const container = document.querySelector("#sector-flow-results");
  container.classList.toggle("regime-mode", regimeMode);
  document.querySelector("#sector-regime-results").classList.toggle("hidden", !regimeMode);
  if (regimeMode) {
    renderSectorRegimeResult(job);
    return;
  }
  const metrics = summary.metrics || {};
  const water = summary.latest_water_state || {};
  const status = document.querySelector("#sector-flow-result-status");
  const riverButton = document.querySelector("#open-river-view");
  const riverAvailable = job.status === "completed" && ["river-topology-v2", "river-regime-v3"].includes(summary.protocol);
  riverButton.classList.toggle("hidden", !riverAvailable);
  riverButton.href = riverAvailable ? `/jobs/${encodeURIComponent(job.id)}/river` : "#";
  status.textContent = job.status === "completed"
    ? `${formatInteger(summary.oos_predictions)} 条样本外预测`
    : job.status === "failed" ? "运行失败，请查看日志" : "水势计算中";
  const waterValues = [
    ["方向", formatNumber(water.direction, 3)],
    ["水势强度", formatNumber(water.flow_strength, 3)],
    ["活跃度", formatNumber(water.activity, 2)],
    ["广度", formatPercent(water.breadth)],
    ["同步性", formatPercent(water.coherence)],
    ["不确定性", formatNumber(water.uncertainty, 3)],
  ];
  document.querySelector("#sector-flow-water-state").innerHTML = waterValues
    .map(([label, value]) => `<div class="water-state-item"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");

  const ranking = summary.latest_ranking || [];
  document.querySelector("#sector-flow-ranking-table").innerHTML = ranking.map(item => `<tr>
    <td><strong>${escapeHtml(item.symbol)}</strong></td>
    <td>${formatPercent(item.source_proximity_probability)}</td>
    <td>${formatPercent(item.lead_or_response_probability)}</td>
    <td>${formatPercent(item.coupling_probability)}</td>
    <td>${formatPercent(item.efficiency_probability)}</td>
    <td>${formatPercent(item.low_residual_probability)}</td>
    <td>${formatPercent(item.activity_confirmation_probability)}</td>
    <td>${formatPercent(item.leader_probability)}</td>
    <td>${formatPercent(item.response_probability)}</td>
  </tr>`).join("");

  const baselineLabels = {
    random: "随机",
    momentum: "一分钟动量",
    relative_volume: "相对成交量",
    path_efficiency: "路径效率",
    static_beta: "静态 Beta",
  };
  document.querySelector("#sector-flow-baseline-table").innerHTML = Object.entries(summary.baselines || {}).map(([name, value]) => `<tr>
    <td>${baselineLabels[name] || escapeHtml(name)}</td>
    <td>${formatNumber(value.rank_ic, 4)}</td>
    <td>${formatPercent(value.top_bottom_return)}</td>
    <td>${formatPercent(value.net_top_bottom_return)}</td>
    <td>${formatPercent(value.turnover)}</td>
  </tr>`).join("");

  setImage("sector-flow-image", job.artifacts?.["sector-flow.png"], `?v=${encodeURIComponent(job.finished_at || Date.now())}`);
  const downloads = {
    "sector-flow-v2-report.md": "v2 完整报告",
    "structure-evaluation.json": "结构评估",
    "sector-flow-report.md": "完整报告",
    "evaluation.json": "评估数据",
    "ranking.csv": "排名明细",
    "predictions.csv": "样本外预测",
  };
  document.querySelector("#sector-flow-downloads").innerHTML = Object.entries(downloads)
    .filter(([name]) => job.artifacts?.[name])
    .map(([name, label]) => `<a href="${job.artifacts[name]}" download>${label}</a>`)
    .join("");

  document.querySelector("#metrics-grid").innerHTML = [
    ["Brier", formatNumber(metrics.brier_score, 4)],
    ["Log Loss", formatNumber(metrics.log_loss, 4)],
    ["Rank IC", formatNumber(metrics.rank_ic, 4)],
    ["成本后 Top-Bottom", formatPercent(metrics.net_top_bottom_return)],
    ["后验覆盖率", formatPercent(metrics.posterior_coverage)],
    ["换手率", formatPercent(metrics.turnover)],
  ].map(([label, value]) => `<div class="metric-card"><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function renderSectorRegimeResult(job) {
  const summary = job.summary || {};
  const regime = summary.regime_metrics || {};
  const policy = summary.policy_metrics || {};
  const latest = summary.latest_regime || {};
  const latestPolicy = summary.latest_policy || {};
  const probabilities = latest.probability_mean || [0, 0, 0];
  const labels = ["看跌", "平稳", "看涨"];
  const tones = ["bear", "flat", "bull"];
  const predicted = Number.isInteger(latest.predicted_state) ? latest.predicted_state : -1;
  const status = document.querySelector("#sector-flow-result-status");
  status.textContent = job.status === "completed"
    ? `${formatInteger(summary.oos_regime_predictions)} 次板块样本外预测`
    : job.status === "failed" ? "运行失败，请查看日志" : "正在训练板块状态模型";
  const riverButton = document.querySelector("#open-river-view");
  const riverAvailable = job.status === "completed";
  riverButton.classList.toggle("hidden", !riverAvailable);
  riverButton.href = riverAvailable ? `/jobs/${encodeURIComponent(job.id)}/river` : "#";

  document.querySelector("#sector-regime-probabilities").innerHTML = probabilities.map((value, index) => `
    <div class="regime-probability ${tones[index]}${predicted === index ? " active" : ""}">
      <div><span>${labels[index]}</span><strong>${formatPercent(value)}</strong></div>
      <div class="probability-track"><span style="width:${Math.max(0, Math.min(100, Number(value) * 100))}%"></span></div>
    </div>
  `).join("");

  document.querySelector("#sector-regime-baselines").innerHTML = [
    ["模型准确率", formatPercent(regime.accuracy)],
    ["平衡准确率", formatPercent(regime.balanced_accuracy)],
    ["多数类基线", formatPercent(regime.majority_baseline_accuracy)],
    ["昨日状态延续", formatPercent(regime.persistence_baseline_accuracy)],
    ["Macro F1", formatNumber(regime.macro_f1, 4)],
  ].map(([label, value]) => `<div class="regime-stat"><span>${label}</span><strong>${value}</strong></div>`).join("");

  document.querySelector("#sector-regime-policy").innerHTML = [
    ["当前动作", latestPolicy.selected_symbols?.length ? `买入 ${latestPolicy.selected_symbols.join("、")}` : "空仓"],
    ["开仓比例", formatPercent(policy.exposure_rate)],
    ["策略总收益", formatPercent(policy.total_return)],
    ["最大回撤", formatPercent(policy.maximum_drawdown)],
    ["开仓日胜率", formatPercent(policy.win_rate)],
  ].map(([label, value]) => `<div class="regime-stat"><span>${label}</span><strong>${escapeHtml(String(value))}</strong></div>`).join("");

  document.querySelector("#metrics-grid").innerHTML = [
    ["平衡准确率", formatPercent(regime.balanced_accuracy)],
    ["多数类基线", formatPercent(regime.majority_baseline_accuracy)],
    ["Brier", formatNumber(regime.brier_score, 4)],
    ["Log Loss", formatNumber(regime.log_loss, 4)],
    ["策略收益", formatPercent(policy.total_return)],
    ["实际开仓", formatInteger(policy.exposed_sessions)],
  ].map(([label, value]) => `<div class="metric-card"><span>${label}</span><strong>${value}</strong></div>`).join("");

  const downloads = {
    "sector-regime-report.md": "中文报告",
    "sector-regime-predictions.json": "板块预测",
    "sector-regime-evaluation.json": "板块评估",
    "sector-policy-evaluation.json": "策略评估",
    "sector-state-forecasts.npz": "状态图预测",
  };
  document.querySelector("#sector-flow-downloads").innerHTML = Object.entries(downloads)
    .filter(([name]) => job.artifacts?.[name])
    .map(([name, label]) => `<a href="${job.artifacts[name]}" download>${label}</a>`)
    .join("");
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
  const isGpu = !["fk-stock", "boundary-flow-fk"].includes(job.request.strategy) && String(job.request.device).startsWith("cuda:");
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
  if (summary.model_scope === "sector-pooled") detail += " 各股票独立构造日态并独立归一化，只在严格按日期切分后共享板块模型。";
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
  table.innerHTML = folds.map(fold => `<tr><td>${escapeHtml(fold.symbol || "共享（旧）")}</td><td>${fold.fold + 1}</td><td>${fold.tuning_start}</td><td>${fold.test_start} → ${fold.test_end}</td><td>${fold.selected_epochs ?? "闭式"}</td><td>${fold.pretest_sequences ?? fold.generator_fit_rows ?? "—"}</td><td>${fold.outer_test_predictions}</td><td>${formatNumber(fold.tuning_best_validation_loss ?? fold.tuning_metrics?.log_loss, 4)}</td></tr>`).join("");
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
  const names = { "evaluation.json": "评估数据", "predictions.json": "OOS 预测", "fk-diagnostics.json": "FK 场诊断", "run-summary.json": "运行摘要", "repair-stage-report.md": "阶段报告", "sector-flow-report.md": "水势报告", "ranking.csv": "排名明细", "predictions.csv": "预测明细" };
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
  document.querySelector("#run-window-comparison").addEventListener("click", submitWindowComparison);
  document.querySelector("#delete-comparison-button").addEventListener("click", deleteSelectedComparison);
  document.querySelector("#device-select").addEventListener("change", updateDeviceNote);
  document.querySelectorAll("[name=strategy]").forEach(field => field.addEventListener("change", changeStrategy));
  document.querySelector("#repair-variant").addEventListener("change", () => {
    setNextExperimentName("holo-repair");
    syncStrategySelection();
  });
  for (const name of ["velocity_bins", "holographic_volume_bins", "latent_states", "transition_rank", "sequence_length", "strong_threshold_sigma", "return_standardization_lookback", "return_standardization_min_observations", "d_model", "heads", "auxiliary_gate_floor", "auxiliary_gate_initial"]) {
    document.querySelector(`[name=${name}]`).addEventListener("input", syncModelConstraints);
  }
  document.querySelector("#refresh-button").addEventListener("click", refreshAll);
  document.querySelector("#edit-history-button").addEventListener("click", () => {
    state.historyEditing = !state.historyEditing;
    if (!state.historyEditing) state.historySelected.clear();
    renderJobList();
  });
  document.querySelector("#history-select-all").addEventListener("change", event => {
    const eligible = state.jobs.filter(job => !["queued", "running"].includes(job.status));
    state.historySelected.clear();
    if (event.target.checked) eligible.forEach(job => state.historySelected.add(job.id));
    renderJobList();
  });
  document.querySelector("#batch-delete-button").addEventListener("click", deleteSelectedJobs);
  document.querySelector("#choose-symbols").addEventListener("click", openSymbolDialog);
  document.querySelector("#training-scope").addEventListener("change", syncDatasetSelection);
  document.querySelector("#dataset-select").addEventListener("change", syncDatasetSelection);
  document.querySelector("#boundary-data-build").addEventListener("click", buildBoundaryData);
  document.querySelectorAll("[data-boundary-tab]").forEach(button => {
    button.addEventListener("click", () => switchBoundaryResultTab(button.dataset.boundaryTab));
  });
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

async function deleteSelectedJobs() {
  const jobs = state.jobs.filter(
    job => state.historySelected.has(job.id) && !["queued", "running"].includes(job.status),
  );
  if (!jobs.length || state.historyDeleting) return;
  if (!window.confirm(`确定删除选中的 ${jobs.length} 个测试及其全部检查点和报告吗？此操作不可撤销。`)) return;
  state.historyDeleting = true;
  renderJobList();
  const results = await Promise.allSettled(
    jobs.map(job => api(`/api/jobs/${encodeURIComponent(job.id)}`, { method: "DELETE" })),
  );
  const deletedIds = new Set(
    results
      .map((result, index) => result.status === "fulfilled" ? jobs[index].id : null)
      .filter(Boolean),
  );
  state.jobs = state.jobs.filter(job => !deletedIds.has(job.id));
  deletedIds.forEach(identifier => {
    state.historySelected.delete(identifier);
    state.resultScopeByJob.delete(identifier);
    state.boundaryResultCache.delete(identifier);
    state.boundaryTabByJob.delete(identifier);
    for (const key of state.symbolReportCache.keys()) {
      if (key.startsWith(`${identifier}:`)) state.symbolReportCache.delete(key);
    }
  });
  if (deletedIds.has(state.selectedId)) state.selectedId = state.jobs[0]?.id || null;
  const failures = results.filter(result => result.status === "rejected");
  state.historyDeleting = false;
  if (!failures.length) {
    state.historyEditing = false;
    state.historySelected.clear();
  }
  renderJobList();
  await refreshAll();
  if (failures.length) {
    window.alert(`已删除 ${deletedIds.size} 个，${failures.length} 个删除失败：${failures[0].reason.message}`);
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
  setupDatasets(state.config.datasets);
  setupRepairVariants(state.config.repair_variants);
  fillForm(state.config.defaults);
  setupDevices(state.config.devices);
  await refreshAll();
  setNextExperimentName();
  icons();
  setInterval(refreshAll, 2500);
}

initialize().catch(exception => {
  document.querySelector("#job-subtitle").textContent = exception.message;
});
