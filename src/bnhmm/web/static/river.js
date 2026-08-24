const riverState = {
  jobId: jobIdFromPath(),
  manifest: null,
  month: null,
  session: null,
  topology: null,
  slice: null,
  migration: null,
  selectedSymbol: null,
  channel: "velocity",
};

function jobIdFromPath() {
  const parts = window.location.pathname.split("/").filter(Boolean);
  return parts.length >= 3 ? decodeURIComponent(parts[1]) : "";
}

async function api(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    const message = await response.json().catch(() => ({}));
    throw new Error(message.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function endpoint(suffix) {
  return `/api/jobs/${encodeURIComponent(riverState.jobId)}/river/${suffix}`;
}

async function boot() {
  if (!riverState.jobId) return showError("任务编号无效");
  bindControls();
  try {
    riverState.manifest = await api(endpoint("manifest"));
    populateSelectors();
    const topologies = riverState.manifest.topologies || [];
    const slices = riverState.manifest.slices || [];
    if (!topologies.length || !slices.length) throw new Error("该任务没有可视化制品");
    riverState.month = topologies.at(-1).month;
    riverState.session = slices.filter(item => item.topology_month === riverState.month).at(-1)?.session || slices.at(-1).session;
    document.querySelector("#month-select").value = riverState.month;
    syncSessionOptions();
    document.querySelector("#session-select").value = riverState.session;
    await Promise.all([loadTopology(riverState.month), loadSlice(riverState.session), loadMigration()]);
    document.querySelector("#river-status").textContent = `${slices.length} 个交易日 · 月内排序冻结`;
  } catch (error) {
    showError(error.message);
  }
  window.lucide?.createIcons();
}

function bindControls() {
  document.querySelectorAll(".river-tab").forEach(button => button.addEventListener("click", () => {
    document.querySelectorAll(".river-tab").forEach(item => item.classList.toggle("active", item === button));
    document.querySelectorAll(".river-panel").forEach(panel => panel.classList.toggle("hidden", panel.id !== button.dataset.view));
  }));
  document.querySelector("#month-select").addEventListener("change", async event => {
    riverState.month = event.target.value;
    syncSessionOptions();
    riverState.session = document.querySelector("#session-select").value;
    try {
      await Promise.all([loadTopology(riverState.month), loadSlice(riverState.session)]);
    } catch (error) {
      showError(error.message);
    }
  });
  document.querySelector("#session-select").addEventListener("change", async event => {
    riverState.session = event.target.value;
    try { await loadSlice(riverState.session); } catch (error) { showError(error.message); }
  });
  document.querySelectorAll('input[name="channel"]').forEach(input => input.addEventListener("change", event => {
    riverState.channel = event.target.value;
    renderHeatmap();
  }));
  document.querySelector("#density-opacity").addEventListener("change", renderHeatmap);
  const canvas = document.querySelector("#river-heatmap");
  canvas.addEventListener("mousemove", showHeatmapTooltip);
  canvas.addEventListener("mouseleave", () => document.querySelector("#heatmap-tooltip").classList.add("hidden"));
  canvas.addEventListener("click", selectHeatmapRow);
}

function populateSelectors() {
  const months = (riverState.manifest.topologies || []).map(item => item.month);
  document.querySelector("#month-select").innerHTML = months.map(month => `<option value="${escapeHtml(month)}">${escapeHtml(month)}</option>`).join("");
}

function syncSessionOptions() {
  const sessions = (riverState.manifest.slices || []).filter(item => item.topology_month === riverState.month);
  document.querySelector("#session-select").innerHTML = sessions.map(item => `<option value="${escapeHtml(item.session)}">${escapeHtml(item.session)}</option>`).join("");
}

async function loadTopology(month) {
  riverState.topology = await api(endpoint(`topology/${encodeURIComponent(month)}`));
  document.querySelector("#topology-cutoff").textContent = `拓扑截止 ${formatTime(riverState.topology.training_cutoff)}`;
  renderTopology(riverState.topology);
  renderStructure(riverState.topology);
  if (riverState.slice) renderHeatmap();
}

async function loadSlice(session) {
  riverState.slice = await api(endpoint(`slice/${encodeURIComponent(session)}`));
  clearError();
  renderHeatmap();
  if (riverState.selectedSymbol) renderStockInspector(riverState.selectedSymbol);
}

async function loadMigration() {
  riverState.migration = await api(endpoint("migration"));
  renderMigration(riverState.migration);
}

function renderHeatmap() {
  const item = riverState.slice;
  if (!item) return;
  const canvas = document.querySelector("#river-heatmap");
  const context = canvas.getContext("2d");
  const rows = item.symbols.length;
  const columns = item.timestamps.length;
  const labelWidth = 76;
  const cellWidth = Math.max(3, Math.floor(1050 / Math.max(columns, 1)));
  const rowHeight = 18;
  const width = labelWidth + columns * cellWidth;
  const height = Math.max(rows * rowHeight, 120);
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.fillStyle = "#eef1f2";
  context.fillRect(0, 0, width, height);
  context.font = '10px "Segoe UI", "Microsoft YaHei", sans-serif';
  context.textBaseline = "middle";
  const values = item[riverState.channel] || item.velocity;
  const scale = robustScale(values, item.valid);
  const opacityEnabled = document.querySelector("#density-opacity").checked && riverState.channel !== "activity";
  const bandBySymbol = topologyBandMap();
  let previousBand = null;
  for (let row = 0; row < rows; row += 1) {
    const y = row * rowHeight;
    const selected = riverState.selectedSymbol === item.symbols[row];
    context.fillStyle = selected ? "#e8f4ef" : "#ffffff";
    context.fillRect(0, y, labelWidth, rowHeight);
    context.fillStyle = selected ? "#066348" : "#344249";
    context.fillText(item.symbols[row], 7, y + rowHeight / 2);
    const band = bandBySymbol.get(item.symbols[row]);
    if (row > 0 && band !== previousBand) {
      context.fillStyle = "#637078";
      context.fillRect(0, y, width, 1);
    }
    previousBand = band;
    for (let column = 0; column < columns; column += 1) {
      const valid = Boolean(item.valid?.[row]?.[column]);
      const value = values?.[row]?.[column];
      const activity = item.activity?.[row]?.[column];
      context.fillStyle = cellColor(value, scale, valid, riverState.channel, opacityEnabled ? activity : null);
      context.fillRect(labelWidth + column * cellWidth, y, cellWidth, rowHeight - 1);
    }
  }
  canvas.dataset.labelWidth = String(labelWidth);
  canvas.dataset.cellWidth = String(cellWidth);
  canvas.dataset.rowHeight = String(rowHeight);
}

function robustScale(matrix, valid) {
  const values = [];
  (matrix || []).forEach((row, rowIndex) => row.forEach((value, column) => {
    if (valid?.[rowIndex]?.[column] && Number.isFinite(Number(value))) values.push(Math.abs(Number(value)));
  }));
  values.sort((a, b) => a - b);
  return values[Math.floor(values.length * 0.95)] || 1;
}

function cellColor(value, scale, valid, channel, activity) {
  if (!valid || value === null || !Number.isFinite(Number(value))) return "#d7dcde";
  if (channel === "activity") {
    const level = Math.max(0, Math.min(1, Number(value) / 3));
    const shade = Math.round(242 - 128 * level);
    return `rgb(${shade}, ${Math.round(shade + 12)}, ${Math.round(shade + 18)})`;
  }
  const magnitude = Math.min(1, Math.abs(Number(value)) / Math.max(scale, 1e-12));
  const confidence = activity === null || !Number.isFinite(Number(activity)) ? 1 : Math.max(0.25, Math.min(1, Number(activity) / 2));
  const weight = magnitude * confidence;
  const neutral = [238, 240, 239];
  const target = Number(value) >= 0 ? [208, 64, 57] : [19, 129, 99];
  return `rgb(${target.map((part, index) => Math.round(neutral[index] + (part - neutral[index]) * weight)).join(",")})`;
}

function heatmapCell(event) {
  const canvas = event.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const labelWidth = Number(canvas.dataset.labelWidth || 76);
  const cellWidth = Number(canvas.dataset.cellWidth || 3);
  const rowHeight = Number(canvas.dataset.rowHeight || 18);
  const row = Math.floor(y / rowHeight);
  const column = Math.floor((x - labelWidth) / cellWidth);
  return { row, column, overLabel: x < labelWidth };
}

function showHeatmapTooltip(event) {
  const { row, column, overLabel } = heatmapCell(event);
  const item = riverState.slice;
  const tooltip = document.querySelector("#heatmap-tooltip");
  if (!item || overLabel || row < 0 || row >= item.symbols.length || column < 0 || column >= item.timestamps.length) {
    tooltip.classList.add("hidden");
    return;
  }
  const value = item[riverState.channel]?.[row]?.[column];
  const valid = Boolean(item.valid?.[row]?.[column]);
  tooltip.innerHTML = `<strong>${escapeHtml(item.symbols[row])}</strong><br>${escapeHtml(formatTime(item.timestamps[column]))}<br>数值 ${valid ? formatNumber(value, 4) : "缺失"}<br>成交密度 ${formatNumber(item.activity?.[row]?.[column], 3)}`;
  tooltip.style.left = `${Math.min(event.clientX + 12, window.innerWidth - 182)}px`;
  tooltip.style.top = `${Math.min(event.clientY + 12, window.innerHeight - 92)}px`;
  tooltip.classList.remove("hidden");
}

function selectHeatmapRow(event) {
  const { row } = heatmapCell(event);
  if (!riverState.slice || row < 0 || row >= riverState.slice.symbols.length) return;
  riverState.selectedSymbol = riverState.slice.symbols[row];
  renderHeatmap();
  renderStockInspector(riverState.selectedSymbol);
}

async function renderStockInspector(symbol) {
  const inspector = document.querySelector("#stock-inspector");
  const row = riverState.slice?.symbols.indexOf(symbol) ?? -1;
  const topologyIndex = riverState.topology?.symbols.indexOf(symbol) ?? -1;
  if (row < 0 || topologyIndex < 0) return;
  const velocity = riverState.slice.velocity[row].filter(Number.isFinite);
  const residual = riverState.slice.residual_velocity[row].filter(Number.isFinite);
  const displacement = velocity.reduce((sum, value) => sum + value, 0);
  const residualEnergy = residual.length ? residual.reduce((sum, value) => sum + value * value, 0) / residual.length : 0;
  inspector.innerHTML = `<div class="inspector-head"><h2>${escapeHtml(symbol)}</h2><span>流带 ${formatInteger(riverState.topology.band_ids[topologyIndex])}</span></div>
    <div class="inspector-grid">
      <div><span>日内位移</span><strong>${formatNumber(displacement, 3)}</strong></div>
      <div><span>残差能量</span><strong>${formatNumber(residualEnergy, 3)}</strong></div>
      <div><span>凝聚度</span><strong>${formatNumber(riverState.topology.cohesion?.[topologyIndex], 3)}</strong></div>
      <div><span>中心性</span><strong>${formatNumber(riverState.topology.centrality?.[topologyIndex], 3)}</strong></div>
    </div><div class="history-list"><h3>近期样本外预测</h3><div id="symbol-history">正在读取</div></div>`;
  try {
    const history = await api(endpoint(`symbol/${encodeURIComponent(symbol)}`));
    if (riverState.selectedSymbol !== symbol) return;
    const rows = (history.predictions || []).slice(-8).reverse();
    document.querySelector("#symbol-history").innerHTML = rows.length ? rows.map(item => `<div class="history-row"><span>${escapeHtml(String(item.signal_time).slice(0, 10))}</span><strong>${formatNumber(item.expected_relative_return, 4)}</strong></div>`).join("") : '<span class="muted">暂无预测记录</span>';
  } catch (error) {
    document.querySelector("#symbol-history").textContent = error.message;
  }
}

function renderTopology(topology) {
  const svg = document.querySelector("#topology-view");
  clearSvg(svg);
  const width = 1080;
  const height = Math.max(620, topology.symbols.length * 22 + 80);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const positions = topologyPositions(topology, width, height);
  const edges = [];
  for (let first = 0; first < topology.symbols.length; first += 1) {
    for (let second = first + 1; second < topology.symbols.length; second += 1) {
      const weight = Number(topology.graph?.[first]?.[second] || 0);
      if (weight > 0.04) edges.push({ first, second, weight });
    }
  }
  edges.sort((a, b) => b.weight - a.weight).slice(0, 360).forEach(edge => {
    const line = svgNode("line", { x1: positions[edge.first].x, y1: positions[edge.first].y, x2: positions[edge.second].x, y2: positions[edge.second].y, stroke: "#aebbc0", "stroke-width": 0.4 + edge.weight * 1.8, opacity: 0.35 + edge.weight * 0.35 });
    svg.append(line);
  });
  topology.symbols.forEach((symbol, index) => {
    const position = positions[index];
    const group = svgNode("g", { tabindex: "0", role: "button", "aria-label": symbol });
    const circle = svgNode("circle", { cx: position.x, cy: position.y, r: 5 + Number(topology.centrality?.[index] || 0) * 5, fill: bandColor(topology.band_ids[index]), stroke: "#ffffff", "stroke-width": 1.5 });
    const label = svgNode("text", { x: position.x + 9, y: position.y + 3, fill: "#344249", "font-size": 9 });
    label.textContent = symbol;
    group.append(circle, label);
    group.addEventListener("click", () => { riverState.selectedSymbol = symbol; document.querySelector('[data-view="heatmap-panel"]').click(); renderHeatmap(); renderStockInspector(symbol); });
    svg.append(group);
  });
  document.querySelector("#topology-summary").textContent = `${topology.symbols.length} 只股票 · ${new Set(topology.band_ids.filter(value => value >= 0)).size} 个稳定流带`;
}

function topologyPositions(topology, width, height) {
  const bands = [...new Set(topology.band_ids.filter(value => value >= 0))].sort((a, b) => a - b);
  const bandIndex = new Map(bands.map((band, index) => [band, index]));
  const counts = new Map();
  return topology.symbols.map((symbol, index) => {
    const band = topology.band_ids[index];
    const column = bandIndex.has(band) ? bandIndex.get(band) : bands.length;
    const order = counts.get(column) || 0;
    counts.set(column, order + 1);
    return { x: 90 + column * ((width - 180) / Math.max(bands.length, 1)), y: 55 + order * 25 + (column % 2) * 11 };
  });
}

function renderMigration(migration) {
  const svg = document.querySelector("#migration-view");
  clearSvg(svg);
  const months = migration.months || [];
  const transitions = migration.transitions || [];
  const width = Math.max(900, months.length * 180);
  const maxRow = Math.max(20, ...transitions.flatMap(item => [item.from_row, item.to_row]));
  const height = Math.max(620, maxRow * 13 + 90);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  months.forEach((month, index) => {
    const x = 80 + index * ((width - 160) / Math.max(months.length - 1, 1));
    svg.append(svgNode("line", { x1: x, y1: 50, x2: x, y2: height - 30, stroke: "#dce2e5", "stroke-width": 1 }));
    const label = svgNode("text", { x, y: 30, "text-anchor": "middle", fill: "#68757b", "font-size": 10 });
    label.textContent = month;
    svg.append(label);
  });
  transitions.forEach(item => {
    const fromIndex = months.indexOf(item.from_month);
    const toIndex = months.indexOf(item.to_month);
    if (fromIndex < 0 || toIndex < 0) return;
    const x1 = 80 + fromIndex * ((width - 160) / Math.max(months.length - 1, 1));
    const x2 = 80 + toIndex * ((width - 160) / Math.max(months.length - 1, 1));
    const y1 = 55 + item.from_row * 13;
    const y2 = 55 + item.to_row * 13;
    const path = svgNode("path", { d: `M ${x1} ${y1} C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2} ${y2}`, fill: "none", stroke: bandColor(item.to_band), "stroke-width": 1, opacity: 0.45 });
    svg.append(path);
  });
  document.querySelector("#migration-summary").textContent = months.length > 1 ? `${months.length} 个月 · ${transitions.length} 条迁移轨迹` : "需要至少两个月拓扑才会出现迁移轨迹";
}

function renderStructure(topology) {
  const assigned = topology.assigned.filter(Boolean).length;
  const bands = new Set(topology.band_ids.filter(value => value >= 0)).size;
  const cohesion = mean(topology.cohesion.filter((_, index) => topology.assigned[index]));
  const centrality = mean(topology.centrality.filter((_, index) => topology.assigned[index]));
  document.querySelector("#structure-metrics").innerHTML = [
    ["覆盖股票", `${assigned} / ${topology.symbols.length}`],
    ["稳定流带", formatInteger(bands)],
    ["平均凝聚度", formatNumber(cohesion, 3)],
    ["平均中心性", formatNumber(centrality, 3)],
  ].map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
  document.querySelector("#structure-table").innerHTML = topology.row_order.map(index => `<tr><td><strong>${escapeHtml(topology.symbols[index])}</strong></td><td>${formatInteger(topology.band_ids[index])}</td><td>${formatNumber(topology.cohesion[index], 4)}</td><td>${formatNumber(topology.centrality[index], 4)}</td><td>${topology.assigned[index] ? "是" : "否"}</td></tr>`).join("");
}

function topologyBandMap() {
  const result = new Map();
  riverState.topology?.symbols.forEach((symbol, index) => result.set(symbol, riverState.topology.band_ids[index]));
  return result;
}

function svgNode(name, attributes) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function clearSvg(svg) { while (svg.firstChild) svg.firstChild.remove(); }

function bandColor(band) {
  const colors = ["#087f5b", "#276e9e", "#c83f39", "#a15c00", "#6f5aa8", "#387f8a", "#8a5d4d", "#59666d"];
  return band < 0 ? "#9aa5aa" : colors[Math.abs(Number(band)) % colors.length];
}

function mean(values) { return values.length ? values.reduce((sum, value) => sum + Number(value || 0), 0) / values.length : 0; }
function formatInteger(value) { return Number.isFinite(Number(value)) ? Math.round(Number(value)).toLocaleString("zh-CN") : "--"; }
function formatNumber(value, digits = 3) { return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "--"; }
function formatTime(value) { return value ? String(value).replace("T", " ").replace(/\.\d+$/, "") : "--"; }
function escapeHtml(value) { return String(value ?? "").replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]); }
function showError(message) { const element = document.querySelector("#river-error"); element.textContent = message; element.classList.remove("hidden"); document.querySelector("#river-status").textContent = "制品读取失败"; }
function clearError() { document.querySelector("#river-error").classList.add("hidden"); }

window.addEventListener("DOMContentLoaded", boot);
