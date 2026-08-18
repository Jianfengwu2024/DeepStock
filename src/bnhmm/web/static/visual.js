const visualState = {
  page: 1,
  pageSize: 24,
  total: 0,
  summary: null,
  items: [],
};

function visualIcons() {
  if (window.lucide) window.lucide.createIcons();
}

async function visualApi(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function escapeVisual(value) {
  return String(value).replace(/[&<>"]/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[character]));
}

function percent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number >= 0 ? "+" : ""}${(number * 100).toFixed(2)}%` : "—";
}

async function loadSummary() {
  const summary = await visualApi("/api/visual-dataset/summary");
  visualState.summary = summary;
  document.querySelector("#dataset-subtitle").textContent = summary.available
    ? `${summary.data_start} 至 ${summary.data_end} · 昨收锚定 · 因果滚动归一化`
    : "图像集尚未生成";
  document.querySelector("#visual-summary").innerHTML = [
    ["图像", Number(summary.image_count || 0).toLocaleString("zh-CN")],
    ["股票", Number(summary.symbol_count || 0).toLocaleString("zh-CN")],
    ["状态", summary.status === "complete" ? "完整" : summary.available ? "生成中" : "不可用"],
  ].map(([label, value]) => `<span><strong>${escapeVisual(value)}</strong>${label}</span>`).join("");
  const tensor = summary.tensor || {};
  document.querySelector("#tensor-size").textContent = `${tensor.width || 840} × ${tensor.height || 21}`;
  const select = document.querySelector("#visual-symbol");
  select.replaceChildren(
    new Option("全部股票", ""),
    ...(summary.symbols || []).map(symbol => new Option(symbol, symbol)),
  );
}

function queryString() {
  const form = new FormData(document.querySelector("#visual-filter"));
  visualState.pageSize = Number(form.get("page_size") || 24);
  const params = new URLSearchParams({
    page: String(visualState.page),
    page_size: String(visualState.pageSize),
  });
  for (const name of ["symbol", "start", "end"]) {
    const value = String(form.get(name) || "").trim();
    if (value) params.set(name, value);
  }
  return params.toString();
}

async function loadImages() {
  const grid = document.querySelector("#visual-grid");
  grid.setAttribute("aria-busy", "true");
  const result = await visualApi(`/api/visual-dataset/images?${queryString()}`);
  visualState.total = result.total;
  visualState.items = result.items;
  renderImages();
  grid.removeAttribute("aria-busy");
}

function renderImages() {
  const grid = document.querySelector("#visual-grid");
  const empty = document.querySelector("#visual-empty");
  grid.innerHTML = visualState.items.map((item, index) => {
    const change = Number(item.close_return);
    const direction = change >= 0 ? "up" : "down";
    return `<figure class="state-figure" data-index="${index}" tabindex="0">
      <img src="${escapeVisual(item.image_url)}" loading="lazy" alt="${escapeVisual(item.symbol)} ${escapeVisual(item.session)} 二维状态">
      <figcaption><div><strong>${escapeVisual(item.symbol)} · ${escapeVisual(item.session)}</strong><small>O ${Number(item.open).toFixed(3)} · H ${Number(item.high).toFixed(3)} · L ${Number(item.low).toFixed(3)} · C ${Number(item.close).toFixed(3)}</small></div><span class="state-change ${direction}">${percent(change)}</span></figcaption>
    </figure>`;
  }).join("");
  empty.classList.toggle("hidden", visualState.items.length > 0);
  document.querySelectorAll(".state-figure").forEach(figure => {
    const open = () => openVisual(Number(figure.dataset.index));
    figure.addEventListener("click", open);
    figure.addEventListener("keydown", event => { if (event.key === "Enter") open(); });
  });
  const pages = Math.max(1, Math.ceil(visualState.total / visualState.pageSize));
  document.querySelector("#page-status").textContent = `第 ${visualState.page} / ${pages} 页 · ${visualState.total.toLocaleString("zh-CN")} 张`;
  document.querySelector("#previous-page").disabled = visualState.page <= 1;
  document.querySelector("#next-page").disabled = visualState.page >= pages;
}

function openVisual(index) {
  const item = visualState.items[index];
  if (!item) return;
  document.querySelector("#dialog-title").textContent = `${item.symbol} · ${item.session}`;
  document.querySelector("#dialog-meta").textContent = `昨收 ${Number(item.previous_close).toFixed(3)} · O ${percent(item.open_return)} · H ${percent(item.high_return)} · L ${percent(item.low_return)} · C ${percent(item.close_return)}`;
  document.querySelector("#dialog-image").src = item.image_url;
  document.querySelector("#visual-dialog").showModal();
}

function setupVisualEvents() {
  document.querySelector("#visual-filter").addEventListener("submit", event => {
    event.preventDefault();
    visualState.page = 1;
    loadImages().catch(showVisualError);
  });
  document.querySelector("#previous-page").addEventListener("click", () => {
    visualState.page -= 1;
    loadImages().then(() => window.scrollTo({ top: 0, behavior: "smooth" })).catch(showVisualError);
  });
  document.querySelector("#next-page").addEventListener("click", () => {
    visualState.page += 1;
    loadImages().then(() => window.scrollTo({ top: 0, behavior: "smooth" })).catch(showVisualError);
  });
  document.querySelector("#close-visual-dialog").addEventListener("click", () => document.querySelector("#visual-dialog").close());
}

function showVisualError(error) {
  document.querySelector("#dataset-subtitle").textContent = error.message;
}

async function initializeVisual() {
  setupVisualEvents();
  await loadSummary();
  await loadImages();
  visualIcons();
}

initializeVisual().catch(showVisualError);
