// Código Sintético — Agent Observability dashboard
// Vanilla JS, sin build step. Habla exclusivamente con la API HTTP
// documentada en /docs — nada aquí es "falso": cada número viene de un
// endpoint real.

const API = "/api/v1";
const API_KEY_STORAGE = "sintetico_api_key";

function getApiKey() {
  try { return localStorage.getItem(API_KEY_STORAGE) || ""; } catch { return ""; }
}
function setApiKey(value) {
  try {
    if (value) localStorage.setItem(API_KEY_STORAGE, value);
    else localStorage.removeItem(API_KEY_STORAGE);
  } catch { /* localStorage no disponible: la sesión simplemente no persiste la key */ }
}

const state = {
  live: true,
  eventSource: null,
  windowMinutes: 60,
  seenEventIds: new Set(),
  charts: {},
  lastAgentCorrelationId: null,
};

const LEVEL_COLOR = { INFO: "#6C8EFF", WARNING: "#FBBF24", ERROR: "#FB5D5D", CRITICAL: "#FB5D5D" };
const EVENT_TYPE_COLOR = {
  agent_start: "#6C8EFF", agent_finish: "#34D399", reasoning_step: "#22D3C7",
  tool_result: "#8B96A5", tool_invoked: "#8B96A5", decision: "#FBBF24",
  security_event: "#FB5D5D", budget_exceeded: "#FB5D5D", circuit_breaker_check: "#FB5D5D",
  emergency_override: "#FBBF24", swarm_halted: "#FB5D5D",
};

// ─── Utilidades ─────────────────────────────────────────────────────
function $(sel, root = document) { return root.querySelector(sel); }
function $all(sel, root = document) { return [...root.querySelectorAll(sel)]; }
function fmtUsd(n) { return `$${Number(n || 0).toFixed(6)}`; }
function fmtMs(n) { return `${Math.round(Number(n || 0))} ms`; }
function fmtTime(iso) {
  try { return new Date(iso).toLocaleTimeString("es-ES", { hour12: false }); }
  catch { return iso || "—"; }
}
function shortId(id, n = 18) { return !id ? "—" : (id.length > n ? id.slice(0, n) + "…" : id); }

async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  const apiKey = getApiKey();
  if (apiKey) headers["X-API-Key"] = apiKey;

  const res = await fetch(`${API}${path}`, { ...opts, headers });
  if (res.status === 401) {
    openAuthModal();
    throw new Error("Se requiere una API key válida (ver el candado en la esquina superior).");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

function toast(message, kind = "info") {
  const stack = $("#toast-stack");
  const el = document.createElement("div");
  el.className = `toast${kind === "error" ? " toast-error" : kind === "success" ? " toast-success" : ""}`;
  el.textContent = message;
  stack.appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

// ─── Autenticación (API key opcional) ────────────────────────────────
function refreshAuthButton(authEnabled) {
  const btn = $("#auth-btn");
  const label = $("#auth-btn-label");
  const hasKey = !!getApiKey();
  btn.classList.remove("is-locked", "is-unlocked");
  if (!authEnabled) {
    label.textContent = "Sin auth";
  } else if (hasKey) {
    btn.classList.add("is-unlocked");
    label.textContent = "Autenticado";
  } else {
    btn.classList.add("is-locked");
    label.textContent = "API key";
  }
}

function openAuthModal() {
  $("#auth-modal-backdrop").hidden = false;
  $("#auth-modal").hidden = false;
  $("#auth-modal-input").value = getApiKey();
  $("#auth-modal-input").focus();
}
function closeAuthModal() {
  $("#auth-modal-backdrop").hidden = true;
  $("#auth-modal").hidden = true;
}

$("#auth-btn").addEventListener("click", openAuthModal);
$("#auth-modal-backdrop").addEventListener("click", closeAuthModal);
$("#auth-modal-save").addEventListener("click", () => {
  setApiKey($("#auth-modal-input").value.trim());
  closeAuthModal();
  toast("API key guardada en este navegador", "success");
  loadHealth();
  refreshMetrics();
  loadRecentEvents();
});
$("#auth-modal-clear").addEventListener("click", () => {
  setApiKey("");
  $("#auth-modal-input").value = "";
  toast("API key eliminada de este navegador", "info");
  refreshAuthButton($("#auth-btn").dataset.authEnabled === "true");
});

// ─── Navegación ─────────────────────────────────────────────────────
const VIEW_META = {
  overview: ["Overview", "Estado agregado de los 4 pilares y del agente en vivo"],
  traces: ["Trazas", "Live tail de eventos estructurados de todos los agentes"],
  agent: ["Agente en vivo", "Invoca el agente ReAct con trazabilidad completa"],
  api: ["API / Swagger", "Documentación interactiva de la API que alimenta esta consola"],
};

function switchView(view) {
  $all(".nav-item").forEach((b) => b.classList.toggle("is-active", b.dataset.view === view));
  $all("[data-view-panel]").forEach((p) => { p.hidden = p.id !== `view-${view}`; });
  const [title, subtitle] = VIEW_META[view];
  $("#view-title").textContent = title;
  $("#view-subtitle").textContent = subtitle;
}

$all(".nav-item").forEach((btn) => btn.addEventListener("click", () => switchView(btn.dataset.view)));

// ─── Salud / entorno ────────────────────────────────────────────────
async function loadHealth() {
  try {
    const health = await api("/health");
    const pill = $("#env-pill");
    if (health.providers_available.length) {
      pill.innerHTML = `<span class="dot dot-ok"></span> API real: ${health.providers_available.join(", ")}`;
    } else {
      pill.innerHTML = `<span class="dot dot-warn"></span> modo simulado (sin API key)`;
    }
    $("#agent-provider-hint").textContent = health.providers_available.length
      ? `usará ${health.providers_available[0]} (API real)`
      : "usará el simulador ReAct (sin API key en el servidor)";

    $("#auth-btn").dataset.authEnabled = String(health.auth_enabled);
    $("#auth-modal-copy").textContent = health.auth_enabled
      ? "Este servidor requiere una API key. Introdúcela abajo; se enviará como cabecera X-API-Key en cada petición."
      : "Este servidor no requiere autenticación en este momento: la API está abierta.";
    refreshAuthButton(health.auth_enabled);
  } catch {
    $("#env-pill").innerHTML = `<span class="dot dot-bad"></span> API no disponible`;
  }
}

// ─── Métricas / KPIs / gráficos ─────────────────────────────────────
function ensureChart(id, config) {
  if (state.charts[id]) { state.charts[id].destroy(); }
  const ctx = $(`#${id}`).getContext("2d");
  state.charts[id] = new Chart(ctx, config);
}

const CHART_BASE = {
  plugins: { legend: { labels: { color: "#8B96A5", font: { family: "Inter", size: 11 }, boxWidth: 10 } } },
  scales: {
    x: { ticks: { color: "#5B6472", font: { size: 10.5 } }, grid: { color: "#1A2029" } },
    y: { ticks: { color: "#5B6472", font: { size: 10.5 } }, grid: { color: "#1A2029" }, beginAtZero: true },
  },
};

async function refreshMetrics() {
  const metrics = await api(`/metrics/summary?window_minutes=${state.windowMinutes}`);

  $("#kpi-cost").textContent = fmtUsd(metrics.total_cost_usd);
  $("#kpi-events").textContent = metrics.total_events.toLocaleString("es-ES");
  $("#kpi-latency").textContent = fmtMs(metrics.p95_latency_ms);
  $("#kpi-latency-avg").textContent = Math.round(metrics.avg_latency_ms);
  const runsCompleted = metrics.runs_by_status.completed || 0;
  const runsRunning = metrics.runs_by_status.running || 0;
  $("#kpi-runs").textContent = runsCompleted;
  $("#kpi-runs-hint").textContent = `${runsRunning} en curso`;
  $("#nav-live-count").textContent = metrics.total_events;

  const modelEntries = Object.entries(metrics.cost_by_model);
  ensureChart("chart-cost-model", {
    type: "doughnut",
    data: {
      labels: modelEntries.length ? modelEntries.map(([k]) => k) : ["Sin datos"],
      datasets: [{
        data: modelEntries.length ? modelEntries.map(([, v]) => v) : [1],
        backgroundColor: ["#6C8EFF", "#22D3C7", "#FBBF24", "#34D399", "#FB5D5D"],
        borderColor: "#12161D", borderWidth: 2,
      }],
    },
    options: { plugins: CHART_BASE.plugins, cutout: "62%" },
  });

  const pillarEntries = Object.entries(metrics.events_by_pillar);
  ensureChart("chart-events-pillar", {
    type: "bar",
    data: {
      labels: pillarEntries.map(([k]) => k),
      datasets: [{ data: pillarEntries.map(([, v]) => v), backgroundColor: "#6C8EFF", borderRadius: 4 }],
    },
    options: { ...CHART_BASE, plugins: { legend: { display: false } } },
  });

  const levelEntries = Object.entries(metrics.events_by_level);
  ensureChart("chart-events-level", {
    type: "bar",
    data: {
      labels: levelEntries.map(([k]) => k),
      datasets: [{
        data: levelEntries.map(([, v]) => v),
        backgroundColor: levelEntries.map(([k]) => LEVEL_COLOR[k] || "#8B96A5"),
        borderRadius: 4,
      }],
    },
    options: { ...CHART_BASE, plugins: { legend: { display: false } } },
  });
}

// ─── Tabla de trazas (histórico + live tail) ────────────────────────
function renderTraceRow(ev, { animate = false } = {}) {
  const tbody = $("#trace-table-body");
  const emptyRow = tbody.querySelector(".empty-row");
  if (emptyRow) emptyRow.remove();

  const tr = document.createElement("tr");
  if (animate) tr.classList.add("row-enter");
  tr.dataset.correlationId = ev.correlation_id;
  tr.innerHTML = `
    <td class="mono">${fmtTime(ev.timestamp)}</td>
    <td><span class="level-chip level-${ev.level}">${ev.level}</span></td>
    <td class="pillar-tag">${ev.pillar || "—"}</td>
    <td>${ev.event_type}</td>
    <td class="mono corr-id">${shortId(ev.correlation_id)}</td>
    <td class="mono">${ev.model_used ? shortId(ev.model_used, 22) : "—"}</td>
    <td class="mono">${ev.latency_ms != null ? fmtMs(ev.latency_ms) : "—"}</td>
    <td class="mono">${ev.cost_usd ? fmtUsd(ev.cost_usd) : "—"}</td>
  `;
  tr.addEventListener("click", () => openTraceDrawer(ev.correlation_id));
  tbody.prepend(tr);

  while (tbody.children.length > 300) tbody.lastElementChild.remove();
}

async function loadRecentEvents() {
  const events = await api("/traces?limit=60");
  const tbody = $("#trace-table-body");
  tbody.innerHTML = "";
  if (!events.length) {
    tbody.innerHTML = `<tr class="empty-row"><td colspan="8">Sin eventos todavía. Ejecuta un pilar desde Overview.</td></tr>`;
    return;
  }
  events.slice().reverse().forEach((ev) => renderTraceRow(ev));
}

// ─── Live tail (SSE) ────────────────────────────────────────────────
function connectLiveTail() {
  if (state.eventSource) return;
  const es = new EventSource(`${API}/traces/stream`);
  es.onmessage = (msg) => {
    try {
      const ev = JSON.parse(msg.data);
      if (!ev.event_type || state.seenEventIds.has(ev.id)) return;
      state.seenEventIds.add(ev.id);
      renderTraceRow(ev, { animate: true });
      const current = parseInt($("#nav-live-count").textContent || "0", 10);
      $("#nav-live-count").textContent = current + 1;
    } catch { /* keep-alive u otro payload no-JSON: ignorar */ }
  };
  es.onerror = () => { /* el navegador reintenta solo; no hacemos nada ruidoso aquí */ };
  state.eventSource = es;
}

function disconnectLiveTail() {
  if (state.eventSource) { state.eventSource.close(); state.eventSource = null; }
}

$("#live-toggle").addEventListener("click", () => {
  state.live = !state.live;
  const btn = $("#live-toggle");
  btn.setAttribute("aria-pressed", String(state.live));
  btn.innerHTML = state.live
    ? `<span class="pulse"></span> Live`
    : `<span class="pulse"></span> Pausado`;
  if (state.live) connectLiveTail(); else disconnectLiveTail();
});

// ─── Waterfall de traza (drawer) ─────────────────────────────────────
async function openTraceDrawer(correlationId) {
  const backdrop = $("#drawer-backdrop");
  const drawer = $("#trace-drawer");
  backdrop.hidden = false;
  drawer.hidden = false;
  $("#drawer-correlation-id").textContent = correlationId;
  $("#drawer-summary").innerHTML = `<span>Cargando…</span>`;
  $("#drawer-waterfall").innerHTML = "";
  $("#drawer-events").innerHTML = "";

  let events;
  try {
    events = await api(`/traces/${encodeURIComponent(correlationId)}`);
  } catch (err) {
    $("#drawer-summary").innerHTML = `<span>${err.message}</span>`;
    return;
  }
  if (!events.length) return;

  const t0 = new Date(events[0].timestamp).getTime();
  const tEnd = new Date(events[events.length - 1].timestamp).getTime();
  const totalMs = Math.max(tEnd - t0, 1);
  const totalCost = events.reduce((s, e) => s + (e.cost_usd || 0), 0);
  const pillar = events.find((e) => e.pillar)?.pillar || "—";
  const agent = events.find((e) => e.agent_id)?.agent_id || "—";

  $("#drawer-summary").innerHTML = `
    <span>Pilar <b>${pillar}</b></span>
    <span>Agente <b>${agent}</b></span>
    <span>Eventos <b>${events.length}</b></span>
    <span>Duración <b>${totalMs} ms</b></span>
    <span>Coste <b>${fmtUsd(totalCost)}</b></span>
  `;

  const wf = $("#drawer-waterfall");
  events.forEach((ev) => {
    const offset = new Date(ev.timestamp).getTime() - t0;
    const width = ev.latency_ms ? Math.max((ev.latency_ms / totalMs) * 100, 1.2) : 1.2;
    const left = Math.min((offset / totalMs) * 100, 98);
    const color = EVENT_TYPE_COLOR[ev.event_type] || "#8B96A5";
    const row = document.createElement("div");
    row.className = "wf-row";
    row.innerHTML = `
      <span class="wf-label" title="${ev.event_type}">${ev.event_type}</span>
      <span class="wf-track"><span class="wf-bar" style="left:${left}%; width:${width}%; background:${color}" title="${ev.event_type} · +${offset}ms"></span></span>
      <span class="wf-dur">${ev.latency_ms != null ? Math.round(ev.latency_ms) + "ms" : "+" + offset + "ms"}</span>
    `;
    wf.appendChild(row);
  });

  const evWrap = $("#drawer-events");
  events.forEach((ev) => {
    const p = ev.payload || {};
    const bodyText = p.thought || p.rationale || p.message
      || (p.result != null ? `resultado: ${JSON.stringify(p.result)}` : "")
      || (p.details ? JSON.stringify(p.details) : "");
    const div = document.createElement("div");
    div.className = "dev-event";
    div.style.borderLeftColor = EVENT_TYPE_COLOR[ev.event_type] || "#232A35";
    div.innerHTML = `
      <div class="dev-event-head">
        <span class="level-chip level-${ev.level}">${ev.level}</span>
        <strong>${ev.event_type}</strong>
        <span class="dev-event-time">${fmtTime(ev.timestamp)}</span>
      </div>
      <div class="dev-event-body">${bodyText ? escapeHtml(bodyText) : "<em>sin detalle adicional</em>"}</div>
    `;
    evWrap.appendChild(div);
  });
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function closeDrawer() {
  $("#drawer-backdrop").hidden = true;
  $("#trace-drawer").hidden = true;
}
$("#drawer-close").addEventListener("click", closeDrawer);
$("#drawer-backdrop").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

// ─── Ejecutar pilares ────────────────────────────────────────────────
$all(".run-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const pillar = btn.dataset.run;
    btn.disabled = true;
    const originalText = btn.textContent;
    btn.textContent = "Ejecutando…";
    try {
      const result = await api(`/runs/${pillar}`, { method: "POST", body: JSON.stringify({ team_id: "demo" }) });
      toast(`Pilar "${pillar}" ejecutado — ver traza en Trazas`, "success");
      await Promise.all([refreshMetrics(), loadRecentEvents()]);
      switchView("traces");
      setTimeout(() => openTraceDrawer(result.correlation_id), 250);
    } catch (err) {
      toast(`Error ejecutando ${pillar}: ${err.message}`, "error");
    } finally {
      btn.disabled = false;
      btn.textContent = originalText;
    }
  });
});

// ─── Agente en vivo ─────────────────────────────────────────────────
function appendAgentMessage(role, text, thought) {
  const chat = $("#agent-chat");
  const div = document.createElement("div");
  div.className = `agent-msg agent-msg-${role}`;
  div.innerHTML = (thought ? `<span class="thought">${escapeHtml(thought)}</span>` : "") + escapeHtml(text);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

$("#agent-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("#agent-input");
  const query = input.value.trim();
  if (!query) return;
  appendAgentMessage("user", query);
  input.value = "";
  const submitBtn = $("#agent-form button");
  submitBtn.disabled = true;

  try {
    const result = await api("/agent/invoke", { method: "POST", body: JSON.stringify({ query, team_id: "demo" }) });
    const summary = result.summary || {};
    appendAgentMessage("agent", summary.result || "(sin respuesta)");

    $("#agent-meta").innerHTML = `
      <dt>Estado</dt><dd>${summary.status || "—"}</dd>
      <dt>Proveedor</dt><dd>${summary.provider || "—"}</dd>
      <dt>Coste</dt><dd>${summary.cost_usd != null ? fmtUsd(summary.cost_usd) : "—"}</dd>
      <dt>Pasos</dt><dd>${summary.steps ?? "—"}</dd>
    `;
    state.lastAgentCorrelationId = result.correlation_id;
    $("#agent-view-trace").disabled = !result.correlation_id;
    refreshMetrics();
  } catch (err) {
    appendAgentMessage("agent", `Error: ${err.message}`);
    toast(`Error invocando al agente: ${err.message}`, "error");
  } finally {
    submitBtn.disabled = false;
  }
});

$("#agent-view-trace").addEventListener("click", () => {
  if (state.lastAgentCorrelationId) openTraceDrawer(state.lastAgentCorrelationId);
});

// ─── Ventana de tiempo ──────────────────────────────────────────────
$("#window-minutes").addEventListener("change", (e) => {
  state.windowMinutes = parseInt(e.target.value, 10);
  refreshMetrics();
});

// ─── Arranque ───────────────────────────────────────────────────────
async function init() {
  await loadHealth();
  await Promise.all([refreshMetrics(), loadRecentEvents()]);
  connectLiveTail();
  setInterval(refreshMetrics, 20_000);
}

init();
