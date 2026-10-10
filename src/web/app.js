const $ = (id) => document.getElementById(id);
const STOP = new Set(("a an and are as at be by can could do does did for from has have how i in into is it my not of on or " +
  "our should that the there this to was were what when where which why will with would you your").split(" "));
const STORE_KEY = "rag.tabs.v1";
const DEBUG_KEY = "rag.debug";

// --- state --------------------------------------------------------------------
// Each tab is an independent search: form values, last results and LLM answer.
// Tabs live in localStorage, so they survive a reload (results are not re-fetched).

let tabs = [];
let activeId = null;
const controllers = {}; // tab id -> AbortController of its in-flight request
let llmInfo = {available: false, model: null, models: [], error: "not checked yet"};

const newId = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

function newTab(init = {}) {
  return {id: newId(), q: "", product: "", mode: "hybrid", rerank: true, llm: false, model: "",
          data: null, answer: null, status: "", busy: false, ...init};
}

const activeTab = () => tabs.find((t) => t.id === activeId);
const isActive = (tab) => tab.id === activeId;

function save() {
  try {
    const plain = tabs.map((t) => ({...t, busy: false, answer: t.answer && {...t.answer, streaming: false}}));
    localStorage.setItem(STORE_KEY, JSON.stringify({tabs: plain, activeId}));
  } catch { /* storage full or blocked: tabs just won't persist */ }
}

function load() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY));
    if (saved?.tabs?.length) {
      tabs = saved.tabs;
      activeId = tabs.some((t) => t.id === saved.activeId) ? saved.activeId : tabs[0].id;
    }
  } catch { /* corrupted or unavailable storage: start fresh */ }
}

function readForm(tab) {
  tab.q = $("q").value.trim();
  tab.product = $("product").value;
  tab.mode = $("mode").value;
  tab.rerank = $("rerank").checked;
  tab.llm = $("llm").checked;
  tab.model = $("model").value;
}

function writeForm(tab) {
  $("q").value = tab.q;
  $("product").value = tab.product;
  $("mode").value = tab.mode;
  $("rerank").checked = tab.rerank;
  $("llm").checked = tab.llm;
  $("model").value = [...$("model").options].some((o) => o.value === tab.model) ? tab.model : "";
  updateSubmitLabel(tab);
}

// Only the Find / Ask button runs a query; changing settings just marks the results as outdated.
const SETTINGS = ["q", "product", "mode", "rerank", "llm", "model"];
const settingsOf = (tab) => Object.fromEntries(SETTINGS.map((k) => [k, tab[k]]));
const isStale = (tab) => !!(tab.data && tab.ran && SETTINGS.some((k) => tab.ran[k] !== tab[k]));

const submitLabel = (tab) => (tab.llm && llmInfo.available ? "Ask" : "Find");

function updateSubmitLabel(tab) {
  $("submit").textContent = submitLabel(tab);
}

function statusText(tab) {
  const label = submitLabel(tab);
  return (tab.status || "") + (isStale(tab) ? ` · settings changed, press ${label} to update` : "");
}

function onSettingsChange() {
  const tab = activeTab();
  readForm(tab);
  updateSubmitLabel(tab);
  $("status").textContent = statusText(tab);
  renderTabs(); // the tab shows the chosen LLM model
  save();
}

// --- text helpers -------------------------------------------------------------

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}

function terms(query) {
  return [...new Set(query.toLowerCase().match(/[\p{L}\p{N}_]{3,}/gu) || [])].filter((t) => !STOP.has(t));
}

// Light markdown cleanup for display: drop image markers and heading hashes.
function clean(text) {
  return text.replace(/\[image: [^\]]*\]/g, "").replace(/^#{1,6}\s+/gm, "").replace(/\n{2,}/g, "\n").trim();
}

function shortUrl(u) {
  const s = u.replace(/^https?:\/\/(www\.)?/, "");
  return s.length > 48 ? s.slice(0, 45) + "…" : s;
}

// Escape, then render a safe subset of inline markdown, then highlight query terms.
function highlight(text, ts) {
  let html = escapeHtml(text)
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/&lt;(https?:\/\/\S+?)&gt;/g, (m, u) => `<a href="${u}" target="_blank" rel="noopener">${shortUrl(u)}</a>`);
  if (!ts.length) return html;
  const re = new RegExp("(" + ts.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")", "gi");
  // highlight only outside tags
  return html.replace(/(<[^>]+>)|([^<]+)/g, (m, tag, txt) => tag || txt.replace(re, "<mark>$1</mark>"));
}

function snippet(text, ts, size = 420) {
  const body = clean(text);
  if (body.length <= size) return body;
  const lower = body.toLowerCase();
  const pos = ts.map((t) => lower.indexOf(t)).filter((i) => i >= 0).sort((a, b) => a - b)[0] ?? 0;
  const start = Math.max(0, pos - 120);
  return (start > 0 ? "… " : "") + body.slice(start, start + size).trim() + " …";
}

// --- results --------------------------------------------------------------------

function statusOf(data) {
  const weak = data.results.filter((p) => p.low_relevance).length;
  return weak
    ? `${data.results.length - weak} relevant · ${weak} low-relevance · ${data.took_ms} ms`
    : `${data.results.length} pages · ${data.took_ms} ms`;
}

// Cards are collapsible: the first one starts expanded, the rest collapsed; toggles are kept per tab.
const isOpen = (tab, pageId, i) => tab.open?.[pageId] ?? i === 0;

function renderResults(data, tab) {
  if (!data) { $("results").innerHTML = ""; return; }
  const ts = terms(data.query);
  if (!data.results.length) {
    $("results").innerHTML = '<div class="empty">Nothing found. Try other words or switch the mode.</div>';
    return;
  }
  // With rerank on, pages below the relevance cut-off are flagged and collapsed.
  const strong = data.results.filter((p) => !p.low_relevance);
  const weak = data.results.filter((p) => p.low_relevance);
  const empty = strong.length ? "" : '<div class="empty">No confident matches in the documentation.</div>';
  const order = [...strong, ...weak];
  const cards = (pages) => pages.map((p) => {
    const i = order.indexOf(p);
    return card(p, ts, isOpen(tab, p.page_id, i), i + 1);
  }).join("");
  const weakGroup = weak.length
    ? `<details class="weak-group"${strong.length ? "" : " open"}>
         <summary>${weak.length} low-relevance match${weak.length > 1 ? "es" : ""}</summary>
         ${cards(weak)}
       </details>`
    : "";
  $("results").innerHTML = empty + cards(strong) + weakGroup;
  $("results").classList.remove("has-sources");
}

// --- debug mode: how each result was scored --------------------------------------

let debug = false;
try { debug = localStorage.getItem(DEBUG_KEY) === "1"; } catch { /* storage blocked */ }

const num = (x, digits = 3) => (typeof x === "number" ? x.toFixed(digits) : "–");

// One line per signal: vector #rank cosine · bm25 #rank score · rrf · retrieval #rank → rerank
function signalsHtml(sig) {
  if (!debug || !sig) return "";
  const parts = [];
  if (sig.vector) parts.push(`vector #${sig.vector.rank} <b>${num(sig.vector.score)}</b>`);
  if (sig.fts) parts.push(`bm25 #${sig.fts.rank} <b>${num(sig.fts.score, 2)}</b>`);
  if (sig.rrf !== undefined) parts.push(`rrf <b>${num(sig.rrf, 4)}</b>`);
  if (sig.retrieval_rank) parts.push(`retrieval #${sig.retrieval_rank}`);
  if (sig.rerank !== undefined) parts.push(`rerank <b>${num(sig.rerank)}</b>`);
  return parts.length ? `<div class="debug">${parts.join(" · ")}</div>` : "";
}

function setDebug(on) {
  debug = on;
  try { localStorage.setItem(DEBUG_KEY, on ? "1" : "0"); } catch { /* storage blocked */ }
  $("debug").setAttribute("aria-pressed", String(on));
  renderContent(activeTab());
}

function card(p, ts, open, rank) {
  const crumbs = p.breadcrumbs.filter((c) => c !== "Home").join(" › ") || p.product;
  const updated = p.updated ? new Date(p.updated).toLocaleDateString() : "";
  const chunks = p.chunks.map((c, ci) => `
    <div class="chunk">
      ${c.section_path.length ? `<div class="section">§ ${escapeHtml(c.section_path.join(" › "))}</div>` : ""}
      ${ci > 0 ? signalsHtml(c.signals) : "" /* the best fragment's signals are in the header */}
      <div class="snippet">${highlight(snippet(c.text, ts), ts)}</div>
      ${c.text.length > 420 ? `<details><summary>Show full fragment</summary><div class="full">${highlight(clean(c.text), ts)}</div></details>` : ""}
    </div>`).join("");
  return `
    <details class="result" data-page-id="${escapeHtml(p.page_id)}"${open ? " open" : ""}>
      <summary>
        <div class="crumbs">${escapeHtml(crumbs)}</div>
        <h2><span class="rank-badge" title="Rank ${rank}">${rank}</span><a href="${escapeHtml(p.url)}" target="_blank" rel="noopener">${escapeHtml(p.title)}</a></h2>
        <div class="meta"><span>Updated ${updated}</span>${p.labels.length ? `<span>${p.labels.map(escapeHtml).join(", ")}</span>` : ""}<span>score ${p.score.toFixed(3)}</span></div>
        ${signalsHtml(p.chunks[0]?.signals)}
      </summary>
      ${chunks}
    </details>`;
}

// --- LLM answer ---------------------------------------------------------------

function inlineMd(text, sources) {
  return escapeHtml(text.replace(/\$?\\(?:long)?(?:right)?arrow\$?|\$\\to\$/gi, "→"))
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*\n]*)\*/g, "$1<em>$2</em>")
    .replace(/\[(\d+)\]/g, (m, n) => {
      const src = sources[n - 1];
      return src ? `<a class="cite" href="#src-${n}" title="${escapeHtml(src.title)}">${n}</a>` : m;
    });
}

// Minimal block markdown for LLM output: paragraphs, bullet/numbered lists, headings.
function renderMarkdown(text, sources) {
  let html = "", list = null, para = [];
  const flushPara = () => { if (para.length) { html += `<p>${inlineMd(para.join(" "), sources)}</p>`; para = []; } };
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  const openList = (tag) => { flushPara(); if (list !== tag) { closeList(); html += `<${tag}>`; list = tag; } };
  for (const line of text.split("\n")) {
    let m;
    if (!line.trim()) { flushPara(); closeList(); }
    else if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) { openList("ul"); html += `<li>${inlineMd(m[1], sources)}</li>`; }
    else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) { openList("ol"); html += `<li>${inlineMd(m[1], sources)}</li>`; }
    else if ((m = line.match(/^#{1,6}\s+(.*)$/))) { flushPara(); closeList(); html += `<h3>${inlineMd(m[1], sources)}</h3>`; }
    else { closeList(); para.push(line.trim()); }
  }
  flushPara(); closeList();
  return html;
}

// Cards used as answer sources get a filled badge with the source number (= their rank, as the
// sources are the top relevant pages in order) and an anchor for the [n] citations.
function markSources(sources) {
  if (!sources?.length) return;
  const byPage = Object.fromEntries(sources.map((s) => [s.page_id, s.n]));
  $("results").classList.add("has-sources");
  document.querySelectorAll(".result").forEach((card) => {
    const n = byPage[card.dataset.pageId];
    if (!n) return;
    card.id = `src-${n}`;
    const badge = card.querySelector(".rank-badge");
    badge.classList.add("src");
    badge.textContent = n;
    badge.title = `Source ${n} of the answer`;
  });
}

function renderAnswer(answer) {
  if (!answer) { $("answer").hidden = true; return; }
  const {text, sources, streaming, footer, error, model} = answer;
  $("answer").hidden = false;
  $("answer").innerHTML = `<details class="answer-panel"${answer.open === false ? "" : " open"}>
    <summary class="answer-head">Answer${model ? ` · <span class="answer-model">${escapeHtml(model)}</span>` : ""}${streaming ? ' <span class="pulse">generating…</span>' : ""}</summary>
    <div class="answer-body${streaming ? " streaming" : ""}">${text ? renderMarkdown(text, sources) : (streaming ? "" : "<p>No answer.</p>")}</div>
    ${error ? `<div class="answer-error">${escapeHtml(error)}</div>` : ""}
    ${footer ? `<div class="answer-foot">${escapeHtml(footer)}</div>` : ""}
  </details>`;
}

// --- view -----------------------------------------------------------------------

function renderTabs() {
  // Tabs are named by position: Tab1, Tab2...; the query shows in the tooltip.
  $("tabs").innerHTML = tabs.map((t, i) => `
    <div class="tab${t.id === activeId ? " active" : ""}" data-id="${t.id}" title="${escapeHtml(t.q || "New search")}" role="tab" aria-selected="${t.id === activeId}">
      ${t.busy ? '<span class="tab-busy" aria-label="loading"></span>' : ""}
      <span class="tab-title">Tab${i + 1}</span>
      ${t.llm && t.model ? `<span class="tab-model">${escapeHtml(t.model.split("/").pop())}</span>` : ""}
      ${tabs.length > 1 ? `<button class="tab-close" data-close="${t.id}" aria-label="Close tab" title="Close tab">×</button>` : ""}
    </div>`).join("") +
    '<button class="tab-add" data-add title="New tab" aria-label="New tab">+</button>';
}

// Results + answer of a tab (the form is written only on tab switch, so typing isn't clobbered).
function renderContent(tab) {
  $("status").textContent = statusText(tab);
  renderResults(tab.data, tab);
  renderAnswer(tab.answer); // set only when the last run used the LLM
  if (tab.answer) markSources(tab.answer.sources);
}

function renderView() {
  const tab = activeTab();
  renderTabs();
  writeForm(tab);
  renderContent(tab);
  updateUrl(tab);
}

function updateUrl(tab) {
  if (!tab.q) { history.replaceState(null, "", location.pathname); return; }
  const params = new URLSearchParams({q: tab.q, mode: tab.mode, rerank: tab.rerank});
  if (tab.product) params.set("product", tab.product);
  if (tab.llm) params.set("llm", "true");
  if (tab.llm && tab.model) params.set("model", tab.model);
  history.replaceState(null, "", "?" + params);
}

function switchTab(id) {
  if (id === activeId) return;
  readForm(activeTab()); // keep unsent edits of the tab we leave
  activeId = id;
  renderView();
  save();
}

function addTab() {
  const current = activeTab();
  readForm(current);
  // a new tab inherits the settings, not the query
  const tab = newTab({product: current.product, mode: current.mode, rerank: current.rerank, llm: current.llm, model: current.model});
  tabs.push(tab);
  activeId = tab.id;
  renderView();
  save();
  $("q").focus();
}

function closeTab(id) {
  controllers[id]?.abort();
  const i = tabs.findIndex((t) => t.id === id);
  tabs.splice(i, 1);
  if (!tabs.length) tabs.push(newTab());
  if (id === activeId) activeId = tabs[Math.min(i, tabs.length - 1)].id;
  renderView();
  save();
}

// Clear the active tab: query, results and answer; the filters stay.
function clearResults() {
  const tab = activeTab();
  controllers[tab.id]?.abort();
  delete controllers[tab.id];
  Object.assign(tab, {q: "", data: null, answer: null, status: "", busy: false, open: {}});
  $("q").value = "";
  renderTabs();
  renderContent(tab);
  updateUrl(tab);
  save();
  $("q").focus();
}

// --- requests ---------------------------------------------------------------------

async function readNdjson(res, onEvent) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += decoder.decode(value, {stream: true});
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
}

async function runAnswer(tab, params, signal) {
  if (tab.model) params.set("model", tab.model);
  const res = await fetch("/api/answer?" + params, {signal});
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  const a = tab.answer;
  await readNdjson(res, (ev) => {
    if (ev.type === "results") {
      tab.data = ev;
      tab.status = statusOf(ev);
      if (isActive(tab)) { $("status").textContent = tab.status; renderResults(ev, tab); }
      return;
    }
    if (ev.type === "sources") a.sources = ev.sources;
    else if (ev.type === "delta") a.text += ev.text;
    else if (ev.type === "done") {
      a.streaming = false;
      a.model = ev.model || a.model;
      a.footer = ev.model ? `${(ev.took_ms / 1000).toFixed(1)} s · based on ${a.sources.length} source${a.sources.length === 1 ? "" : "s"}` : "";
    } else if (ev.type === "error") { a.streaming = false; a.error = "LLM error: " + ev.message; }
    if (isActive(tab)) { renderAnswer(a); if (ev.type === "sources") markSources(a.sources); }
  });
}

async function run() {
  const tab = activeTab();
  readForm(tab);
  if (!tab.q) return;
  const useLlm = tab.llm && llmInfo.available;
  const params = new URLSearchParams({q: tab.q, mode: tab.mode, rerank: tab.rerank});
  if (tab.product) params.set("product", tab.product);

  controllers[tab.id]?.abort();
  const controller = controllers[tab.id] = new AbortController();
  Object.assign(tab, {status: "Searching…", data: null, busy: true, open: {}, ran: settingsOf(tab),
    answer: useLlm ? {text: "", sources: [], streaming: true, model: tab.model || llmInfo.model} : null});
  renderTabs();
  renderContent(tab);
  updateUrl(tab);
  try {
    if (useLlm) {
      await runAnswer(tab, params, controller.signal);
    } else {
      const res = await fetch("/api/search?" + params, {signal: controller.signal});
      if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
      tab.data = await res.json();
      tab.status = statusOf(tab.data);
    }
  } catch (err) {
    if (err.name === "AbortError") return; // superseded by a newer request or tab closed
    tab.status = "Error: " + err.message;
    if (tab.answer) tab.answer.streaming = false;
  }
  if (controllers[tab.id] !== controller) return;
  delete controllers[tab.id];
  tab.busy = false;
  if (isActive(tab)) renderContent(tab);
  renderTabs();
  save();
}

// --- LLM models ---------------------------------------------------------------

async function refreshLlmStatus() {
  llmInfo = await fetch("/api/llm").then((r) => r.json())
    .catch(() => ({available: false, model: null, models: [], error: "backend not reachable"}));
  const selected = $("model").value || activeTab()?.model || "";
  $("model").innerHTML = `<option value="">Default (${escapeHtml(llmInfo.model || "none")})</option>` +
    llmInfo.models.map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.id)}${m.loaded === false ? " (not loaded)" : ""}</option>`).join("");
  $("model").value = [...$("model").options].some((o) => o.value === selected) ? selected : "";
  $("llm").disabled = $("model").disabled = !llmInfo.available;
  $("llm-label").title = llmInfo.available
    ? "Models marked (not loaded) are loaded by LM Studio on first use, which takes a while"
    : `LLM unavailable: ${llmInfo.error}`;
  if (activeTab()) updateSubmitLabel(activeTab());
}

// --- init ---------------------------------------------------------------------

async function init() {
  const products = await fetch("/api/products").then((r) => r.json()).catch(() => []);
  $("product").insertAdjacentHTML("beforeend", products.map((p) => `<option>${escapeHtml(p)}</option>`).join(""));
  await refreshLlmStatus();

  load();
  // A shared link (?q=...) opens in its own tab unless a tab with the same query exists.
  const params = new URLSearchParams(location.search);
  const q = params.get("q");
  let runNow = false;
  if (q) {
    const existing = tabs.find((t) => t.q === q);
    if (existing) activeId = existing.id;
    else {
      const tab = newTab({q, product: params.get("product") || "", mode: params.get("mode") || "hybrid",
        rerank: params.get("rerank") !== "false", llm: params.get("llm") === "true", model: params.get("model") || ""});
      tabs.push(tab);
      activeId = tab.id;
      runNow = true;
    }
  }
  if (!tabs.length) { tabs.push(newTab()); activeId = tabs[0].id; }
  $("debug").setAttribute("aria-pressed", String(debug));
  renderView();
  if (runNow) run();
}

$("form").addEventListener("submit", (e) => { e.preventDefault(); run(); });
$("clear").addEventListener("click", clearResults);
$("debug").addEventListener("click", () => setDebug(!debug));
["product", "mode", "rerank", "llm", "model"].forEach((id) => $(id).addEventListener("change", onSettingsChange));
$("tabs").addEventListener("click", (e) => {
  const close = e.target.closest("[data-close]");
  if (close) { e.stopPropagation(); closeTab(close.dataset.close); return; }
  if (e.target.closest("[data-add]")) { addTab(); return; }
  const tab = e.target.closest(".tab");
  if (tab) switchTab(tab.dataset.id);
});
// Middle click closes a tab, like in a browser.
$("tabs").addEventListener("auxclick", (e) => {
  const tab = e.target.closest(".tab");
  if (e.button === 1 && tab && tabs.length > 1) closeTab(tab.dataset.id);
});
// Remember expanded / collapsed cards and answer per tab ("toggle" doesn't bubble: capture it).
document.addEventListener("toggle", (e) => {
  const el = e.target, tab = activeTab();
  if (!tab || !el.isConnected) return;
  if (el.matches(".result")) (tab.open ||= {})[el.dataset.pageId] = el.open;
  else if (el.matches(".answer-panel") && tab.answer) tab.answer.open = el.open;
  else return;
  save();
}, true);
// A citation [n] expands the source card it jumps to.
$("answer").addEventListener("click", (e) => {
  const cite = e.target.closest("a.cite");
  const target = cite && document.querySelector(cite.getAttribute("href"));
  if (target) target.open = true;
});
// Models may be loaded in LM Studio after the page was opened.
$("llm-label").addEventListener("mouseenter", refreshLlmStatus);
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== $("q")) { e.preventDefault(); $("q").focus(); }
});
init();
