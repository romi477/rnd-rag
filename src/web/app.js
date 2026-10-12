const $ = (id) => document.getElementById(id);
const STOP = new Set(("a an and are as at be by can could do does did for from has have how i in into is it my not of on or " +
  "our should that the there this to was were what when where which why will with would you your").split(" "));
const STORE_KEY = "rag.tabs.v1";
const DEBUG_KEY = "rag.debug";
const THEME_KEY = "rag.theme";

// --- state --------------------------------------------------------------------
// Each tab is an independent search: form values, last results and LLM answer.
// Tabs live in localStorage, so they survive a reload (results are not re-fetched).

let tabs = [];
let activeId = null;
let closedIds = []; // recently closed tabs, so that another window's copy of them is dropped too
const controllers = {}; // tab id -> AbortController of its in-flight request
let llmInfo = {available: false, model: null, models: [], error: "not checked yet", checked: false};
let products = null; // product filter options, null until loaded
// The UI always searches hybrid + rerank (the rerank score is the relevance cut-off); the API and CLI still
// take other modes, for experiments and eval.
const SEARCH_PARAMS = {mode: "hybrid", rerank: "true"};
const VIEWS = ["search", "browse"];
const MODES = ["docs", "code"]; // per tab: the documentation, or the code of the modules (explains a failed job's traceback)
// The one switch in the header has three positions: Search and Browse are the views of the documentation, Code is a mode.
const POSITIONS = ["search", "browse", "code"];
const positionOf = (tab) => (tab.mode === "code" ? "code" : tab.view);
const TRACE_MAX = 30000; // = the API limit
const SEARCH_PLACEHOLDER = $("q").placeholder;
const DOCS_SUBTITLE = $("subtitle").textContent;
const CODE_PLACEHOLDER = "Optional: a question about the failure…";
const CODE_DEFAULTS = {trace: "", module: "", version: "", docs: false};
const CODE_FIELDS = Object.keys(CODE_DEFAULTS);
const SETTING_DEFAULTS = {mode: "docs", ...CODE_DEFAULTS, lang: ""}; // what a saved `ran` snapshot may lack
// The languages an answer can be written in (GET /api/languages); "" in a tab means the server's default.
let languages = [{code: "en", name: "English", native: "English"}];
let defaultLang = "en";
let languagesLoaded = false;
let metaError = ""; // the product list could not be loaded
// Browse: every indexed article (GET /api/pages), loaded once and shared by all tabs.
let pages = null;
let pagesError = "";
let pagesPromise = null;
// Code mode: the indexed modules (GET /api/code/modules), loaded once like the products.
let codeModules = null;
let codeError = "";
let codeModulesPromise = null;

const newId = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

// view: "search" or "browse" (the title filter of the article list is `filter`, kept apart from the query `q`).
function newTab(init = {}) {
  return {id: newId(), mode: "docs", view: "search", q: "", filter: "", product: "", llm: false, model: "",
          ...CODE_DEFAULTS, lang: "", data: null, answer: null, status: "", error: "", busy: false, ...init};
}

const activeTab = () => tabs.find((t) => t.id === activeId);
const isActive = (tab) => tab.id === activeId;

// A tab saved while its request runs is stored as interrupted: after a reload it must not look
// finished (or "Searching…" forever).
function persisted(t) {
  if (!t.busy) return t;
  const answer = t.answer && {...t.answer, streaming: false,
    error: t.answer.streaming ? "Interrupted before the answer was complete." : t.answer.error};
  return {...t, busy: false, answer, status: "Interrupted: run it again to update"};
}

let saveTimer = null;
function save() {
  clearTimeout(saveTimer);
  saveTimer = null;
  try {
    const stored = readStored();
    // Tabs opened in another window since we last looked are kept, not overwritten.
    const known = new Set([...tabs.map((t) => t.id), ...closedIds]);
    const foreign = (stored?.tabs || []).filter((t) => !known.has(t.id));
    const closed = [...new Set([...closedIds, ...(stored?.closed || [])])].slice(-50);
    localStorage.setItem(STORE_KEY, JSON.stringify({tabs: [...tabs.map(persisted), ...foreign], activeId, closed}));
  } catch { /* storage full or blocked: tabs just won't persist */ }
}

// While an answer streams, save at most twice a second instead of on every token.
function saveSoon() {
  if (!saveTimer) saveTimer = setTimeout(save, 500);
}

// A reload or close inside that half second would lose the last tokens or the last typed filter.
window.addEventListener("pagehide", () => { if (saveTimer) save(); });

function readStored() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)); } catch { return null; }
}

function load() {
  const saved = readStored();
  if (saved?.tabs?.length) {
    tabs = saved.tabs.map(sanitize);
    activeId = tabs.some((t) => t.id === saved.activeId) ? saved.activeId : tabs[0].id;
  }
}

// Another window changed the tabs: take its new ones and drop the ones it closed.
window.addEventListener("storage", (e) => {
  if (e.key !== STORE_KEY) return;
  const stored = readStored();
  if (!stored?.tabs) return;
  const gone = new Set(stored.closed || []);
  const mine = new Set(tabs.map((t) => t.id));
  tabs = tabs.filter((t) => !gone.has(t.id) || controllers[t.id]);
  tabs.push(...stored.tabs.filter((t) => !mine.has(t.id) && !closedIds.includes(t.id)).map(sanitize));
  if (!tabs.length) tabs.push(newTab());
  if (!activeTab()) { activeId = tabs[0].id; renderView(); } else renderTabs();
});

// Values from localStorage or a URL may be stale or hand-made.
function sanitize(t) {
  const tab = {...newTab(), ...t};
  delete tab.rerank; // a setting of older versions (their search `mode` is replaced below: it is not "docs" / "code")
  if (!MODES.includes(tab.mode)) tab.mode = "docs"; // tabs saved before the Code mode are Docs tabs
  if (!VIEWS.includes(tab.view) || tab.mode === "code") tab.view = "search";
  for (const k of ["q", "filter", "product", "model", "error", "trace", "module", "version", "lang"]) if (typeof tab[k] !== "string") tab[k] = "";
  tab.q = tab.q.slice(0, 500);
  tab.filter = tab.filter.slice(0, 500);
  tab.trace = tab.trace.slice(0, TRACE_MAX);
  tab.version = tab.version.slice(0, 50);
  tab.llm = !!tab.llm;
  tab.docs = !!tab.docs;
  tab.lang = tab.lang.slice(0, 10);
  if (tab.ran) tab.ran = {...SETTING_DEFAULTS, ...tab.ran}; // else the new settings would look "changed"
  return tab;
}

const hasOption = (select, value) => [...$(select).options].some((o) => o.value === value);

// The one input holds the query in the Search view and the title filter in Browse: each keeps its own text.
function readForm(tab) {
  if (tab.view === "browse") tab.filter = $("q").value.slice(0, 500);
  else tab.q = $("q").value.trim();
  tab.product = $("product").value;
  if (tab.mode === "code") {
    tab.trace = $("trace").value.slice(0, TRACE_MAX);
    tab.module = $("code-module").value;
    tab.version = $("client-version").value.trim().slice(0, 50);
    tab.docs = $("code-docs").checked;
  }
  tab.llm = $("llm").checked;
  if (llmInfo.checked) tab.model = $("model").value;
  if (languagesLoaded) tab.lang = $("lang").value;
}

const effectiveLang = (tab) => (languages.some((l) => l.code === tab.lang) ? tab.lang : defaultLang);

// What a request carries: nothing until the list is loaded, so that the server's own default applies.
const langParam = (tab) => (languagesLoaded ? effectiveLang(tab) : "");

function fillLanguages(tab) {
  $("lang").innerHTML = languages.map((l) => `<option value="${escapeHtml(l.code)}" title="${escapeHtml(l.name)}">${escapeHtml(l.native)}</option>`).join("");
  $("lang").value = tab ? effectiveLang(tab) : defaultLang;
}

// Options of the product select. Until the list is loaded (or when it failed to load) the tab's own product
// stays listed, so that the select shows what the search will really filter by.
function fillProducts(tab) {
  const list = products || [];
  if (products && !products.includes(tab.product)) tab.product = ""; // a product that no longer exists
  const options = tab.product && !list.includes(tab.product) ? [tab.product, ...list] : list;
  $("product").innerHTML = '<option value="">All</option>' + options.map((p) => `<option>${escapeHtml(p)}</option>`).join("");
  $("product").value = tab.product;
}

// Options of the module select, like fillProducts: the tab's own module stays listed until the list is loaded.
function fillModules(tab) {
  const list = (codeModules || []).map((m) => m.name);
  const options = tab.module && !list.includes(tab.module) ? [tab.module, ...list] : list;
  $("code-module").innerHTML = '<option value="">All</option>' + options.map((m) => `<option>${escapeHtml(m)}</option>`).join("");
  $("code-module").value = tab.module;
}

function updatePlaceholder(tab) {
  $("q").placeholder = tab.mode === "code" ? CODE_PLACEHOLDER
    : tab.view !== "browse" ? SEARCH_PLACEHOLDER
    : pages ? `Filter ${pages.length} articles by title…` : "Filter articles by title…";
}

function writeForm(tab) {
  const browse = tab.view === "browse";
  fillProducts(tab);
  fillModules(tab);
  $("trace").value = tab.trace;
  $("client-version").value = tab.version;
  $("code-docs").checked = tab.docs;
  document.body.dataset.mode = tab.mode;
  $("subtitle").textContent = tab.mode === "code" ? "· Failed job analysis (experimental)" : DOCS_SUBTITLE;
  // a model that no longer exists falls back to the default model
  if (llmInfo.checked && !hasOption("model", tab.model)) tab.model = "";
  $("q").value = browse ? tab.filter : tab.q;
  updatePlaceholder(tab);
  $("llm").checked = tab.llm;
  $("model").value = hasOption("model", tab.model) ? tab.model : "";
  $("lang").value = effectiveLang(tab);
  for (const pos of POSITIONS) $(`pos-${pos}`).setAttribute("aria-pressed", String(positionOf(tab) === pos));
  updateSubmitLabel(tab);
}

// Only the Find / Ask button runs a query; changing settings just marks the results as outdated.
// The model matters only when the answer is written by the LLM.
const SETTINGS = ["mode", "q", "product", "trace", "module", "version", "docs", "llm", "model", "lang"];
const LLM_ONLY = ["model", "lang"]; // matter only when the answer is written by the LLM
const settingsOf = (tab) => Object.fromEntries(SETTINGS.map((k) => [k, tab[k]]));
const isStale = (tab) => !!(tab.data && tab.ran && SETTINGS.some((k) => tab.ran[k] !== tab[k] && (!LLM_ONLY.includes(k) || tab.llm)));

const submitLabel = (tab) => {
  const llm = tab.llm && llmInfo.available;
  return tab.mode === "code" ? (llm ? "Explain" : "Analyze") : (llm ? "Ask" : "Find");
};

function updateSubmitLabel(tab) {
  $("submit").textContent = submitLabel(tab);
}

function statusText(tab) {
  const label = submitLabel(tab);
  return (tab.status || "") + (isStale(tab) ? ` · settings changed, press ${label} to update` : "");
}

function onSettingsChange() {
  const tab = activeTab();
  if (!tab) return;
  readForm(tab);
  updateSubmitLabel(tab);
  $("status").textContent = statusText(tab);
  if (tab.view === "browse") renderBrowse(tab); // the product narrows the article list
  renderTabs(); // the tab shows the chosen LLM model
  updateUrl(tab);
  save();
}

// --- text helpers -------------------------------------------------------------

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}

function terms(query) {
  return [...new Set(query.toLowerCase().match(/[\p{L}\p{N}_]{3,}/gu) || [])].filter((t) => !STOP.has(t));
}

// Light markdown cleanup for display: drop heading hashes and blank lines.
function clean(text) {
  return text.replace(/^#{1,6}\s+/gm, "").replace(/\n{2,}/g, "\n").trim();
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

// What the Search view shows before the first search of a tab.
const HINT = `<div class="hint-empty">
  <div>Search the product documentation.</div>
  <div class="hint">Turn on "Answer with LLM" to get an answer with links to the sources.</div>
  <div class="hint">Not sure what to look for? <a href="#" data-browse>Browse all articles</a></div>
</div>`;

function renderResults(data, tab) {
  if (!data) { $("results").innerHTML = tab.busy || tab.error ? "" : HINT; return; }
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

// --- filter separators: no pipe before a filter that starts a wrapped row ---------------

function markFilterRows() {
  let prev = null;
  for (const el of document.querySelector(".filters").children) {
    if (!el.getClientRects().length) continue; // not shown in this view
    // filters are centred vertically and differ in height: a new row starts below the previous one
    el.classList.toggle("row-start", !prev || el.offsetTop >= prev.offsetTop + prev.offsetHeight);
    prev = el;
  }
}

// --- theme: follows the system until the button picks one (saved per browser) ------

const systemDark = matchMedia("(prefers-color-scheme: dark)");

function savedTheme() {
  let theme = null;
  try { theme = localStorage.getItem(THEME_KEY); } catch { /* storage blocked */ }
  return theme === "light" || theme === "dark" ? theme : null;
}

function applyTheme(theme = savedTheme() || (systemDark.matches ? "dark" : "light")) {
  document.documentElement.dataset.theme = theme;
  const label = `Switch to ${theme === "dark" ? "light" : "dark"} theme`;
  $("theme").title = label;
  $("theme").setAttribute("aria-label", label);
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  try { localStorage.setItem(THEME_KEY, next); } catch { /* storage blocked: this page only */ }
  applyTheme(next);
}

// Only http(s) links reach an href: the URL comes from the index, not from this page.
const safeUrl = (url) => (/^https?:\/\//i.test(url || "") ? url : null);

function card(p, ts, open, rank) {
  const crumbs = p.breadcrumbs.join(" › ") || p.product;
  const href = safeUrl(p.url);
  const updated = p.updated ? new Date(p.updated).toLocaleDateString() : "";
  const chunks = p.chunks.map((c, ci) => `
    <div class="chunk">
      ${c.section_path.length ? `<div class="section">§ ${escapeHtml(c.section_path.join(" › "))}</div>` : ""}
      ${ci > 0 ? signalsHtml(c.signals) : "" /* the best fragment's signals are in the header */}
      <div class="snippet">${highlight(snippet(c.text, ts), ts)}</div>
      ${clean(c.text).length > 420 ? `<details><summary>Show full fragment</summary><div class="full">${highlight(clean(c.text), ts)}</div></details>` : ""}
    </div>`).join("");
  return `
    <details class="result" data-page-id="${escapeHtml(p.page_id)}"${open ? " open" : ""}>
      <summary>
        <div class="crumbs">${escapeHtml(crumbs)}</div>
        <h2><span class="rank-badge" title="Rank ${rank}">${rank}</span>${href ? `<a href="${escapeHtml(href)}" target="_blank" rel="noopener">${escapeHtml(p.title)}</a>` : escapeHtml(p.title)}</h2>
        <div class="meta"><span>Updated ${updated}</span>${p.labels.length ? `<span>${p.labels.map(escapeHtml).join(", ")}</span>` : ""}<span>score ${p.score.toFixed(3)}</span></div>
        ${signalsHtml(p.chunks[0]?.signals)}
      </summary>
      ${chunks}
    </details>`;
}

// --- LLM answer ---------------------------------------------------------------

// A page source has a title; a code source is named by where it is.
const sourceTitle = (src) => src.title || `${src.module}/${src.path}:${src.line_start}-${src.line_end}`;

function inlineMd(text, sources) {
  // Code spans first, into placeholders: their content is literal (no *emphasis*, no [n] links).
  const codes = [];
  const html = escapeHtml(text.replace(/\$?\\(?:long)?(?:right)?arrow\$?|\$\\to\$/gi, "→"))
    .replace(/`([^`\n]+)`/g, (m, code) => `\u0000${codes.push(code) - 1}\u0000`)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    // *em*: opens after a space or line start, closes before a space or punctuation, so that
    // "*.pdf and *.xml" stays as typed
    .replace(/(^|[\s(])\*([^*\s](?:[^*\n]*[^*\s])?)\*(?=[\s.,;:!?)]|$)/g, "$1<em>$2</em>")
    .replace(/\[(?:sources?\s*:?\s*)?(\d+)\]/gi, (m, n) => {  // small models write "[Source 3]"
      const src = sources[n - 1];
      return src ? `<a class="cite" href="#src-${n}" title="${escapeHtml(sourceTitle(src))}">${n}</a>` : m;
    });
  return html.replace(/\u0000(\d+)\u0000/g, (m, i) => `<code>${codes[i]}</code>`);
}

// Block markdown for LLM output: paragraphs, headings, fenced code, nested bullet / numbered lists.
// A blank line between list items doesn't end the list (models often write "1. …\n\n2. …").
function renderMarkdown(text, sources) {
  let html = "", para = [], fence = null, code = [];
  const lists = []; // open lists, innermost last: {tag, indent}
  const flushPara = () => { if (para.length) { html += `<p>${inlineMd(para.join(" "), sources)}</p>`; para = []; } };
  const closeLists = (indent = -1) => {
    while (lists.length && lists.at(-1).indent > indent) html += `</li></${lists.pop().tag}>`;
  };
  // Some models answer with CRLF: a "\r" left at the end of a line would defeat the "$" of the patterns below.
  for (const line of text.split(/\r?\n/)) {
    if (fence) {
      if (line.trim().startsWith(fence) && !line.trim().slice(fence.length).trim()) {
        html += `<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`;
        fence = null; code = [];
      } else code.push(line);
      continue;
    }
    let m;
    if ((m = line.match(/^\s*(`{3,}|~{3,})/))) { flushPara(); closeLists(); fence = m[1]; continue; }
    if (!line.trim()) { flushPara(); continue; }
    if ((m = line.match(/^(\s*)([-*•]|(\d+)[.)])\s+(.*)$/))) {
      flushPara();
      const indent = m[1].replace(/\t/g, "    ").length, tag = m[3] ? "ol" : "ul";
      closeLists(indent);
      const top = lists.at(-1);
      if (top && top.indent === indent && top.tag !== tag) { html += `</li></${lists.pop().tag}>`; }
      if (lists.at(-1)?.indent === indent) html += "</li><li>";
      else {
        const start = m[3] && m[3] !== "1" ? ` start="${+m[3]}"` : "";
        html += `<${tag}${start}><li>`;
        lists.push({tag, indent});
      }
      html += inlineMd(m[4], sources);
      continue;
    }
    if ((m = line.match(/^#{1,6}\s+(.*)$/))) { flushPara(); closeLists(); html += `<h3>${inlineMd(m[1], sources)}</h3>`; continue; }
    if (lists.length && /^\s/.test(line)) { html += " " + inlineMd(line.trim(), sources); continue; } // item continues
    closeLists();
    para.push(line.trim());
  }
  if (fence) html += `<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`; // still streaming
  flushPara(); closeLists();
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

const answerBody = ({text, sources, streaming}) =>
  text ? renderMarkdown(text, sources) : (streaming ? "" : "<p>No answer.</p>");

function renderAnswer(answer) {
  if (!answer) { $("answer").hidden = true; return; }
  const {streaming, footer, error, model} = answer;
  $("answer").hidden = false;
  $("answer").innerHTML = `<details class="answer-panel"${answer.open === false ? "" : " open"}>
    <summary class="answer-head">Answer${model ? ` · <span class="answer-model">${escapeHtml(model)}</span>` : ""}${streaming ? ' <span class="pulse">generating…</span>' : ""}</summary>
    <div class="answer-body${streaming ? " streaming" : ""}">${answerBody(answer)}</div>
    ${error ? `<div class="answer-error">${escapeHtml(error)}</div>` : ""}
    ${footer ? `<div class="answer-foot">${escapeHtml(footer)}</div>` : ""}
    ${usageHtml(answer)}
  </details>`;
}

// Debug mode: what the answer cost in tokens, as the LLM server reported it.
function usageHtml({usage, firstTokenMs}) {
  if (!debug || (!usage && firstTokenMs == null)) return "";
  const parts = [];
  if (usage) {
    parts.push(`prompt <b>${usage.prompt_tokens}</b> tokens`);
    if (usage.completion_tokens != null) parts.push(`answer <b>${usage.completion_tokens}</b>`);
    if (usage.reasoning_tokens) parts.push(`of them reasoning <b>${usage.reasoning_tokens}</b>`);
  }
  if (firstTokenMs != null) parts.push(`first token <b>${(firstTokenMs / 1000).toFixed(1)}</b> s`);
  return `<div class="debug answer-debug">${parts.join(" · ")}</div>`;
}

// A new token only re-renders the answer text, not the panel around it.
function renderAnswerText(answer) {
  const body = $("answer").querySelector(".answer-body");
  if (body) body.innerHTML = answerBody(answer); else renderAnswer(answer);
}

// --- view -----------------------------------------------------------------------

function renderTabs() {
  $("stop").hidden = !activeTab()?.busy; // Stop belongs to the form: it follows the active tab's request
  // Tabs are named by position: Tab1, Tab2...; the query shows in the tooltip.
  $("tabs").innerHTML = tabs.map((t, i) => `
    <div class="tab${t.id === activeId ? " active" : ""}" data-id="${t.id}" title="${escapeHtml(t.mode === "code" ? `Code: ${t.q || (t.trace ? "traceback" : "new analysis")}` : t.view === "browse" ? `Browse: ${t.filter || "all articles"}` : t.q || "New search")}" role="tab" aria-selected="${t.id === activeId}" tabindex="${t.id === activeId ? 0 : -1}">
      ${t.busy ? '<span class="tab-busy" aria-label="loading"></span>' : ""}
      <span class="tab-title">Tab${i + 1}</span>
      ${t.mode === "code" ? '<span class="tab-model">code</span>' : t.view === "browse" ? '<span class="tab-model">browse</span>' : t.llm && t.model ? `<span class="tab-model">${escapeHtml(t.model.split("/").pop())}</span>` : ""}
      ${tabs.length > 1 ? `<button class="tab-close" data-close="${t.id}" aria-label="Close tab" title="Close tab">×</button>` : ""}
    </div>`).join("") +
    '<button class="tab-add" data-add title="New tab" aria-label="New tab">+</button>';
}

// A problem the user should notice: the last request of the tab failed, or a list could not be loaded.
function renderBanner(tab) {
  const text = tab.mode === "code" ? tab.error || codeError : (tab.view === "browse" ? pagesError : tab.error) || metaError;
  $("banner").textContent = text;
  $("banner").hidden = !text;
}

// Results + answer of a tab (the form is written only on tab switch, so typing isn't clobbered).
function renderContent(tab) {
  document.body.dataset.view = tab.view; // CSS shows the parts of the page that belong to the view
  document.body.dataset.mode = tab.mode;
  renderBanner(tab);
  if (tab.mode === "code") {
    $("status").textContent = statusText(tab);
    renderCode(tab.data, tab);
    renderAnswer(tab.answer);
    return;
  }
  if (tab.view === "browse") { renderBrowse(tab); return; }
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
  markFilterRows(); // the view shows other filters
}

// --- Code mode: what a traceback maps to in the code of the modules ---------------------------

const ROLE_LABEL = {frame: "failing frame", caller: "caller", override: "also defined here", abstract: "abstract declaration",
  "error code": "error code", search: "search"};
const HINT_CODE = `<div class="hint-empty">
  <div>Explain a failed job from its traceback.</div>
  <div class="hint">Paste the traceback of a <code>queue.job</code> above (a question is optional). The frames are looked up in the code of the indexed modules.</div>
  <div class="hint">Turn on "Answer with LLM" to get an explanation with references to the code.</div>
</div>`;

const looksLikeTraceback = (text) => /Traceback \(most recent call last\)|File "[^"]+", line \d+/.test(text);
const pieceKey = (module, path, start, end) => `${module}/${path}:${start}-${end}`;
const lineRange = (a, b) => (b && b !== a ? `${a}-${b}` : `${a}`);

function codeStatus(data) {
  const found = data.frames.filter((f) => f.found).length;
  const frames = data.frames.length ? `${found} of ${data.frames.length} frames found · ` : "";
  return `${frames}${data.pieces.length} fragment${data.pieces.length === 1 ? "" : "s"} · ${data.took_ms} ms`;
}

// The sources the LLM got, by where they are: the numbers its [n] citations use.
function codeNumbers(tab) {
  const sources = (tab.answer?.sources || []).filter((s) => s.kind === "code");
  return new Map(sources.map((s) => [pieceKey(s.module, s.path, s.line_start, s.line_end), s.n]));
}

function failureHtml(data) {
  const exc = data.exception;
  const codes = data.error_codes.map((c) => `<span class="chip" title="Error code of the framework">${escapeHtml(c)}</span>`).join("");
  const notes = data.notes.map((n) => `<div class="code-note">${escapeHtml(n)}</div>`).join("");
  if (!exc && !codes && !notes) return "";
  return `<div class="code-failure">
    ${exc ? `<div class="crumbs">Failure</div><h2>${escapeHtml(exc.name)}</h2>${exc.message ? `<pre>${escapeHtml(exc.message)}</pre>` : ""}` : ""}
    ${codes ? `<div class="meta">${codes}</div>` : ""}${notes}
  </div>`;
}

function framesHtml(data, numbers) {
  if (!data.frames.length) return "";
  return `<ol class="frames" aria-label="Frames in the indexed modules, innermost first">${data.frames.map((f) => {
    if (!f.found) {
      return `<li class="miss"><span class="loc">${escapeHtml(f.module)}/${escapeHtml(f.path)}:${f.line}</span> in ${escapeHtml(f.func)}
        <span class="at">not in the indexed code</span></li>`;
    }
    const n = numbers.get(pieceKey(f.module, f.path, f.line_start, f.line_end));
    return `<li><span class="loc">${escapeHtml(f.module)}/${escapeHtml(f.path)}:${lineRange(f.line_start, f.line_end)}</span>
      ${escapeHtml(f.name)}${f.name_matches ? "" : ` <span class="at">(frame ${escapeHtml(f.func)})</span>`}
      <span class="at">line ${f.line}</span>${n ? `<a class="cite" href="#src-${n}">${n}</a>` : ""}</li>`;
  }).join("")}</ol>`;
}

function pieceHtml(p, i, tab, numbers) {
  const key = pieceKey(p.module, p.path, p.line_start, p.line_end);
  const n = numbers.get(key);
  const sent = !tab.answer || n; // without an answer nothing was "sent": every fragment counts
  const badge = n ? `<span class="rank-badge src" title="Source ${n} of the answer">${n}</span>`
    : `<span class="rank-badge" title="${sent ? `Fragment ${i + 1}` : "Not sent to the LLM: over the token budget"}">${sent ? i + 1 : "–"}</span>`;
  const version = p.module_version ? `<span>${escapeHtml(p.module_version)}</span>` : "";
  return `<details class="result code-piece" data-page-id="${escapeHtml(key)}"${n ? ` id="src-${n}"` : ""}${isOpen(tab, key, i) ? " open" : ""}>
    <summary>
      <h2>${badge}<span class="loc">${escapeHtml(p.module)}/${escapeHtml(p.path)}:${lineRange(p.line_start, p.line_end)}</span></h2>
      <div class="meta"><span class="chip">${escapeHtml(ROLE_LABEL[p.role] || p.role)}</span><span>${escapeHtml(p.label)}</span>${version}</div>
    </summary>
    <pre><code>${escapeHtml(p.text)}</code></pre>
  </details>`;
}

function renderCode(data, tab) {
  if (!data) { $("results").innerHTML = tab.busy || tab.error ? "" : HINT_CODE; return; }
  const numbers = codeNumbers(tab);
  const docs = (tab.answer?.sources || []).filter((s) => s.kind === "page");
  const docsHtml = docs.length ? `<div class="code-docs"><b>Documentation</b>${docs.map((s) => {
    const href = safeUrl(s.url);
    return ` <span id="src-${s.n}">[${s.n}] ${href ? `<a href="${escapeHtml(href)}" target="_blank" rel="noopener">${escapeHtml(s.title)}</a>` : escapeHtml(s.title)}</span>`;
  }).join("")}</div>` : "";
  const empty = data.pieces.length ? "" : '<div class="empty">Nothing found in the indexed code for this input.</div>';
  $("results").classList.toggle("has-sources", numbers.size > 0);
  $("results").innerHTML = failureHtml(data) + framesHtml(data, numbers) +
    data.pieces.map((p, i) => pieceHtml(p, i, tab, numbers)).join("") + empty + docsHtml;
}

// --- Browse: every article, filtered by title as the user types -------------------------

// Articles of the product whose title has every word of the filter, grouped by product and section (the first two
// breadcrumbs). The server sends them sorted by product and path, so a group is a run of neighbours.
function groupPages(list, product, filter) {
  const words = filter.toLowerCase().split(/\s+/).filter(Boolean);
  const groups = [];
  let count = 0;
  for (const p of list) {
    const title = p.title || "";
    if ((product && p.product !== product) || !words.every((w) => title.toLowerCase().includes(w))) continue;
    const name = (p.breadcrumbs?.length ? p.breadcrumbs.slice(0, 2) : [p.product]).join(" · ");
    if (groups.at(-1)?.name !== name) groups.push({name, items: []});
    groups.at(-1).items.push({p, title: highlight(title, words)});
    count++;
  }
  return {count, groups};
}

function renderBrowse(tab) {
  if (!pages) {
    $("browse-count").textContent = "";
    $("browse").innerHTML = pagesError ? "" : '<div class="empty">Loading the articles…</div>';
    return;
  }
  const {count, groups} = groupPages(pages, tab.product, tab.filter);
  $("browse-count").textContent = `${count} article${count === 1 ? "" : "s"}`;
  $("browse").innerHTML = count ? groups.map((g) => `
    <section class="group">
      <h3 class="group-name">${escapeHtml(g.name)}</h3>
      <ul>${g.items.map(({p, title}) => {
        const href = safeUrl(p.url);
        const date = p.updated ? new Date(p.updated).toLocaleDateString() : "";
        return `<li class="page">${href
          ? `<a class="page-title" href="${escapeHtml(href)}" target="_blank" rel="noopener" title="Open in Confluence">${title}</a>`
          : `<span class="page-title">${title}</span>`}${date ? `<span class="page-date">${escapeHtml(date)}</span>` : ""}</li>`;
      }).join("")}</ul>
    </section>`).join("") : '<div class="empty">No article title matches. Clear the filter or choose another product.</div>';
}

// The article list and the product list load once; a failed load is retried when they are needed again.
function loadPages() {
  if (!pagesPromise) {
    pagesPromise = (async () => {
      try {
        const res = await fetch("/api/pages");
        if (!res.ok) throw await httpError(res);
        pages = (await res.json()).pages || [];
        pagesError = "";
      } catch (err) {
        pagesError = "Could not load the articles: " + err.message;
        pagesPromise = null;
      }
      const tab = activeTab();
      if (tab) { updatePlaceholder(tab); renderContent(tab); } // not writeForm: the user may be typing a filter
    })();
  }
  return pagesPromise;
}

let productsPromise = null;
function loadProducts() {
  if (!productsPromise) {
    productsPromise = (async () => {
      try {
        const res = await fetch("/api/products");
        if (!res.ok) throw await httpError(res);
        products = await res.json();
        metaError = "";
      } catch (err) {
        metaError = "Could not load the products: " + err.message;
        productsPromise = null;
      }
      const tab = activeTab();
      if (tab) {
        readForm(tab); // keep a product picked meanwhile: the options are rebuilt
        fillProducts(tab);
        renderBanner(tab);
        if (tab.view === "browse") renderBrowse(tab);
      }
    })();
  }
  return productsPromise;
}

// The module list of Code mode loads once; a failed load (no code index on the server) is retried when needed again.
function loadCodeModules() {
  if (!codeModulesPromise) {
    codeModulesPromise = (async () => {
      try {
        const res = await fetch("/api/code/modules");
        if (!res.ok) throw await httpError(res);
        codeModules = (await res.json()).modules || [];
        codeError = "";
      } catch (err) {
        codeError = "Code mode is not available: " + err.message;
        codeModulesPromise = null;
      }
      const tab = activeTab();
      if (tab?.mode === "code") { readForm(tab); fillModules(tab); renderBanner(tab); }
    })();
  }
  return codeModulesPromise;
}

// Switch the active tab between the documentation and the code; the question stays, the results do not.
function setMode(mode) {
  const tab = activeTab();
  if (!tab || tab.mode === mode) return;
  controllers[tab.id]?.abort();
  delete controllers[tab.id];
  readForm(tab);
  Object.assign(tab, {mode, view: "search", data: null, answer: null, status: "", error: "", busy: false, open: {}, ran: null});
  if (mode === "code") loadCodeModules(); else loadProducts();
  renderView();
  save();
  $(mode === "code" ? "trace" : "q").focus();
}

// The three-position switch: Code is a mode of its own, Search and Browse are the views of the documentation mode.
function setPosition(pos) {
  const tab = activeTab();
  if (!tab || positionOf(tab) === pos) return;
  if (pos === "code") { setMode("code"); return; }
  if (tab.mode === "code") setMode("docs"); // back to the documentation, in its Search view
  setView(pos);
}

// Switch the active tab between searching and browsing; each view keeps the text typed in it.
function setView(view) {
  const tab = activeTab();
  if (!tab || tab.view === view) return;
  readForm(tab);
  tab.view = view;
  if (view === "browse") loadPages();
  loadProducts();
  renderView();
  save();
  $("q").focus();
  $("q").select();
}

function updateUrl(tab) {
  if (tab.mode === "code") {  // the traceback is too long for a link: only the question and the module
    const params = new URLSearchParams({mode: "code"});
    if (tab.q) params.set("q", tab.q);
    if (tab.module) params.set("module", tab.module);
    history.replaceState(null, "", "?" + params);
    return;
  }
  if (tab.view === "browse") {
    const params = new URLSearchParams({view: "browse"});
    if (tab.product) params.set("product", tab.product);
    if (tab.filter) params.set("filter", tab.filter);
    history.replaceState(null, "", "?" + params);
    return;
  }
  if (!tab.q) { history.replaceState(null, "", location.pathname); return; }
  const params = new URLSearchParams({q: tab.q});
  if (tab.product) params.set("product", tab.product);
  if (tab.llm) params.set("llm", "true");
  if (tab.llm && tab.model) params.set("model", tab.model);
  if (tab.llm && tab.lang) params.set("lang", tab.lang);
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
  const tab = newTab({product: current.product, llm: current.llm, model: current.model, lang: current.lang});
  tabs.push(tab);
  activeId = tab.id;
  renderView();
  save();
  $("q").focus();
}

function closeTab(id) {
  if (id !== activeId) readForm(activeTab()); // keep what is typed in the tab that stays
  closedIds = [...closedIds, id].slice(-50);
  controllers[id]?.abort();
  const i = tabs.findIndex((t) => t.id === id);
  tabs.splice(i, 1);
  if (!tabs.length) tabs.push(newTab());
  if (id === activeId) activeId = tabs[Math.min(i, tabs.length - 1)].id;
  renderView();
  save();
}

// Clear the active tab: query, results and answer; the filters stay. In Browse it clears the title filter.
function clearResults() {
  const tab = activeTab();
  if (!tab) return;
  if (tab.view === "browse") {
    tab.filter = "";
    $("q").value = "";
    renderBrowse(tab);
    updateUrl(tab);
    save();
    $("q").focus();
    return;
  }
  controllers[tab.id]?.abort();
  delete controllers[tab.id];
  Object.assign(tab, {q: "", trace: "", data: null, answer: null, status: "", error: "", busy: false, open: {}});
  $("q").value = "";
  $("trace").value = "";
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

// "422 {"detail":[{"msg": ...}]}" -> "422: String should have at most 500 characters"
async function httpError(res) {
  const body = await res.text();
  let detail = body;
  try {
    const d = JSON.parse(body).detail;
    detail = Array.isArray(d) ? d.map((x) => x.msg).join("; ") : String(d ?? body);
  } catch { /* not JSON */ }
  return new Error(`${res.status}: ${detail}`.slice(0, 300));
}

// The events after the evidence, shared by the documentation and the code answers: delta* -> done | error.
function applyAnswerEvent(tab, ev) {
  const a = tab.answer;
  if (ev.type === "delta") {
    a.text += ev.text;
    if (isActive(tab)) renderAnswerText(a);
    saveSoon();
    return;
  }
  if (ev.type === "sources") a.sources = ev.sources;
  else if (ev.type === "done") {
    a.streaming = false;
    a.model = ev.model || a.model;
    a.footer = ev.model ? `${(ev.took_ms / 1000).toFixed(1)} s · based on ${a.sources.length} source${a.sources.length === 1 ? "" : "s"}` : "";
    a.usage = ev.usage || null;
    a.firstTokenMs = ev.first_token_ms ?? null;
    if (ev.truncated) a.error = "The answer was cut off: it reached the max_tokens limit (RAG_LLM_MAX_TOKENS).";
  } else if (ev.type === "error") { a.streaming = false; a.error = "LLM error: " + ev.message; }
  if (!isActive(tab)) return;
  renderAnswer(a);
  if (ev.type === "sources") { if (tab.mode === "code") renderCode(tab.data, tab); else markSources(a.sources); }
}

async function runAnswer(tab, params, signal) {
  if (tab.model) params.set("model", tab.model);
  if (langParam(tab)) params.set("lang", langParam(tab));
  const res = await fetch("/api/answer?" + params, {signal});
  if (!res.ok) throw await httpError(res);
  await readNdjson(res, (ev) => {
    if (ev.type !== "results") { applyAnswerEvent(tab, ev); return; }
    tab.data = ev;
    tab.status = statusOf(ev);
    if (isActive(tab)) { $("status").textContent = tab.status; renderResults(ev, tab); }
  });
  const a = tab.answer;
  if (a.streaming) { a.streaming = false; a.error = "The connection closed before the answer was complete."; }
}

// Code mode: the evidence for the traceback arrives first; with the LLM on, the explanation streams after it.
async function runCode(tab, useLlm, signal) {
  const body = {question: tab.q, traceback: tab.trace, modules: tab.module ? [tab.module] : [],
    client_version: tab.version, docs: tab.docs, answer: useLlm, language: langParam(tab) || null};
  if (useLlm && tab.model) body.model = tab.model;
  const res = await fetch("/api/code/answer", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body), signal});
  if (!res.ok) throw await httpError(res);
  await readNdjson(res, (ev) => {
    if (ev.type !== "results") { applyAnswerEvent(tab, ev); return; }
    tab.data = ev;
    tab.status = codeStatus(ev);
    if (isActive(tab)) { $("status").textContent = tab.status; renderCode(ev, tab); }
  });
  const a = tab.answer;
  if (a?.streaming) { a.streaming = false; a.error = "The connection closed before the answer was complete."; }
}

// Stop the request of the active tab. What has streamed so far stays on the screen.
function stopRun() {
  const tab = activeTab();
  const controller = tab && controllers[tab.id];
  if (!controller) return;
  delete controllers[tab.id];
  controller.abort();
  tab.busy = false;
  if (tab.answer?.streaming) Object.assign(tab.answer, {streaming: false, error: "Stopped."});
  if (!tab.data) tab.status = "Stopped.";
  renderTabs();
  renderContent(tab);
  save();
}

async function run() {
  const tab = activeTab();
  if (!tab || tab.view === "browse") return; // Browse filters as the user types: Enter has nothing to run
  readForm(tab);
  const code = tab.mode === "code";
  if (code ? !tab.q && !tab.trace.trim() : !tab.q) return;
  if (code) loadCodeModules(); else loadProducts(); // retried here if the first load failed
  const useLlm = tab.llm && llmInfo.available;
  const params = new URLSearchParams({q: tab.q, ...SEARCH_PARAMS});
  if (tab.product) params.set("product", tab.product);

  controllers[tab.id]?.abort();
  const controller = controllers[tab.id] = new AbortController();
  Object.assign(tab, {status: "Searching…", error: "", data: null, busy: true, open: {}, ran: settingsOf(tab),
    answer: useLlm ? {text: "", sources: [], streaming: true, model: tab.model || llmInfo.model} : null});
  renderTabs();
  renderContent(tab);
  updateUrl(tab);
  try {
    if (code) {
      await runCode(tab, useLlm, controller.signal);
    } else if (useLlm) {
      await runAnswer(tab, params, controller.signal);
    } else {
      const res = await fetch("/api/search?" + params, {signal: controller.signal});
      if (!res.ok) throw await httpError(res);
      tab.data = await res.json();
      tab.status = statusOf(tab.data);
    }
  } catch (err) {
    if (err.name === "AbortError") return; // superseded by a newer request or tab closed
    tab.error = err.message;
    if (!tab.data) tab.status = "";
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

const LLM_HELP = $("llm-label").title; // the hover help from index.html

async function refreshLlmStatus() {
  const info = await fetch("/api/llm").then((r) => (r.ok ? r.json() : Promise.reject(r)))
    .catch(() => ({available: false, model: null, models: [], error: "backend not reachable"}));
  const selected = llmInfo.checked ? $("model").value : activeTab()?.model || "";
  llmInfo = {...info, checked: true};
  $("model").innerHTML = `<option value="">Default (${escapeHtml(llmInfo.model || "none")})</option>` +
    llmInfo.models.map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.id)}${m.loaded === false ? " (not loaded)" : ""}</option>`).join("");
  $("model").value = [...$("model").options].some((o) => o.value === selected) ? selected : "";
  $("llm").disabled = $("model").disabled = $("lang").disabled = !llmInfo.available;
  // With the Claude API the question and the doc excerpts leave this machine: say so where the switch is.
  const external = llmInfo.provider === "anthropic";
  $("llm-provider").hidden = !external;
  $("llm-label").title = LLM_HELP +
    (external ? "\n\nThis server answers with the Claude API: the question and the doc excerpts are sent to Anthropic." : "") +
    (llmInfo.available ? "" : `\n\nUnavailable now: ${llmInfo.error}`);
  if (activeTab()) updateSubmitLabel(activeTab());
}

// --- answer languages -----------------------------------------------------------

async function loadLanguages() {
  try {
    const res = await fetch("/api/languages");
    if (!res.ok) throw await httpError(res);
    const body = await res.json();
    languages = body.languages;
    defaultLang = body.default;
    languagesLoaded = true;
  } catch { return; } // keep English only: the answer then uses the server's default
  const tab = activeTab();
  if (tab) fillLanguages(tab);
}

// --- init ---------------------------------------------------------------------

const flag = (v, fallback) => (v === null ? fallback : ["1", "true", "yes", "on"].includes(v.toLowerCase()));

// A shared link (?q=...) shows the tab with exactly these settings (the active one first),
// or opens a new tab and runs it.
// ?view=browse[&product=...&filter=...] opens the article list the same way (nothing to run).
function openFromUrl() {
  const params = new URLSearchParams(location.search);
  const browse = params.get("view") === "browse";
  const code = params.get("mode") === "code";
  if (!browse && !code && !params.get("q")) return false;
  const wanted = code
    ? sanitize({mode: "code", q: (params.get("q") || "").trim(), module: params.get("module") || ""})
    : browse
    ? sanitize({view: "browse", product: params.get("product") || "", filter: params.get("filter") || ""})
    : sanitize({q: params.get("q").trim(), product: params.get("product") || "", llm: flag(params.get("llm"), false),
        model: params.get("model") || "", lang: params.get("lang") || ""}); // mode / rerank in older links are ignored
  const same = code
    ? (t) => t.mode === "code" && t.q === wanted.q && t.module === wanted.module
    : browse
    ? (t) => t.mode === "docs" && t.view === "browse" && t.product === wanted.product && t.filter === wanted.filter
    : (t) => t.mode === "docs" && t.view === "search" &&
        SETTINGS.every((k) => t[k] === wanted[k] || (LLM_ONLY.includes(k) && !t.llm) || CODE_FIELDS.includes(k));
  const runs = code ? !!wanted.q : !browse; // a link to Code mode runs only when it carries a question
  const existing = [activeTab(), ...tabs].find((t) => t && same(t));
  if (existing) { activeId = existing.id; return runs && !existing.data; }
  tabs.push(wanted);
  activeId = wanted.id;
  return runs;
}

async function init() {
  // Tabs first, so the form works right away; the options load after.
  load();
  const runNow = openFromUrl();
  if (!tabs.length) { tabs.push(newTab()); activeId = tabs[0].id; }
  $("debug").setAttribute("aria-pressed", String(debug));
  applyTheme();
  renderView();
  markFilterRows();
  refreshLlmStatus(); // LM Studio may be slow to answer: don't wait for it
  loadLanguages();
  if (activeTab().view === "browse") loadPages();
  if (activeTab().mode === "code") loadCodeModules();
  await loadProducts();
  if (runNow) run();
}

$("form").addEventListener("submit", (e) => { e.preventDefault(); run(); });
$("stop").addEventListener("click", stopRun);
$("clear").addEventListener("click", clearResults);
for (const pos of POSITIONS) $(`pos-${pos}`).addEventListener("click", () => setPosition(pos));
$("trace").addEventListener("input", () => {  // saved without the settings round trip: a traceback is long
  const tab = activeTab();
  if (!tab) return;
  tab.trace = $("trace").value.slice(0, TRACE_MAX);
  $("status").textContent = statusText(tab);
  saveSoon();
});
$("trace").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); run(); }
});
// A traceback pasted into the documentation search box (it holds 500 characters) opens Code mode with it.
$("q").addEventListener("paste", (e) => {
  const tab = activeTab();
  const text = e.clipboardData?.getData("text") || "";
  if (!tab || tab.mode !== "docs" || tab.view !== "search" || !looksLikeTraceback(text)) return;
  e.preventDefault();
  setMode("code");
  tab.trace = text.slice(0, TRACE_MAX);
  tab.status = "That looks like a traceback: opened in Code mode.";
  $("trace").value = tab.trace;
  renderContent(tab);
  save();
});
$("results").addEventListener("click", (e) => {
  if (!e.target.closest("[data-browse]")) return;
  e.preventDefault();
  setView("browse");
});
$("q").addEventListener("input", () => {  // Browse filters as the user types
  const tab = activeTab();
  if (tab?.view !== "browse") return;
  tab.filter = $("q").value.slice(0, 500);
  renderBrowse(tab);
  updateUrl(tab);
  saveSoon();
});
$("debug").addEventListener("click", () => setDebug(!debug));
$("theme").addEventListener("click", toggleTheme);
{ // the sticky block's height, for what has to stick or scroll to just below it
  const chrome = $("chrome");
  const publish = () => document.documentElement.style.setProperty(
    "--chrome-h", getComputedStyle(chrome).position === "sticky" ? `${chrome.offsetHeight}px` : "0px");
  new ResizeObserver(publish).observe(chrome);
  publish();
}
{ // the row breaks move when the window or a filter (e.g. the model list) changes width
  const rows = new ResizeObserver(markFilterRows);
  rows.observe(document.querySelector(".filters"));
  for (const el of document.querySelector(".filters").children) rows.observe(el);
}
systemDark.addEventListener("change", () => { if (!savedTheme()) applyTheme(); });
window.addEventListener("storage", (e) => { if (e.key === THEME_KEY) applyTheme(); });
["product", "llm", "model", "lang", "code-module", "code-docs"].forEach((id) => $(id).addEventListener("change", onSettingsChange));
$("client-version").addEventListener("input", onSettingsChange);
$("tabs").addEventListener("click", (e) => {
  const close = e.target.closest("[data-close]");
  if (close) { e.stopPropagation(); closeTab(close.dataset.close); return; }
  if (e.target.closest("[data-add]")) { addTab(); return; }
  const tab = e.target.closest(".tab");
  if (tab) switchTab(tab.dataset.id);
});
// Keyboard: Enter / Space selects the focused tab, arrows move between tabs.
$("tabs").addEventListener("keydown", (e) => {
  const tab = e.target.closest(".tab");
  if (!tab || e.target.closest("button")) return;
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); switchTab(tab.dataset.id); }
  else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const i = tabs.findIndex((t) => t.id === tab.dataset.id) + (e.key === "ArrowRight" ? 1 : -1);
    if (tabs[i]) { switchTab(tabs[i].id); $("tabs").querySelector(`[data-id="${tabs[i].id}"]`)?.focus(); }
  }
});
// Middle click closes a tab, like in a browser.
$("tabs").addEventListener("auxclick", (e) => {
  const tab = e.target.closest(".tab");
  if (e.button === 1 && tab && tabs.length > 1) closeTab(tab.dataset.id);
});
// Remember expanded / collapsed cards and answer per tab ("toggle" doesn't bubble: capture it).
// Re-rendering an open <details> fires "toggle" too: only a real change is saved.
document.addEventListener("toggle", (e) => {
  const el = e.target, tab = activeTab();
  if (!tab || !el.isConnected) return;
  if (el.matches(".result")) {
    const cards = [...document.querySelectorAll(".result")];
    if (isOpen(tab, el.dataset.pageId, cards.indexOf(el)) === el.open) return;
    (tab.open ||= {})[el.dataset.pageId] = el.open;
  } else if (el.matches(".answer-panel") && tab.answer) {
    if ((tab.answer.open !== false) === el.open) return;
    tab.answer.open = el.open;
  } else return;
  save();
}, true);
// A citation [n] expands the source card and scrolls to it smoothly (the gap above it is the
// card's scroll-margin-top); the URL keeps no #src-n, so a reload doesn't jump.
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
$("answer").addEventListener("click", (e) => {
  const cite = e.target.closest("a.cite");
  const target = cite && document.querySelector(cite.getAttribute("href"));
  if (!target) return;
  e.preventDefault();
  document.querySelectorAll(".result.cited").forEach((el) => el.classList.remove("cited"));
  target.classList.add("cited");
  target.open = true;
  target.scrollIntoView({behavior: reducedMotion.matches ? "auto" : "smooth", block: "start"});
  target.querySelector("summary")?.focus({preventScroll: true});
});
// Models may be loaded in LM Studio after the page was opened.
$("llm-label").addEventListener("mouseenter", refreshLlmStatus);
$("refresh-models").addEventListener("click", async () => {
  const button = $("refresh-models");
  button.classList.add("spinning");
  try {  // spin at least one turn, so a fast answer still shows that something happened
    await Promise.all([refreshLlmStatus(), new Promise((r) => setTimeout(r, 600))]);
  } finally {
    button.classList.remove("spinning");
  }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== $("q")) { e.preventDefault(); $("q").focus(); }
});
init();
