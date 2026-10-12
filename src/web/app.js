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
const MODES = ["hybrid", "vector", "fts"];
const VIEWS = ["search", "browse"];
const SEARCH_PLACEHOLDER = $("q").placeholder;
let metaError = ""; // the product list could not be loaded
// Browse: every indexed article (GET /api/pages), loaded once and shared by all tabs.
let pages = null;
let pagesError = "";
let pagesPromise = null;

const newId = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

// view: "search" or "browse" (the title filter of the article list is `filter`, kept apart from the query `q`).
function newTab(init = {}) {
  return {id: newId(), view: "search", q: "", filter: "", product: "", mode: "hybrid", rerank: true, llm: false, model: "",
          data: null, answer: null, status: "", error: "", busy: false, ...init};
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
  if (!MODES.includes(tab.mode)) tab.mode = "hybrid";
  if (!VIEWS.includes(tab.view)) tab.view = "search";
  for (const k of ["q", "filter", "product", "model", "error"]) if (typeof tab[k] !== "string") tab[k] = "";
  tab.q = tab.q.slice(0, 500);
  tab.filter = tab.filter.slice(0, 500);
  for (const k of ["rerank", "llm"]) tab[k] = !!tab[k];
  return tab;
}

const hasOption = (select, value) => [...$(select).options].some((o) => o.value === value);

// The one input holds the query in the Search view and the title filter in Browse: each keeps its own text.
function readForm(tab) {
  if (tab.view === "browse") tab.filter = $("q").value.slice(0, 500);
  else tab.q = $("q").value.trim();
  tab.product = $("product").value;
  tab.mode = $("mode").value;
  tab.rerank = $("rerank").checked;
  tab.llm = $("llm").checked;
  if (llmInfo.checked) tab.model = $("model").value;
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

function updatePlaceholder(tab) {
  $("q").placeholder = tab.view !== "browse" ? SEARCH_PLACEHOLDER
    : pages ? `Filter ${pages.length} articles by title…` : "Filter articles by title…";
}

function writeForm(tab) {
  const browse = tab.view === "browse";
  fillProducts(tab);
  // a model that no longer exists falls back to the default model
  if (llmInfo.checked && !hasOption("model", tab.model)) tab.model = "";
  $("q").value = browse ? tab.filter : tab.q;
  updatePlaceholder(tab);
  $("mode").value = tab.mode;
  $("rerank").checked = tab.rerank;
  $("llm").checked = tab.llm;
  $("model").value = hasOption("model", tab.model) ? tab.model : "";
  for (const view of VIEWS) $(`view-${view}`).setAttribute("aria-pressed", String(tab.view === view));
  updateSubmitLabel(tab);
}

// Only the Find / Ask button runs a query; changing settings just marks the results as outdated.
// The model matters only when the answer is written by the LLM.
const SETTINGS = ["q", "product", "mode", "rerank", "llm", "model"];
const settingsOf = (tab) => Object.fromEntries(SETTINGS.map((k) => [k, tab[k]]));
const isStale = (tab) => !!(tab.data && tab.ran && SETTINGS.some((k) => tab.ran[k] !== tab[k] && (k !== "model" || tab.llm)));

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

function inlineMd(text, sources) {
  // Code spans first, into placeholders: their content is literal (no *emphasis*, no [n] links).
  const codes = [];
  const html = escapeHtml(text.replace(/\$?\\(?:long)?(?:right)?arrow\$?|\$\\to\$/gi, "→"))
    .replace(/`([^`\n]+)`/g, (m, code) => `\u0000${codes.push(code) - 1}\u0000`)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    // *em*: opens after a space or line start, closes before a space or punctuation, so that
    // "*.pdf and *.xml" stays as typed
    .replace(/(^|[\s(])\*([^*\s](?:[^*\n]*[^*\s])?)\*(?=[\s.,;:!?)]|$)/g, "$1<em>$2</em>")
    .replace(/\[(\d+)\]/g, (m, n) => {
      const src = sources[n - 1];
      return src ? `<a class="cite" href="#src-${n}" title="${escapeHtml(src.title)}">${n}</a>` : m;
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
  </details>`;
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
    <div class="tab${t.id === activeId ? " active" : ""}" data-id="${t.id}" title="${escapeHtml(t.view === "browse" ? `Browse: ${t.filter || "all articles"}` : t.q || "New search")}" role="tab" aria-selected="${t.id === activeId}" tabindex="${t.id === activeId ? 0 : -1}">
      ${t.busy ? '<span class="tab-busy" aria-label="loading"></span>' : ""}
      <span class="tab-title">Tab${i + 1}</span>
      ${t.view === "browse" ? '<span class="tab-model">browse</span>' : t.llm && t.model ? `<span class="tab-model">${escapeHtml(t.model.split("/").pop())}</span>` : ""}
      ${tabs.length > 1 ? `<button class="tab-close" data-close="${t.id}" aria-label="Close tab" title="Close tab">×</button>` : ""}
    </div>`).join("") +
    '<button class="tab-add" data-add title="New tab" aria-label="New tab">+</button>';
}

// A problem the user should notice: the last request of the tab failed, or a list could not be loaded.
function renderBanner(tab) {
  const text = (tab.view === "browse" ? pagesError : tab.error) || metaError;
  $("banner").textContent = text;
  $("banner").hidden = !text;
}

// Results + answer of a tab (the form is written only on tab switch, so typing isn't clobbered).
function renderContent(tab) {
  document.body.dataset.view = tab.view; // CSS shows the parts of the page that belong to the view
  renderBanner(tab);
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
  if (tab.view === "browse") {
    const params = new URLSearchParams({view: "browse"});
    if (tab.product) params.set("product", tab.product);
    if (tab.filter) params.set("filter", tab.filter);
    history.replaceState(null, "", "?" + params);
    return;
  }
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
  Object.assign(tab, {q: "", data: null, answer: null, status: "", error: "", busy: false, open: {}});
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

async function runAnswer(tab, params, signal) {
  if (tab.model) params.set("model", tab.model);
  const res = await fetch("/api/answer?" + params, {signal});
  if (!res.ok) throw await httpError(res);
  const a = tab.answer;
  await readNdjson(res, (ev) => {
    if (ev.type === "results") {
      tab.data = ev;
      tab.status = statusOf(ev);
      if (isActive(tab)) { $("status").textContent = tab.status; renderResults(ev, tab); }
      return;
    }
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
      if (ev.truncated) a.error = "The answer was cut off: it reached the max_tokens limit (RAG_LLM_MAX_TOKENS).";
    } else if (ev.type === "error") { a.streaming = false; a.error = "LLM error: " + ev.message; }
    if (isActive(tab)) { renderAnswer(a); if (ev.type === "sources") markSources(a.sources); }
  });
  if (a.streaming) { a.streaming = false; a.error = "The connection closed before the answer was complete."; }
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
  if (!tab.q) return;
  loadProducts(); // retried here if the first load failed
  const useLlm = tab.llm && llmInfo.available;
  const params = new URLSearchParams({q: tab.q, mode: tab.mode, rerank: tab.rerank});
  if (tab.product) params.set("product", tab.product);

  controllers[tab.id]?.abort();
  const controller = controllers[tab.id] = new AbortController();
  Object.assign(tab, {status: "Searching…", error: "", data: null, busy: true, open: {}, ran: settingsOf(tab),
    answer: useLlm ? {text: "", sources: [], streaming: true, model: tab.model || llmInfo.model} : null});
  renderTabs();
  renderContent(tab);
  updateUrl(tab);
  try {
    if (useLlm) {
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
  $("llm").disabled = $("model").disabled = !llmInfo.available;
  $("llm-label").title = llmInfo.available ? LLM_HELP : `${LLM_HELP}\n\nUnavailable now: ${llmInfo.error}`;
  if (activeTab()) updateSubmitLabel(activeTab());
}

// --- init ---------------------------------------------------------------------

const flag = (v, fallback) => (v === null ? fallback : ["1", "true", "yes", "on"].includes(v.toLowerCase()));

// A shared link (?q=...) shows the tab with exactly these settings (the active one first),
// or opens a new tab and runs it.
// ?view=browse[&product=...&filter=...] opens the article list the same way (nothing to run).
function openFromUrl() {
  const params = new URLSearchParams(location.search);
  const browse = params.get("view") === "browse";
  if (!browse && !params.get("q")) return false;
  const wanted = browse
    ? sanitize({view: "browse", product: params.get("product") || "", filter: params.get("filter") || ""})
    : sanitize({q: params.get("q").trim(), product: params.get("product") || "", mode: params.get("mode") || "hybrid",
        rerank: flag(params.get("rerank"), true), llm: flag(params.get("llm"), false), model: params.get("model") || ""});
  const same = browse
    ? (t) => t.view === "browse" && t.product === wanted.product && t.filter === wanted.filter
    : (t) => t.view === "search" && SETTINGS.every((k) => t[k] === wanted[k] || (k === "model" && !t.llm));
  const existing = [activeTab(), ...tabs].find((t) => t && same(t));
  if (existing) { activeId = existing.id; return !browse && !existing.data; }
  tabs.push(wanted);
  activeId = wanted.id;
  return !browse;
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
  if (activeTab().view === "browse") loadPages();
  await loadProducts();
  if (runNow) run();
}

$("form").addEventListener("submit", (e) => { e.preventDefault(); run(); });
$("stop").addEventListener("click", stopRun);
$("clear").addEventListener("click", clearResults);
for (const view of VIEWS) $(`view-${view}`).addEventListener("click", () => setView(view));
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
["product", "mode", "rerank", "llm", "model"].forEach((id) => $(id).addEventListener("change", onSettingsChange));
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
