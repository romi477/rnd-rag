const $ = (id) => document.getElementById(id);
const STOP = new Set(("a an and are as at be by can could do does did for from has have how i in into is it my not of on or " +
  "our should that the there this to was were what when where which why will with would you your").split(" "));

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

function render(data) {
  const ts = terms(data.query);
  $("status").textContent = `${data.results.length} pages · ${data.took_ms} ms`;
  if (!data.results.length) {
    $("results").innerHTML = '<div class="empty">Nothing found. Try other words or switch the mode.</div>';
    return;
  }
  $("results").innerHTML = data.results.map((p) => {
    const crumbs = p.breadcrumbs.filter((c) => c !== "Home").join(" › ") || p.product;
    const updated = p.updated ? new Date(p.updated).toLocaleDateString() : "";
    const chunks = p.chunks.map((c) => `
      <div class="chunk">
        ${c.section_path.length ? `<div class="section">§ ${escapeHtml(c.section_path.join(" › "))}</div>` : ""}
        <div class="snippet">${highlight(snippet(c.text, ts), ts)}</div>
        ${c.text.length > 420 ? `<details><summary>Show full fragment</summary><div class="full">${highlight(clean(c.text), ts)}</div></details>` : ""}
      </div>`).join("");
    return `
      <article class="result">
        <div class="crumbs">${escapeHtml(crumbs)}</div>
        <h2><a href="${escapeHtml(p.url)}" target="_blank" rel="noopener">${escapeHtml(p.title)}</a></h2>
        <div class="meta"><span>Updated ${updated}</span>${p.labels.length ? `<span>${p.labels.map(escapeHtml).join(", ")}</span>` : ""}<span>score ${p.score.toFixed(3)}</span></div>
        ${chunks}
      </article>`;
  }).join("");
}

async function run() {
  const q = $("q").value.trim();
  if (!q) return;
  const params = new URLSearchParams({q, mode: $("mode").value, rerank: $("rerank").checked});
  if ($("product").value) params.set("product", $("product").value);
  history.replaceState(null, "", "?" + params);
  $("status").textContent = "Searching…";
  try {
    const res = await fetch("/api/search?" + params);
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    render(await res.json());
  } catch (err) {
    $("status").textContent = "Error: " + err.message;
  }
}

async function init() {
  const products = await fetch("/api/products").then((r) => r.json()).catch(() => []);
  $("product").insertAdjacentHTML("beforeend", products.map((p) => `<option>${escapeHtml(p)}</option>`).join(""));
  const params = new URLSearchParams(location.search);
  if (params.get("product")) $("product").value = params.get("product");
  if (params.get("mode")) $("mode").value = params.get("mode");
  $("rerank").checked = params.get("rerank") === "true";
  if (params.get("q")) { $("q").value = params.get("q"); run(); }
}

$("form").addEventListener("submit", (e) => { e.preventDefault(); run(); });
["product", "mode", "rerank"].forEach((id) => $(id).addEventListener("change", run));
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== $("q")) { e.preventDefault(); $("q").focus(); }
});
init();
