"""Sync a Confluence Cloud space into raw-data/ for the RAG pipeline.

Layout (per space):
    raw-data/confluence/<space>/
        pages/<id>.storage.html   original storage-format body (lossless)
        pages/<id>.md             markdown with YAML frontmatter (for chunking)
        pages/<id>.json           page metadata (version, parent, breadcrumbs, url)
        attachments/<id>/<attachment id>-<file>   the images a page shows (only with --attachments), `_index.json` maps titles to files
        _manifest.json            id -> {title, version, path}

Incremental: a page is rewritten when its version or its breadcrumbs (a parent renamed or moved)
changed. Files are written atomically; requests are retried on 429 / 5xx.
Pages that disappeared from the space are removed locally, but a sync that would remove more than
a quarter of the pages stops instead (an empty or partial listing), unless --force.

Usage (token falls back to $JIRA_TOKEN_VENT, defined in interactive zsh):
    zsh -ic 'uv run rag sync'                # sync space from .env
    zsh -ic 'uv run rag sync --attachments'  # also download the images the pages show
    zsh -ic 'uv run rag sync --force'        # rewrite everything
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import httpx
from markdownify import markdownify

RETRY_STATUS = {429, 500, 502, 503, 504}
RETRIES = 5
# Refuse to delete more local pages than this in one sync, unless --force (a broken or partial
# listing must not wipe the corpus, and with it the index).
MAX_REMOVED_SHARE = 0.25


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if not value:
        sys.exit(f"Missing env var {name} (see .env.example)")
    return value


class Confluence:
    def __init__(self, base_url: str, email: str, token: str):
        self.base = base_url.rstrip("/")
        self.http = httpx.Client(
            auth=(email, token),
            headers={"Accept": "application/json"},
            timeout=60,
            follow_redirects=True,
        )

    def get(self, url: str, **params) -> httpx.Response:
        if url.startswith("/"):
            url = f"{self.base}{url}"
        for attempt in range(RETRIES + 1):
            try:
                resp = self.http.get(url, params=params or None)
            except httpx.TransportError:
                if attempt == RETRIES:
                    raise
                time.sleep(2**attempt)
                continue
            if resp.status_code not in RETRY_STATUS or attempt == RETRIES:
                break
            time.sleep(retry_delay(resp, attempt))
        if resp.status_code == 401:
            sys.exit("401 Unauthorized: check CONFLUENCE_EMAIL / CONFLUENCE_API_TOKEN")
        resp.raise_for_status()
        return resp

    def download(self, url: str, dest: Path) -> None:
        """Stream a file to disk (atomically), without holding it in memory."""
        tmp = dest.with_name(dest.name + ".part")
        with self.http.stream("GET", f"{self.base}{url}") as resp:
            resp.raise_for_status()
            with tmp.open("wb") as f:
                for block in resp.iter_bytes():
                    f.write(block)
        tmp.replace(dest)

    def paginate(self, path: str, **params):
        """Follow v2 cursor pagination via _links.next."""
        url, query = path, params
        while url:
            data = self.get(url, **query).json()
            yield from data.get("results", [])
            url, query = data.get("_links", {}).get("next"), {}

    def space_id(self, key: str) -> str:
        results = self.get("/wiki/api/v2/spaces", keys=key).json()["results"]
        if not results:
            sys.exit(f"Space '{key}' not found or not accessible")
        return results[0]["id"]

    def pages(self, space_id: str):
        return self.paginate(
            f"/wiki/api/v2/spaces/{space_id}/pages",
            **{"body-format": "storage", "limit": 250, "status": "current"},
        )

    def labels(self, page_id: str) -> list[str]:
        return [l["name"] for l in self.paginate(f"/wiki/api/v2/pages/{page_id}/labels", limit=250)]

    def attachments(self, page_id: str):
        return self.paginate(f"/wiki/api/v2/pages/{page_id}/attachments", limit=250)


def retry_delay(resp: httpx.Response, attempt: int) -> float:
    """Seconds to wait: the server's Retry-After if given, else exponential backoff."""
    try:
        return min(float(resp.headers.get("Retry-After", "")), 60.0)
    except ValueError:
        return float(2**attempt)


# --- storage format -> markdown -------------------------------------------------
# Confluence tags may be self-closing (`<ac:parameter ac:name="x" />`): every pattern that spans
# an element accepts both forms, otherwise it would swallow everything up to the next closing tag.

CODE_MACRO = re.compile(
    r'<ac:structured-macro\b[^>]*ac:name="code"[^>]*?(?:/>|>(.*?)</ac:structured-macro>)', re.S
)
CODE_LANG = re.compile(r'<ac:parameter ac:name="language">([^<]*)</ac:parameter>')
CODE_BODY = re.compile(r"<ac:plain-text-body><!\[CDATA\[(.*?)\]\]></ac:plain-text-body>", re.S)
IMAGE = re.compile(r"<ac:image\b[^>]*?(?:/>|>(.*?)</ac:image>)", re.S)
IMAGE_FILE = re.compile(r'ri:filename="([^"]+)"')
IMAGE_URL = re.compile(r'ri:value="([^"]+)"')
CAPTION = re.compile(r"<ac:caption>(.*?)</ac:caption>", re.S)
LINK = re.compile(r"<ac:link\b[^>]*?(?:/>|>(.*?)</ac:link>)", re.S)
LINK_BODY = re.compile(r"<ac:(?:plain-text-)?link-body>(.*?)</ac:(?:plain-text-)?link-body>", re.S)
LINK_TARGET = re.compile(r'ri:(?:content-title|filename|value|space-key)="([^"]+)"')
# The new editor stores panels as ADF extensions: attributes, the content, and an HTML fallback
# that repeats the content.
ADF_EXTENSION = re.compile(r"<ac:adf-extension>(.*?)</ac:adf-extension>", re.S)
ADF_PANEL_TYPE = re.compile(r'<ac:adf-attribute key="panel-type">([^<]*)</ac:adf-attribute>')
ADF_ATTRIBUTE = re.compile(r"<ac:adf-attribute\b[^>]*?(?:/>|>.*?</ac:adf-attribute>)", re.S)
ADF_FALLBACK = re.compile(r"<ac:adf-fallback>.*?</ac:adf-fallback>", re.S)
TASK = re.compile(r"<ac:task>(.*?)</ac:task>", re.S)
TASK_STATUS = re.compile(r"<ac:task-status>([^<]*)</ac:task-status>")
TASK_BODY = re.compile(r"<ac:task-body>(.*?)</ac:task-body>", re.S)
TASK_LIST = re.compile(r"</?ac:task-list>")
PANEL_MACRO = re.compile(
    r'<ac:structured-macro[^>]*ac:name="(info|note|warning|tip|panel|expand)"[^>]*>', re.S
)
TITLE_PARAM = re.compile(r'<ac:parameter ac:name="title">([^<]+)</ac:parameter>')
CDATA = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.S)
MACRO_PARAM = re.compile(r"<ac:parameter\b[^>]*?(?:/>|>.*?</ac:parameter>)", re.S)
FENCE_LINE = re.compile(r"^\s*(```|~~~)")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _code(match: re.Match) -> str:
    inner = match.group(1) or ""
    lang = CODE_LANG.search(inner)
    body = CODE_BODY.search(inner)
    cls = f' class="language-{lang.group(1)}"' if lang else ""
    return f"<pre><code{cls}>{_escape(body.group(1)) if body else ''}</code></pre>"


def referenced_images(storage: str) -> set[str]:
    """File names of the attachments a page shows as images (`<ac:image>`); pictures from a URL are not attachments."""
    names = (IMAGE_FILE.search(m.group(0)) for m in IMAGE.finditer(storage))
    return {m.group(1) for m in names if m}


def _image(match: re.Match) -> str:
    tag = match.group(0)
    src = IMAGE_FILE.search(tag) or IMAGE_URL.search(tag)
    caption = CAPTION.search(match.group(1) or "")
    out = f"<p>[image: {src.group(1)}]</p>" if src else ""
    return out + (f"<p><em>{caption.group(1)}</em></p>" if caption else "")


def _link(match: re.Match) -> str:
    inner = match.group(1) or ""
    body = LINK_BODY.search(inner)
    if body:
        return body.group(1)
    target = LINK_TARGET.search(match.group(0))
    return _escape(target.group(1)) if target else ""


def _adf(match: re.Match) -> str:
    inner = match.group(1)
    if "<ac:adf-content" in inner:
        inner = ADF_FALLBACK.sub("", inner)
    panel = ADF_PANEL_TYPE.search(inner)
    inner = ADF_ATTRIBUTE.sub("", inner)
    return (f"<p><strong>{panel.group(1).upper()}:</strong></p>" if panel else "") + inner


def _task(match: re.Match) -> str:
    inner = match.group(1)
    status, body = TASK_STATUS.search(inner), TASK_BODY.search(inner)
    box = "[x]" if status and status.group(1) == "complete" else "[ ]"
    return f"<li>{box} {body.group(1) if body else ''}</li>"


def collapse_blank_lines(md: str) -> str:
    """Squeeze runs of blank lines to one, except inside fenced code blocks."""
    out, blank, in_fence = [], 0, False
    for line in md.splitlines():
        if FENCE_LINE.match(line):
            in_fence = not in_fence
        blank = blank + 1 if not line.strip() else 0
        if in_fence or blank <= 1:
            out.append(line)
    return "\n".join(out)


def storage_to_markdown(html: str) -> str:
    html = CODE_MACRO.sub(_code, html)
    html = IMAGE.sub(_image, html)
    html = LINK.sub(_link, html)
    html = ADF_EXTENSION.sub(_adf, html)
    html = TASK.sub(_task, html)
    html = TASK_LIST.sub(lambda m: "</ul>" if m.group(0).startswith("</") else "<ul>", html)
    html = PANEL_MACRO.sub(lambda m: f"<p><strong>{m.group(1).upper()}:</strong></p>", html)
    html = TITLE_PARAM.sub(r"<p><strong>\1</strong></p>", html)
    html = MACRO_PARAM.sub("", html)
    html = CDATA.sub(lambda m: _escape(m.group(1)), html)
    md = markdownify(
        html, heading_style="ATX", bullets="-", code_language_callback=_lang, escape_underscores=False
    )
    return collapse_blank_lines(md).strip() + "\n"


def _lang(el) -> str | None:
    code = el.find("code")
    for cls in (code.get("class") or []) if code else []:
        if cls.startswith("language-"):
            return cls.removeprefix("language-")
    return None


def frontmatter(meta: dict) -> str:
    lines = ["---"]
    for key in ("id", "title", "space", "url", "version", "updated", "breadcrumbs", "labels"):
        lines.append(f"{key}: {json.dumps(meta[key], ensure_ascii=False)}")
    lines.append("---\n")
    return "\n".join(lines)


# --- sync -------------------------------------------------------------------------


def safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name)[:200]


def write_atomic(path: Path, text: str) -> None:
    """Write via a temp file and rename, so an interrupted sync never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def local_pages(pages_dir: Path) -> dict[str, dict]:
    """Pages on disk (id -> metadata), whatever the manifest says: a sync interrupted before
    writing the manifest still leaves its pages known, so they can be updated or removed later."""
    pages = {}
    for path in pages_dir.glob("*.json"):
        try:
            pages[path.stem] = json.loads(path.read_text())
        except (OSError, ValueError):
            pages[path.stem] = {}  # unreadable: rewrite it
    return pages


def sync(space: str, out_root: Path, attachments: bool = False, force: bool = False) -> None:
    """Download new and changed pages; with `force`, rewrite all and allow mass removals."""
    base_url = env("CONFLUENCE_BASE_URL", "https://ventortech.atlassian.net")
    token = os.environ.get("CONFLUENCE_API_TOKEN") or env("JIRA_TOKEN_VENT")
    api = Confluence(base_url, env("CONFLUENCE_EMAIL"), token)

    out = out_root / space
    pages_dir = out / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "_manifest.json"
    known = local_pages(pages_dir)

    space_id = api.space_id(space)
    print(f"Space {space} -> id {space_id}; listing pages...")
    pages = list(api.pages(space_id))
    by_id = {p["id"]: p for p in pages}
    print(f"{len(pages)} pages")

    removed = [pid for pid in known if pid not in by_id]
    if known and len(removed) > MAX_REMOVED_SHARE * len(known) and not force:
        sys.exit(
            f"The listing has {len(pages)} pages, {len(removed)} of {len(known)} local pages would be "
            f"removed. Refusing: check the space and access, or run with --force to accept it."
        )

    def breadcrumbs(page: dict) -> list[str]:
        chain, parent = [], page.get("parentId")
        while parent and parent in by_id:
            chain.append(by_id[parent]["title"])
            parent = by_id[parent].get("parentId")
        return chain[::-1]

    written = skipped = 0
    for page in pages:
        pid, version = page["id"], page["version"]["number"]
        crumbs = breadcrumbs(page)
        old = known.get(pid, {})
        # A renamed or moved parent changes the breadcrumbs without a new page version.
        if not force and old.get("version") == version and old.get("breadcrumbs") == crumbs:
            skipped += 1
        else:
            storage = page.get("body", {}).get("storage", {}).get("value", "")
            meta = {
                "id": pid,
                "title": page["title"],
                "space": space,
                "url": f"{base_url.rstrip('/')}/wiki{page['_links']['webui']}",
                "version": version,
                "updated": page["version"]["createdAt"],
                "created": page.get("createdAt"),
                "parentId": page.get("parentId"),
                "breadcrumbs": crumbs,
                "labels": api.labels(pid),
            }
            md = frontmatter(meta) + f"# {page['title']}\n\n" + storage_to_markdown(storage)
            write_atomic(pages_dir / f"{pid}.storage.html", storage)
            write_atomic(pages_dir / f"{pid}.md", md)
            # the .json goes last: it is what marks the page as synced at this version
            write_atomic(pages_dir / f"{pid}.json", json.dumps(meta, ensure_ascii=False, indent=1))
            written += 1
            print(f"  [{written}] {page['title']}")
        if attachments:
            shown = referenced_images(page.get("body", {}).get("storage", {}).get("value", ""))
            sync_attachments(api, pid, out / "attachments" / pid, only=shown)

    for pid in removed:
        for suffix in (".storage.html", ".json", ".md"):
            (pages_dir / f"{pid}{suffix}").unlink(missing_ok=True)
        shutil.rmtree(out / "attachments" / pid, ignore_errors=True)

    manifest = {
        p["id"]: {"title": p["title"], "version": p["version"]["number"], "path": f"pages/{p['id']}.md"}
        for p in pages
    }
    write_atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"Done: {written} written, {skipped} unchanged, {len(removed)} removed -> {out}")


def reconvert(space: str, out_root: Path) -> int:
    """Rebuild the markdown of every synced page from its stored storage HTML, offline: after a
    converter fix there is no need to download the space again."""
    pages_dir = out_root / space / "pages"
    n = 0
    for meta_path in sorted(pages_dir.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        storage = (pages_dir / f"{meta['id']}.storage.html").read_text()
        md = frontmatter(meta) + f"# {meta['title']}\n\n" + storage_to_markdown(storage)
        write_atomic(pages_dir / f"{meta['id']}.md", md)
        n += 1
    return n


def sync_attachments(api: Confluence, page_id: str, att_dir: Path, only: set[str] | None = None) -> None:
    """Download new or changed attachments of a page (only those titled in `only`, when given); files are named
    `<attachment id>-<title>` so that titles that clash after sanitizing don't overwrite each other. `_index.json`
    maps the attachment id to the file, its version and its title (the API finds a picture by the title)."""
    index_path = att_dir / "_index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    current = {}
    for att in api.attachments(page_id):
        link = att.get("downloadLink")
        if not link or (only is not None and att["title"] not in only):
            continue
        aid, version = att["id"], att.get("version", {}).get("number")
        name = safe_name(f"{aid}-{att['title']}")
        current[aid] = {"name": name, "version": version, "title": att["title"]}
        if index.get(aid) == current[aid] and (att_dir / name).exists():
            continue
        att_dir.mkdir(parents=True, exist_ok=True)
        try:
            api.download(f"/wiki{link}", att_dir / name)
        except httpx.HTTPError as exc:
            print(f"    attachment {att['title']!r} skipped: {exc}")
            current.pop(aid)
    for aid, entry in index.items():
        if aid not in current and entry["name"] not in {e["name"] for e in current.values()}:
            (att_dir / entry["name"]).unlink(missing_ok=True)
    if current or index:
        att_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(index_path, json.dumps(current, ensure_ascii=False, indent=1))
