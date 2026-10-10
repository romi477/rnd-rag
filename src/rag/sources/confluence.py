"""Sync a Confluence Cloud space into raw-data/ for the RAG pipeline.

Layout (per space):
    raw-data/confluence/<space>/
        pages/<id>.storage.html   original storage-format body (lossless)
        pages/<id>.md             markdown with YAML frontmatter (for chunking)
        pages/<id>.json           page metadata (version, parent, breadcrumbs, url)
        attachments/<id>/<file>   page attachments (only with --attachments)
        _manifest.json            id -> {title, version, path}

Incremental: a page is rewritten only when its version number changed.
Pages that disappeared from the space are removed locally.

Usage (token falls back to $JIRA_TOKEN_VENT, defined in interactive zsh):
    zsh -ic 'uv run rag sync'                # sync space from .env
    zsh -ic 'uv run rag sync --attachments'  # also download attachments
    zsh -ic 'uv run rag sync --force'        # rewrite everything
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import httpx
from markdownify import markdownify

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
        resp = self.http.get(url, params=params or None)
        if resp.status_code == 401:
            sys.exit("401 Unauthorized: check CONFLUENCE_EMAIL / CONFLUENCE_API_TOKEN")
        resp.raise_for_status()
        return resp

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


# --- storage format -> markdown -------------------------------------------------

CODE_MACRO = re.compile(
    r'<ac:structured-macro[^>]*ac:name="code"[^>]*>(.*?)</ac:structured-macro>', re.S
)
CODE_LANG = re.compile(r'<ac:parameter ac:name="language">([^<]*)</ac:parameter>')
CODE_BODY = re.compile(r"<ac:plain-text-body><!\[CDATA\[(.*?)\]\]></ac:plain-text-body>", re.S)
IMAGE = re.compile(r"<ac:image[^>]*>.*?</ac:image>", re.S)
IMAGE_FILE = re.compile(r'ri:filename="([^"]+)"')
IMAGE_URL = re.compile(r'ri:value="([^"]+)"')
PANEL_MACRO = re.compile(
    r'<ac:structured-macro[^>]*ac:name="(info|note|warning|tip|panel|expand)"[^>]*>', re.S
)
CDATA = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.S)
MACRO_PARAM = re.compile(r"<ac:parameter[^>]*>.*?</ac:parameter>", re.S)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _code(match: re.Match) -> str:
    inner = match.group(1)
    lang = CODE_LANG.search(inner)
    body = CODE_BODY.search(inner)
    cls = f' class="language-{lang.group(1)}"' if lang else ""
    return f"<pre><code{cls}>{_escape(body.group(1)) if body else ''}</code></pre>"


def _image(match: re.Match) -> str:
    tag = match.group(0)
    src = IMAGE_FILE.search(tag) or IMAGE_URL.search(tag)
    return f"<p>[image: {src.group(1)}]</p>" if src else ""


def storage_to_markdown(html: str) -> str:
    html = CODE_MACRO.sub(_code, html)
    html = IMAGE.sub(_image, html)
    html = MACRO_PARAM.sub("", html)
    html = PANEL_MACRO.sub(lambda m: f"<p><strong>{m.group(1).upper()}:</strong></p>", html)
    html = CDATA.sub(lambda m: _escape(m.group(1)), html)
    md = markdownify(html, heading_style="ATX", bullets="-", code_language_callback=_lang)
    return re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"


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


def sync(space: str, out_root: Path, attachments: bool = False, force: bool = False) -> None:
    base_url = env("CONFLUENCE_BASE_URL", "https://ventortech.atlassian.net")
    token = os.environ.get("CONFLUENCE_API_TOKEN") or env("JIRA_TOKEN_VENT")
    api = Confluence(base_url, env("CONFLUENCE_EMAIL"), token)

    out = out_root / space
    pages_dir = out / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    space_id = api.space_id(space)
    print(f"Space {space} -> id {space_id}; listing pages...")
    pages = list(api.pages(space_id))
    by_id = {p["id"]: p for p in pages}
    print(f"{len(pages)} pages")

    def breadcrumbs(page: dict) -> list[str]:
        chain, parent = [], page.get("parentId")
        while parent and parent in by_id:
            chain.append(by_id[parent]["title"])
            parent = by_id[parent].get("parentId")
        return chain[::-1]

    written = skipped = 0
    for page in pages:
        pid, version = page["id"], page["version"]["number"]
        if not force and manifest.get(pid, {}).get("version") == version:
            skipped += 1
            continue

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
            "breadcrumbs": breadcrumbs(page),
            "labels": api.labels(pid),
        }
        (pages_dir / f"{pid}.storage.html").write_text(storage)
        (pages_dir / f"{pid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        md = frontmatter(meta) + f"# {page['title']}\n\n" + storage_to_markdown(storage)
        (pages_dir / f"{pid}.md").write_text(md)

        if attachments:
            att_dir = out / "attachments" / pid
            for att in api.attachments(pid):
                link = att.get("downloadLink")
                if not link:
                    continue
                att_dir.mkdir(parents=True, exist_ok=True)
                data = api.get(f"/wiki{link}").content
                (att_dir / safe_name(att["title"])).write_bytes(data)

        manifest[pid] = {"title": page["title"], "version": version, "path": f"pages/{pid}.md"}
        written += 1
        print(f"  [{written}] {page['title']}")

    removed = [pid for pid in manifest if pid not in by_id]
    for pid in removed:
        for suffix in (".storage.html", ".json", ".md"):
            (pages_dir / f"{pid}{suffix}").unlink(missing_ok=True)
        del manifest[pid]

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"Done: {written} written, {skipped} unchanged, {len(removed)} removed -> {out}")

