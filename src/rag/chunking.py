"""Structure-aware chunking of synced Confluence pages.

A page is split along its markdown headings into sections, then:
- a page that fits into ``max_tokens`` becomes a single chunk;
- small neighbouring sections (< ``min_tokens``) are merged;
- sections larger than ``max_tokens`` are split on paragraph boundaries;
- fenced code blocks and tables are never split;
- pages without any text (e.g. parent pages holding only a children macro) are skipped.

Each chunk carries a context header (breadcrumbs > page title > section) that is
prepended to the text used for embedding and BM25 (``embed_text``), while ``text``
keeps the original markdown for display.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from rag import config

# The optional closing hashes need a space before them: "## Using C#" keeps its "#".
HEADING = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
IMAGE_MARK = re.compile(r"\[image: [^\]]+\]")
# Title of the space's root page; it heads every page's breadcrumbs and is dropped on load.
ROOT_PAGE = "Home"
# Bump when the chunking logic changes: the indexer then rebuilds (as it does when the limits change).
CHUNKER_VERSION = 2


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars per token for English prose)."""
    return math.ceil(len(text) / 4)


@dataclass
class Page:
    id: str
    title: str
    version: int
    url: str
    updated: str
    breadcrumbs: list[str]
    labels: list[str]
    body: str
    created: str = ""  # when the page was created in Confluence (sync metadata); not indexed

    def __post_init__(self) -> None:
        self.breadcrumbs = [c for c in self.breadcrumbs if c != ROOT_PAGE]

    @property
    def product(self) -> str:
        """Top-level section of the space, e.g. 'Direct Print'."""
        return self.breadcrumbs[0] if self.breadcrumbs else self.title

    @property
    def context_path(self) -> list[str]:
        return self.breadcrumbs + [self.title]


@dataclass
class Section:
    path: list[str]  # heading titles from the outermost to this section
    heading: str  # original heading line ('' for text before the first heading)
    content: str

    @property
    def markdown(self) -> str:
        return f"{self.heading}\n\n{self.content}".strip() if self.heading else self.content.strip()


@dataclass
class Chunk:
    chunk_id: str
    page_id: str
    page_version: int
    title: str
    product: str
    breadcrumbs: list[str]
    section_path: list[str]
    url: str
    updated: str
    labels: list[str]
    text: str
    embed_text: str
    n_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_tokens = approx_tokens(self.text)


# --- loading -------------------------------------------------------------------


def strip_frontmatter(md: str, title: str) -> str:
    if md.startswith("---\n"):
        end = md.find("\n---\n", 4)
        md = md[end + 5 :] if end != -1 else md
    md = md.lstrip("\n")
    first, _, rest = md.partition("\n")
    if first.strip() == f"# {title}":
        md = rest
    return md.strip("\n")


def load_page(meta_path: Path) -> Page:
    meta = json.loads(meta_path.read_text())
    md_path = meta_path.with_suffix(".md")
    body = strip_frontmatter(md_path.read_text(), meta["title"])
    return Page(
        id=meta["id"],
        title=meta["title"],
        version=meta["version"],
        url=meta["url"],
        updated=meta["updated"],
        breadcrumbs=meta.get("breadcrumbs", []),
        labels=meta.get("labels", []),
        body=body,
        created=meta.get("created") or "",
    )


def load_pages(space_dir: Path) -> list[Page]:
    pages = []
    for path in sorted((space_dir / "pages").glob("*.json")):
        try:
            pages.append(load_page(path))
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"Broken synced page {path}: {exc}; re-run the sync") from exc
    return pages


def chunk_settings(max_tokens: int = config.CHUNK_MAX_TOKENS, min_tokens: int = config.CHUNK_MIN_TOKENS) -> dict:
    """What the chunks depend on besides the pages; stored with the index to detect changes."""
    return {"version": CHUNKER_VERSION, "max_tokens": max_tokens, "min_tokens": min_tokens}


# --- splitting -----------------------------------------------------------------


def clean_title(title: str) -> str:
    """Drop bold/italic wrappers from a heading ('**Setup**' -> 'Setup'), keep inline code."""
    return re.sub(r"\*\*|__", "", title).strip("*_ ")


def fence_mask(lines: list[str]) -> list[bool]:
    """Which lines are inside a fenced code block (fence lines included).

    A fence closes only on the same character, at least as long, with nothing after it. A fence that
    never closes is not a fence: one stray ``` must not turn the rest of the page into code.
    """
    mask = [False] * len(lines)
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if m:
            marker = m.group(1)
            close = next(
                (
                    j for j in range(i + 1, len(lines))
                    if (c := FENCE.match(lines[j]))
                    and c.group(1)[0] == marker[0] and len(c.group(1)) >= len(marker) and not c.group(2).strip()
                ),
                None,
            )  # fmt: skip
            if close is not None:
                mask[i : close + 1] = [True] * (close + 1 - i)
                i = close + 1
                continue
        i += 1
    return mask


def has_text(section: Section) -> bool:
    return any(l.strip() and not HEADING.match(l) for l in section.content.splitlines())


def split_sections(body: str) -> list[Section]:
    """Split markdown into sections at headings outside fenced code blocks."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    heading, lines = "", []

    def flush() -> None:
        content = "\n".join(lines).strip("\n")
        if heading or content.strip():
            sections.append(Section([t for _, t in stack], heading, content))

    all_lines = body.splitlines()
    for line, in_fence in zip(all_lines, fence_mask(all_lines)):
        match = None if in_fence else HEADING.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), clean_title(match.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            heading, lines = line, []
        else:
            lines.append(line)
    flush()
    return sections


def split_blocks(text: str) -> list[str]:
    """Split on blank lines, keeping fenced code blocks whole."""
    blocks, current = [], []
    lines = text.splitlines()
    for line, in_fence in zip(lines, fence_mask(lines)):
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def pack_blocks(blocks: list[str], max_tokens: int, min_tokens: int = 0) -> list[str]:
    """Greedily pack blocks into pieces up to max_tokens; an oversized block stays whole.

    A heading never ends a piece: it moves on with the text it introduces. A trailing piece smaller
    than min_tokens is glued to the previous one (slightly exceeding max_tokens beats a
    context-free fragment).
    """
    pieces, current = [], []
    for block in blocks:
        candidate = "\n\n".join(current + [block])
        only_headings = all(HEADING.match(b) for b in current)
        if current and not only_headings and approx_tokens(candidate) > max_tokens:
            carry = []
            while len(current) > 1 and HEADING.match(current[-1]):
                carry.insert(0, current.pop())
            pieces.append("\n\n".join(current))
            current = carry + [block]
        else:
            current.append(block)
    if current:
        pieces.append("\n\n".join(current))
    if len(pieces) > 1 and approx_tokens(pieces[-1]) < min_tokens:
        pieces[-2:] = ["\n\n".join(pieces[-2:])]
    return pieces


def common_prefix(paths: list[list[str]]) -> list[str]:
    prefix = paths[0]
    for path in paths[1:]:
        n = 0
        while n < min(len(prefix), len(path)) and prefix[n] == path[n]:
            n += 1
        prefix = prefix[:n]
    return prefix


def group_path(sections: list[Section]) -> list[str]:
    """Section path of merged sections: their common path, else the path of the largest one
    (merged siblings "## A" + "## B" have no common path, but the chunk is still mostly about one)."""
    common = common_prefix([s.path for s in sections])
    if common or len(sections) == 1:
        return common
    return max(sections, key=lambda s: approx_tokens(s.markdown)).path


def group_sections(
    sections: list[Section], max_tokens: int, min_tokens: int
) -> list[tuple[list[str], str]]:
    """Merge small neighbouring sections and split large ones.

    Returns (section_path, markdown) pairs.
    """
    # Heading-only sections and short intros of a parent section
    # (e.g. "# How to Fix It" + one sentence, followed by "## Option 1")
    # belong with what follows them, not with the previous section.
    attached: list[Section] = []
    pending: list[Section] = []
    for i, section in enumerate(sections):
        nxt = sections[i + 1] if i + 1 < len(sections) else None
        is_parent = nxt is not None and nxt.path[: len(section.path)] == section.path
        short = approx_tokens(section.markdown) < min_tokens
        if not has_text(section) or (short and is_parent):
            pending.append(section)
            continue
        if pending:
            heads = [s.markdown for s in pending]
            section = Section(
                group_path(pending + [section]),
                heads[0],
                "\n\n".join(heads[1:] + [section.markdown]),
            )
            pending = []
        attached.append(section)
    attached.extend(pending)

    groups: list[list[Section]] = []
    for section in attached:
        size = approx_tokens(section.markdown)
        if groups:
            prev = groups[-1]
            prev_size = approx_tokens("\n\n".join(s.markdown for s in prev))
            small = prev_size < min_tokens or size < min_tokens
            # a tiny section (a heading or a line, e.g. "## Related pages") is not worth a chunk
            tiny = size < min_tokens // 2
            if small and (prev_size + size <= max_tokens or tiny):
                prev.append(section)
                continue
        groups.append([section])

    result: list[tuple[list[str], str]] = []
    for group in groups:
        path = group_path(group)
        markdown = "\n\n".join(s.markdown for s in group)
        if approx_tokens(markdown) <= max_tokens:
            result.append((path, markdown))
            continue
        # A single oversized section: keep its heading on the first piece only,
        # the section path in the context header covers the rest.
        for piece in pack_blocks(split_blocks(markdown), max_tokens, min_tokens):
            result.append((path, piece))
    return result


def context_header(page: Page, section_path: list[str]) -> str:
    path = list(page.context_path)
    for title in section_path:
        if title != path[-1]:
            path.append(title)
    return " > ".join(path)


def chunk_page(
    page: Page,
    max_tokens: int = config.CHUNK_MAX_TOKENS,
    min_tokens: int = config.CHUNK_MIN_TOKENS,
) -> list[Chunk]:
    body = page.body.strip()
    content_only = "\n".join(l for l in body.splitlines() if not HEADING.match(l))
    if not IMAGE_MARK.sub("", content_only).strip():
        return []

    if approx_tokens(body) <= max_tokens:
        parts = [([], body)]
    else:
        parts = group_sections(split_sections(body), max_tokens, min_tokens)

    chunks = []
    for i, (section_path, text) in enumerate(parts):
        # image markers carry no text: dropped once here, for the index, the UI and the LLM
        text = re.sub(r"\n{3,}", "\n\n", IMAGE_MARK.sub("", text)).strip()
        header = context_header(page, section_path)
        embed_text = f"{header}\n\n{text}"
        chunks.append(
            Chunk(
                chunk_id=f"{page.id}:{i}",
                page_id=page.id,
                page_version=page.version,
                title=page.title,
                product=page.product,
                breadcrumbs=page.breadcrumbs,
                section_path=section_path,
                url=page.url,
                updated=page.updated,
                labels=page.labels,
                text=text,
                embed_text=embed_text,
            )
        )
    return chunks
