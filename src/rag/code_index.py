"""Odoo addon units -> parts that fit the embedder -> vectors -> `code_units`."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable

from rag import config
from rag.chunking import approx_tokens
from rag.code_store import CodeStore, identifier_words
from rag.sources.odoo_module import CodeUnit

# Bump when splitting or the row layout changes: modules added before must then be added again.
CODE_INDEX_VERSION = 3
PART_SUFFIX = " (part 99/99)"


class CodeIndexError(RuntimeError):
    pass


def split_unit(unit: CodeUnit, max_tokens: int) -> list[CodeUnit]:
    """Parts of at most `max_tokens` (approximate): cut at a blank line when one is in the second half of the part.

    The first line of `embed_text` (module > model > name) is repeated in every part. Only a method's or class header's text
    is its source lines, so only their parts get their own line ranges; other kinds keep the unit's whole range.
    """
    if approx_tokens(unit.embed_text) <= max_tokens:
        return [unit]
    header, _, body = unit.embed_text.partition("\n")
    budget = max(80, max_tokens * 4 - len(header) - len(PART_SUFFIX) - 1)
    items: list[tuple[int, str]] = []  # (index of the source line, text); a huge line becomes several items
    for i, line in enumerate(body.split("\n")):
        items += [(i, line[j : j + budget]) for j in range(0, len(line), budget)] or [(i, "")]

    groups: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    size = 0
    for item in items:
        while cur and size + len(item[1]) + 1 > budget:
            blanks = [i for i, (_, t) in enumerate(cur) if not t.strip() and sum(len(x) + 1 for _, x in cur[:i]) >= budget / 2]
            cut = blanks[-1] if blanks else len(cur)
            groups.append(cur[:cut])
            cur = cur[cut:]
            size = sum(len(t) + 1 for _, t in cur)
        cur.append(item)
        size += len(item[1]) + 1
    if cur:
        groups.append(cur)

    aligned = unit.kind in ("method", "class") and unit.text == body
    parts = []
    for n, group in enumerate(groups, 1):
        text = "\n".join(t for _, t in group)
        start, end = (unit.line_start + group[0][0], unit.line_start + group[-1][0]) if aligned else (unit.line_start, unit.line_end)
        parts.append(dataclasses.replace(
            unit, line_start=start, line_end=end,
            text=text if unit.text == body else unit.text,
            embed_text=f"{header} (part {n}/{len(groups)})\n{text}",
            extra=unit.extra | {"unit_id": unit.unit_id, "part": n, "parts": len(groups)},
        ))
    return parts


def unit_row(unit: CodeUnit) -> dict:
    return {
        "row_id": unit.id, "unit_id": unit.unit_id, "kind": unit.kind, "module": unit.module,
        "module_version": unit.module_version, "origin": unit.origin, "model": unit.model, "name": unit.name,
        "path": unit.path, "line_start": unit.line_start, "line_end": unit.line_end, "text": unit.text,
        "embed_text": unit.embed_text,
        "fts_text": unit.embed_text + ("\n" + words if (words := identifier_words(unit.embed_text)) else ""),
        "extra": json.dumps(unit.extra),
        "part": unit.extra.get("part", 1), "parts": unit.extra.get("parts", 1),
    }


def index_module(
    store: CodeStore,
    embedder,
    module: str,
    units: list[CodeUnit],
    max_tokens: int = config.CODE_MAX_TOKENS,
    log: Callable[[str], None] = print,
    reset: bool = False,
) -> int:
    """Replace the rows of `module` with its units. Returns the number of rows written.

    `reset` first drops every module from the index (after a version or settings change).
    """
    settings = {"code_index_version": CODE_INDEX_VERSION, "code_max_tokens": max_tokens}
    if reset or not store.exists():
        store.reset(embedder.model_name, embedder.dim)
    meta = store.meta
    others = [m for m in store.modules() if m != module]
    if others and any(meta.get(k, v) != v for k, v in settings.items()):
        raise CodeIndexError(
            f"The code index was built with other settings ({ {k: meta.get(k) for k in settings} } vs {settings}); "
            f"re-add all modules: {', '.join(others)}"
        )
    parts = [p for u in units for p in split_unit(u, max_tokens)]
    log(f"Embedding {len(parts)} parts from {len(units)} units of {module}...")
    vectors = embedder.embed_documents([p.embed_text for p in parts]) if parts else []
    store.replace_module(module, [unit_row(p) for p in parts], vectors)
    store.rebuild_fts()
    store.write_meta(**settings)
    return len(parts)
