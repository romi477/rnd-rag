"""Exact lookups for a parsed traceback: frame -> code unit, unit -> overrides, error code -> registry and raise sites.

No embeddings here: a frame names a file and a line, an error code names a string, so the answer is a filter on
`code_units` and a walk of the code graph. Semantic search only fills in what these cannot find.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from rag.code_graph import CodeGraph, MethodRef
from rag.code_store import CodeStore
from rag.sources.odoo_module import CodeUnit
from rag.store import sql_str
from rag.traceback import ERROR_CODE, Frame, ParsedTraceback

COLUMNS = [
    "row_id", "unit_id", "kind", "module", "module_version", "origin", "model", "name", "path", "line_start",
    "line_end", "text", "extra", "part", "parts",
]  # fmt: skip
CODE_KINDS = ("method", "class")  # units that hold executable or class-level lines of a .py file


@dataclass
class FrameUnit:
    """A frame of our own code and the unit that contains its line (`found` is False when none does)."""

    frame: Frame
    found: bool = False
    unit_id: str = ""
    kind: str = ""
    name: str = ""
    model: str = ""
    module_version: str = ""
    line_start: int = 0
    line_end: int = 0
    text: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def module(self) -> str:
        return self.frame.module

    @property
    def path(self) -> str:
        return self.frame.rel

    @property
    def name_matches(self) -> bool:
        """False for a nested function (a decorator's `wrapper`): the unit is the function around it."""
        return self.found and self.name == self.frame.func


@dataclass
class Related:
    chain: list[MethodRef] = field(default_factory=list)  # other definitions of the method, base first
    abstract: list[MethodRef] = field(default_factory=list)  # the adapter method's abstract declaration


def whole_unit(rows: list[dict]) -> dict:
    """One dict for a unit stored in parts: the parts' text joined, the range widened to the whole unit."""
    rows = sorted(rows, key=lambda r: r["part"])
    return rows[0] | {
        "text": "\n".join(r["text"] for r in rows) if rows[0]["parts"] > 1 else rows[0]["text"],
        "line_start": min(r["line_start"] for r in rows),
        "line_end": max(r["line_end"] for r in rows),
    }


def resolve_frames(store: CodeStore, tb: ParsedTraceback) -> list[FrameUnit]:
    """The units of the traceback's own frames, innermost first; frames that match no unit have `found=False`."""
    out = []
    for frame in tb.own_frames:
        where = (
            f"module = {sql_str(frame.module)} AND path = {sql_str(frame.rel)} AND kind IN "
            f"({', '.join(sql_str(k) for k in CODE_KINDS)}) AND line_start <= {frame.line} AND line_end >= {frame.line}"
        )
        hits = store.rows_where(where, COLUMNS)
        if not hits:
            out.append(FrameUnit(frame))
            continue
        part = min(hits, key=lambda r: r["line_end"] - r["line_start"])
        unit = whole_unit(store.rows_where(f"unit_id = {sql_str(part['unit_id'])}", COLUMNS))
        out.append(FrameUnit(
            frame, True, unit["unit_id"], unit["kind"], unit["name"], unit["model"], unit["module_version"],
            unit["line_start"], unit["line_end"], unit["text"], json.loads(unit["extra"] or "{}"),
        ))
    return out


def related_methods(graph: CodeGraph, hit: FrameUnit, visible: set[str]) -> Related:
    """Overrides of the frame's method in `visible` modules, and for an adapter method its abstract declaration.

    Another connector's implementation is not related: the traceback shows which connector failed.
    """
    if not hit.found or hit.kind != "method":
        return Related()

    def is_self(m: MethodRef) -> bool:
        return (m.module, m.path, m.line_start) == (hit.module, hit.path, hit.line_start)

    chain = [m for m in graph.chain(hit.model, hit.name, only=visible) if not is_self(m)] if hit.model else []
    abstract = []
    if hit.extra.get("adapter"):
        abstract = graph.adapter_group(hit.name).abstract
    return Related(chain, abstract)


def find_error_code(store: CodeStore, code: str, limit: int = 8) -> list[CodeUnit]:
    """Units that mention the code as a whole word: the registry entry (a class header) first, then raise sites."""
    rows = store.rows_where(f"text LIKE {sql_str('%' + code + '%')}", COLUMNS)
    rows = [r for r in rows if code in ERROR_CODE.findall(r["text"])]
    by_unit: dict[str, list[dict]] = {}
    for r in rows:
        by_unit.setdefault(r["unit_id"], []).append(r)
    units = [whole_unit(parts) for parts in by_unit.values()]
    units.sort(key=lambda u: (u["kind"] != "class", u["module"], u["path"], u["line_start"]))
    return [
        CodeUnit(
            u["kind"], u["module"], u["module_version"], "", u["model"], u["name"], u["path"], u["line_start"],
            u["line_end"], u["text"], "", json.loads(u["extra"] or "{}"),
        )
        for u in units[:limit]
    ]
