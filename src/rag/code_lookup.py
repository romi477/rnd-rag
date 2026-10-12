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
    drifted: bool = False  # found by the function's name: the line number does not match this version of the code

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


def frame_unit(frame: Frame, row: dict, drifted: bool = False) -> FrameUnit:
    return FrameUnit(
        frame, True, row["unit_id"], row["kind"], row["name"], row["model"], row["module_version"],
        row["line_start"], row["line_end"], row["text"], json.loads(row["extra"] or "{}"), drifted,
    )


def unit_rows_by_name(store: CodeStore, frame: Frame) -> list[dict]:
    """Rows of the methods called `frame.func` in the frame's file, one dict per unit (parts joined)."""
    rows = store.rows_where(
        f"module = {sql_str(frame.module)} AND path = {sql_str(frame.rel)} AND kind = 'method' "
        f"AND name = {sql_str(frame.func)}", COLUMNS,
    )
    by_unit: dict[str, list[dict]] = {}
    for r in rows:
        by_unit.setdefault(r["unit_id"], []).append(r)
    return [whole_unit(parts) for parts in by_unit.values()]


def resolve_frames(store: CodeStore, tb: ParsedTraceback) -> list[FrameUnit]:
    """The units of the traceback's own frames, innermost first; frames that match no unit have `found=False`.

    The line decides, unless the unit at the line is not the function the frame names: then the same-named function of
    that file nearest to the line wins (`drifted`): the client runs another version, whose lines moved. A name that
    exists nowhere in the file (a nested function such as a decorator's `wrapper`) keeps the unit at the line.
    """
    out = []
    for frame in tb.own_frames:
        where = (
            f"module = {sql_str(frame.module)} AND path = {sql_str(frame.rel)} AND kind IN "
            f"({', '.join(sql_str(k) for k in CODE_KINDS)}) AND line_start <= {frame.line} AND line_end >= {frame.line}"
        )
        hits = store.rows_where(where, COLUMNS)
        at_line = None
        if hits:
            part = min(hits, key=lambda r: r["line_end"] - r["line_start"])
            at_line = whole_unit(store.rows_where(f"unit_id = {sql_str(part['unit_id'])}", COLUMNS))
        if at_line and at_line["name"] == frame.func:
            out.append(frame_unit(frame, at_line))
            continue
        named = unit_rows_by_name(store, frame) if frame.func else []
        if named:
            nearest = min(named, key=lambda r: min(abs(r["line_start"] - frame.line), abs(r["line_end"] - frame.line)))
            out.append(frame_unit(frame, nearest, drifted=True))
        elif at_line:
            out.append(frame_unit(frame, at_line))
        else:
            out.append(FrameUnit(frame))
    return out


def related_methods(graph: CodeGraph, hit: FrameUnit, visible: set[str]) -> Related:
    """Other definitions of the frame's method in `visible` modules and in the modules the frame's module depends on,
    and for an adapter method its abstract declaration.

    Another connector's implementation is not related: the traceback shows which connector failed.
    """
    if not hit.found or hit.kind != "method":
        return Related()

    def is_self(m: MethodRef) -> bool:
        return (m.module, m.path, m.line_start) == (hit.module, hit.path, hit.line_start)

    # What the frame's module extends is always related: its base implementation explains the override.
    allowed = visible | graph.depends_closure(hit.module) | {hit.module}
    chain = [m for m in graph.chain(hit.model, hit.name, only=allowed) if not is_self(m)] if hit.model else []
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
