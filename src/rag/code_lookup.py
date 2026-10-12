"""Exact lookups for a parsed traceback: frame -> code unit, unit -> overrides, error code -> registry and raise sites.

No embeddings here: a frame names a file and a line, an error code names a string, so the answer is a filter on
`code_units` and a walk of the code graph. Semantic search only fills in what these cannot find.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass, field, replace

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


def scope_modules(graph: CodeGraph, modules: Collection[str]) -> set[str]:
    """`modules` and everything they depend on among the known modules: the code that explains a failure in them."""
    scope = set(modules)
    for name in modules:
        scope |= graph.depends_closure(name)
    return scope


def sync_problems(graph: CodeGraph, store: CodeStore, only: Collection[str] | None = None) -> dict[str, str]:
    """Module -> what differs between the graph and the search index (empty when they agree).

    `rag module add` writes the index and then the graph, `--no-index` writes only the graph: an interrupted add or
    that flag leaves rows from another version, so the overrides and line numbers do not match the graph.
    """
    rows, indexed = store.modules(), store.indexed_modules()
    problems = {}
    for name in graph.modules if only is None else [n for n in only if n in graph.modules]:
        info, held = graph.modules[name], indexed.get(name)
        if name not in rows:
            problems[name] = "not in the search index (added with --no-index?)"
        elif held is None:
            problems[name] = "no record of the version the search index holds: add the module again"
        elif (held["version"], held["ref"]) != (info.version, info.ref):
            problems[name] = (
                f"the search index holds {held['version']} ({held['ref'] or 'working tree'}), "
                f"the graph {info.version} ({info.ref or 'working tree'}): add the module again"
            )
    if only is None:
        problems |= {name: "in the search index but not in the graph" for name in rows if name not in graph.modules}
    return problems


def whole_unit(rows: list[dict]) -> dict:
    """One dict for a unit stored in parts: the parts' text joined, the range widened to the whole unit.

    A part of a unit whose text is not split (a view) holds the whole text already: it is not joined with its copies."""
    rows = sorted(rows, key=lambda r: r["part"])
    joined = rows[0]["parts"] > 1 and not json.loads(rows[0].get("extra") or "{}").get("whole_text")
    return rows[0] | {
        "text": "\n".join(r["text"] for r in rows) if joined else rows[0]["text"],
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


def resolve_frame(store: CodeStore, frame: Frame) -> FrameUnit:
    """The unit of one frame of our own code: the one at its line, else the same-named function nearest to it."""
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
        return frame_unit(frame, at_line)
    named = unit_rows_by_name(store, frame) if frame.func else []
    if named:
        nearest = min(named, key=lambda r: min(abs(r["line_start"] - frame.line), abs(r["line_end"] - frame.line)))
        return frame_unit(frame, nearest, drifted=True)
    if at_line:
        return frame_unit(frame, at_line)
    return FrameUnit(frame)


def resolve_frames(store: CodeStore, tb: ParsedTraceback) -> list[FrameUnit]:
    """The units of the traceback's own frames, innermost first; frames that match no unit have `found=False`.

    The line decides, unless the unit at the line is not the function the frame names: then the same-named function of
    that file nearest to the line wins (`drifted`): the client runs another version, whose lines moved. A name that
    exists nowhere in the file (a nested function such as a decorator's `wrapper`) keeps the unit at the line.

    A frame whose path reads as several modules (`Frame.alternatives`) is looked up under its first reading, then
    the others; the one that is found becomes the frame's module and path (the frame is changed in place, so the
    traceback, the evidence and the notes agree).
    """
    out = []
    for frame in tb.own_frames:
        fu = resolve_frame(store, frame)
        for module, rel in [] if fu.found else frame.alternatives:
            alt = resolve_frame(store, replace(frame, module=module, rel=rel))
            if alt.found:
                frame.module, frame.rel, frame.alternatives = module, rel, []
                alt.frame = frame
                fu = alt
                break
        out.append(fu)
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
    allowed = visible | scope_modules(graph, [hit.module])
    chain = [m for m in graph.chain(hit.model, hit.name, only=allowed) if not is_self(m)] if hit.model else []
    abstract = []
    if hit.extra.get("adapter"):
        abstract = graph.adapter_group(hit.name).abstract
    return Related(chain, abstract)


def find_error_code(
    store: CodeStore, code: str, modules: Collection[str] | None = None, limit: int = 8
) -> list[CodeUnit]:
    """Units that mention the code as a whole word: the registry entry (a class header) first, then raise sites.

    With `modules` (those of the traceback and what they depend on) only their units are returned, because connectors
    may use the same code for different errors; when none of them mentions the code, any module's units are."""
    rows = store.rows_where(f"text LIKE {sql_str('%' + code + '%')}", COLUMNS, limit=10000)
    rows = [r for r in rows if code in ERROR_CODE.findall(r["text"])]
    if modules and any(r["module"] in modules for r in rows):
        rows = [r for r in rows if r["module"] in modules]
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
