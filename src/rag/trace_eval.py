"""Eval of failed-job analysis over eval/traces.yaml: does the evidence contain the cause?

No LLM here: it measures what the answer is built from. A case is a traceback (or a question) with the function where
the failure really comes from, optionally the error codes it carries and the modules that define that function.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag.code_answer import build_code_context, gather
from rag.code_graph import CodeGraph
from rag.code_search import CodeSearcher
from rag.code_store import CodeStore


@dataclass(frozen=True)
class Cause:
    module: str
    function: str


@dataclass
class TraceCase:
    id: str
    group: str
    traceback: str
    causes: list[Cause]  # any of them counts
    question: str = ""
    codes: list[str] = field(default_factory=list)  # error codes the evidence must carry
    chain: list[str] = field(default_factory=list)  # modules that must show a definition of the cause function


@dataclass
class CaseResult:
    id: str
    group: str
    frames_total: int  # own frames in the traceback
    frames_found: int  # of them found in the index
    cause_rank: int | None  # position of the first piece about a cause function; None: not in the evidence
    cause_sent: bool  # that piece fits the token budget, so the LLM sees it
    codes_found: bool | None  # None: the case expects no code
    chain_ok: bool | None  # None: the case expects no chain


def load_cases(path: Path) -> list[TraceCase]:
    cases = []
    for item in yaml.safe_load(path.read_text()) or []:
        if not item.get("cause"):
            raise ValueError(f"trace case {item.get('id')!r} has no `cause`")
        cases.append(TraceCase(
            str(item["id"]), item.get("group", "other"), item.get("traceback", ""),
            [Cause(c["module"], c["function"]) for c in item["cause"]],
            item.get("question", ""), [str(c) for c in item.get("codes", [])], list(item.get("chain", [])),
        ))
    return cases


def evaluate_case(
    case: TraceCase, store: CodeStore, graph: CodeGraph, searcher: CodeSearcher | None, **gather_kwargs
) -> CaseResult:
    ev = gather(store, graph, case.question, case.traceback, searcher, **gather_kwargs)
    sources, _ = build_code_context(ev.pieces)
    sent = {(s.module, s.path, s.line_start, s.line_end) for s in sources}
    causes = {(c.module, c.function) for c in case.causes}
    rank, is_sent = None, False
    for i, p in enumerate(ev.pieces, 1):
        if (p.module, p.name) in causes:
            rank, is_sent = i, (p.module, p.path, p.line_start, p.line_end) in sent
            break
    own = [f for f in ev.frames]
    codes = None
    if case.codes:
        codes = all(any(p.role == "error code" and code in p.text for p in ev.pieces) for code in case.codes)
    chain = None
    if case.chain:
        names = {c.function for c in case.causes}
        have = {p.module for p in ev.pieces if p.name in names}
        chain = set(case.chain) <= have
    return CaseResult(
        case.id, case.group, len(own), sum(f.found for f in own), rank, is_sent, codes, chain,
    )


def summarize(results: list[CaseResult]) -> dict[str, dict]:
    """Per group and overall: the share of frames found, the cause at rank 1 / within 3 / sent to the LLM, codes, chain."""
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        groups[r.group].append(r)
        groups["all"].append(r)

    def share(values: list[bool]) -> float | None:
        return sum(values) / len(values) if values else None

    out = {}
    for name, rows in groups.items():
        frames_total = sum(r.frames_total for r in rows)
        out[name] = {
            "n": len(rows),
            "frames": sum(r.frames_found for r in rows) / frames_total if frames_total else None,
            "cause@1": share([r.cause_rank == 1 for r in rows]),
            "cause@3": share([r.cause_rank is not None and r.cause_rank <= 3 for r in rows]),
            "cause_sent": share([r.cause_sent for r in rows]),
            "codes": share([r.codes_found for r in rows if r.codes_found is not None]),
            "chain": share([r.chain_ok for r in rows if r.chain_ok is not None]),
        }
    return out
