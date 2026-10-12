"""`rag bench-traces`: the same failures, evidence and prompt to several LLMs; speed numbers and a report with the answers.

The speed numbers and the cheap checks (does the answer name the cause function, does it cite [n] the way the UI
expects) say little about quality: the report puts the answers side by side to be read against the sources.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from rag import config
from rag.bench import Run, as_markdown_block, describe, fmt, generate
from rag.answer import answer_max_tokens
from rag.code_answer import CodeSource, build_code_context, build_code_messages, gather
from rag.code_graph import CodeGraph
from rag.code_search import CodeSearcher
from rag.code_store import CodeStore
from rag.llm import LLM, LLMError
from rag.trace_eval import TraceCase

CITATION = re.compile(r"\[(\d+)\]")
LOOSE_CITATION = re.compile(r"\[\s*sources?\b[^\]]*\]", re.I)  # "[Source 3]", "[Source: Exception]": not the [n] form


@dataclass
class Item:
    case: TraceCase
    sources: list[CodeSource]
    runs: dict[str, Run] = field(default_factory=dict)


def answer_checks(text: str, case: TraceCase, n_sources: int) -> dict:
    numbers = [int(n) for n in CITATION.findall(text)]
    return {
        "cause_named": any(c.function in text for c in case.causes),
        "citations": sum(1 <= n <= n_sources for n in numbers),
        "loose_citations": len(LOOSE_CITATION.findall(text)),
        "unknown_citations": sum(not 1 <= n <= n_sources for n in numbers),
    }


def run_trace_bench(
    cases: list[TraceCase],
    models: list[str],
    store: CodeStore,
    graph: CodeGraph,
    searcher: CodeSearcher | None,
    llm: LLM,
    log: Callable[[str], None] = lambda _: None,
    on_item: Callable[[list[Item]], None] = lambda _: None,
    lang: str = "en",
) -> list[Item]:
    """Every case on every model. A model that fails to warm up stops the bench."""
    for model in models:
        log(f"warming up {model}")
        warm = generate(llm, model, [{"role": "user", "content": "hi"}], max_tokens=1)
        if warm.error and warm.error != "empty answer":
            raise LLMError(f"{model}: {warm.error}")
    items: list[Item] = []
    for i, case in enumerate(cases):
        ev = gather(store, graph, case.question, case.traceback, searcher)
        sources, context = build_code_context(ev.pieces)
        item = Item(case, sources)
        items.append(item)
        if not sources:
            log(f"{i + 1:>2} {case.id}: no sources, skipped")
        else:
            messages = build_code_messages(case.question, ev, context, lang)
            for model in models if i % 2 == 0 else models[::-1]:  # no model always runs right after another one
                run = item.runs[model] = generate(llm, model, messages, max_tokens=answer_max_tokens(config.CODE_LLM_MAX_TOKENS, lang))
                log(f"{i + 1:>2} {case.id} {model}: {describe(run)}")
        on_item(items)
    return items


def summary_rows(items: list[Item], models: list[str]) -> list[list[str]]:
    rows = [["model", "cases", "first token s", "total s", "tok/s", "cause named", "[n] cites/answer", "loose cites"]]
    for model in models:
        runs = [(it, it.runs[model]) for it in items if model in it.runs]
        ok = [(it, r) for it, r in runs if not r.error]
        checks = [answer_checks(r.text, it.case, len(it.sources)) for it, r in ok]
        mean = lambda values: sum(values) / len(values) if values else None  # noqa: E731
        rows.append([
            model, f"{len(ok)}/{len(runs)}",
            fmt(mean([r.ttft_s for _, r in ok if r.ttft_s is not None])), fmt(mean([r.total_s for _, r in ok])),
            fmt(mean([r.tok_per_s for _, r in ok if r.tok_per_s]), ".0f"),
            f"{sum(c['cause_named'] for c in checks)}/{len(checks)}",
            fmt(mean([c["citations"] for c in checks])), str(sum(c["loose_citations"] for c in checks)),
        ])
    return rows


def render_report(items: list[Item], models: list[str], settings: str, partial: bool = False) -> str:
    rows = summary_rows(items, models)
    lines = [
        f"# Trace bench {time.strftime('%Y-%m-%d %H:%M')}" + (" (interrupted)" if partial else ""), "", settings, "",
        "## Summary", "", "| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0]),
        *("| " + " | ".join(r) + " |" for r in rows[1:]), "",
        "cause named: the answer mentions the function the failure comes from; [n] cites: valid `[n]` citations per answer; "
        "loose cites: `[Source ...]` the UI does not link.", "", "## Answers",
    ]
    for i, item in enumerate(items, 1):
        case = item.case
        causes = ", ".join(f"{c.module}.{c.function}" for c in case.causes)
        lines += ["", f"### {i}. {case.id}", "", f"group: {case.group} · cause: {causes}", ""]
        if not item.sources:
            lines.append("No sources: the LLM was not called.")
            continue
        lines.append("Sources: " + " · ".join(f"[{s.n}] {s.module}/{s.path}:{s.line_start}-{s.line_end} ({s.role})" for s in item.sources))
        for model in models:
            run = item.runs[model]
            checks = answer_checks(run.text, case, len(item.sources)) if not run.error else {}
            tag = f" · cause named: {checks['cause_named']}, cites {checks['citations']}, loose {checks['loose_citations']}" if checks else ""
            lines += ["", f"#### {model}: {describe(run)}{tag}", "", as_markdown_block(run.text) or "(empty)"]
    return "\n".join(lines) + "\n"


def default_report_path():
    return config.DATA_DIR / "bench" / f"traces-{time.strftime('%Y%m%d-%H%M%S')}.md"
