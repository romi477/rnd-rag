"""Compare LLMs on answer generation: latency, speed, and the answers side by side.

The prompt is built exactly as for the UI (`answer.build_context` / `build_messages`), so the
numbers are what a user waits for. Quality is not scored automatically: the report puts the
answers next to each other, to be checked against the sources.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag import config
from rag.answer import Source, build_context, build_messages
from rag.llm import LLMClient, LLMError
from rag.search import Mode, Searcher


@dataclass
class Run:
    model: str
    text: str = ""
    ttft_s: float | None = None  # time to the first answer token
    total_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None

    @property
    def tok_per_s(self) -> float | None:
        """Generation speed, excluding the time to the first token (prompt processing)."""
        gen_s = self.total_s - (self.ttft_s or 0)
        return self.completion_tokens / gen_s if self.completion_tokens and gen_s > 0 else None


@dataclass
class Question:
    q: str
    group: str = ""
    expect: list[str] = field(default_factory=list)  # page ids that answer it; [] = not covered


@dataclass
class Item:
    question: Question
    sources: list[Source]
    runs: dict[str, Run] = field(default_factory=dict)  # model -> run; empty if no sources

    @property
    def expected_found(self) -> bool | None:
        """Whether a page that answers the question made it into the sources (None: no expectation)."""
        if not self.question.expect:
            return None
        return any(s.page_id in self.question.expect for s in self.sources)


def load_questions(path: Path, group: str | None = None) -> list[Question]:
    """A YAML list: plain strings or `- q: ...` items with optional `group` and `expect` (page ids).

    eval/queries.yaml works too (its `relevant` ids are used as `expect`).
    """
    questions = []
    for i in yaml.safe_load(path.read_text()) or []:
        if not isinstance(i, dict):
            i = {"q": str(i)}
        expect = [str(e) for e in i.get("expect", i.get("relevant", []))]
        questions.append(Question(i["q"], str(i.get("group", "")), expect))
    return [q for q in questions if group is None or q.group == group]


def generate(llm: LLMClient, model: str, messages: list[dict], **kwargs) -> Run:
    run, usage, parts = Run(model), {}, []
    started = time.perf_counter()
    try:
        for piece in llm.stream_chat(messages, model=model, usage=usage, **kwargs):
            if run.ttft_s is None:
                run.ttft_s = time.perf_counter() - started
            parts.append(piece)
    except LLMError as exc:
        run.error = str(exc)
    run.total_s = time.perf_counter() - started
    run.text = "".join(parts)
    run.prompt_tokens = usage.get("prompt_tokens")
    run.completion_tokens = usage.get("completion_tokens")
    return run


def run_bench(
    searcher: Searcher,
    llm: LLMClient,
    questions: list[Question],
    models: list[str],
    mode: Mode = "hybrid",
    rerank: bool = True,
    log: Callable[[str], None] = lambda _: None,
) -> list[Item]:
    for model in models:  # load / warm up, not measured
        log(f"warming up {model}")
        generate(llm, model, [{"role": "user", "content": "hi"}], max_tokens=1)
    items = []
    for i, question in enumerate(questions):
        hits = searcher.search(question.q, k=10, mode=mode, rerank=rerank)  # same as the UI
        sources, context = build_context(hits)
        item = Item(question, sources)
        items.append(item)
        if item.expected_found is False:
            log(f"{i + 1:>2} expected page is not among the sources")
        if not sources:
            log(f"{i + 1:>2} no sources, skipped: {question.q}")
            continue
        messages = build_messages(question.q, context)
        # alternate the order so that no model always runs right after another one
        for model in models if i % 2 == 0 else models[::-1]:
            run = item.runs[model] = generate(llm, model, messages)
            log(f"{i + 1:>2} {model}: {describe(run)}")
    return items


def describe(run: Run) -> str:
    if run.error:
        return f"error: {run.error}"
    speed = f"{run.tok_per_s:.0f} tok/s" if run.tok_per_s else "? tok/s"
    return (f"first token {run.ttft_s or 0:.1f} s, total {run.total_s:.1f} s, "
            f"{run.completion_tokens or '?'} tok, {speed}")


def summarize(items: list[Item], model: str) -> dict:
    runs = [it.runs[model] for it in items if model in it.runs]
    ok = [r for r in runs if not r.error]

    def stat(values: list, fn) -> float | None:
        values = [v for v in values if v is not None]
        return fn(values) if values else None

    return {
        "model": model,
        "answers": len(ok),
        "errors": len(runs) - len(ok),
        "ttft_avg": stat([r.ttft_s for r in ok], statistics.mean),
        "ttft_max": stat([r.ttft_s for r in ok], max),
        "total_avg": stat([r.total_s for r in ok], statistics.mean),
        "total_max": stat([r.total_s for r in ok], max),
        "total_sum": sum(r.total_s for r in ok),
        "tok_per_s": stat([r.tok_per_s for r in ok], statistics.mean),
        "out_tokens_avg": stat([r.completion_tokens for r in ok], statistics.mean),
    }


def fmt(value: float | None, spec: str = ".1f") -> str:
    return "–" if value is None else format(value, spec)


def summary_rows(items: list[Item], models: list[str]) -> list[list[str]]:
    rows = [["Model", "First token avg / max, s", "Answer avg / max, s", "Tokens/s",
             "Answer tokens", "Total, s", "Errors"]]
    for model in models:
        s = summarize(items, model)
        rows.append([
            model,
            f"{fmt(s['ttft_avg'])} / {fmt(s['ttft_max'])}",
            f"{fmt(s['total_avg'])} / {fmt(s['total_max'])}",
            fmt(s["tok_per_s"], ".0f"),
            fmt(s["out_tokens_avg"], ".0f"),
            fmt(s["total_sum"], ".0f"),
            str(s["errors"]),
        ])
    return rows


def retrieval_line(items: list[Item]) -> str:
    checked = [it.expected_found for it in items if it.expected_found is not None]
    uncovered = sum(1 for it in items if not it.question.expect)
    line = f"Expected page among the sources: {sum(checked)}/{len(checked)}"
    return line + (f"; {uncovered} question(s) not covered by the docs" if uncovered else "")


def render_report(items: list[Item], models: list[str], settings: str) -> str:
    rows = summary_rows(items, models)
    lines = [
        f"# LLM bench {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        settings,
        "",
        "## Summary",
        "",
        "| " + " | ".join(rows[0]) + " |",
        "|" + "---|" * len(rows[0]),
        *("| " + " | ".join(r) + " |" for r in rows[1:]),
        "",
        retrieval_line(items),
        "",
        "## Answers",
    ]
    for i, item in enumerate(items, 1):
        q = item.question
        lines += ["", f"### {i}. {q.q}", ""]
        found = {True: "found", False: "**missing**", None: "none (not covered by the docs)"}
        tags = [f"group: {q.group}"] if q.group else []
        tags.append(f"expected page: {found[item.expected_found]}")
        lines += [" · ".join(tags), ""]
        if not item.sources:
            lines.append("No relevant sources: the LLM was not called.")
            continue
        lines.append("Sources: " + " · ".join(f"[{s.n}] {s.title}" for s in item.sources))
        for model in models:
            run = item.runs[model]
            lines += ["", f"#### {model}: {describe(run)}", "", run.text or "(empty)"]
    return "\n".join(lines) + "\n"


def default_report_path() -> Path:
    return config.DATA_DIR / "bench" / f"{time.strftime('%Y%m%d-%H%M%S')}.md"
