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
from typing import TYPE_CHECKING

import yaml

from rag import config
from rag.answer import Source, answer_max_tokens, build_context, build_messages
from rag.llm import LLM, LLMError

if TYPE_CHECKING:
    from rag.search import Mode, Searcher


@dataclass
class Run:
    model: str
    text: str = ""
    ttft_s: float | None = None  # time to the first answer token; None: no answer token at all
    total_s: float = 0.0
    gen_s: float | None = None  # from the first generated token (reasoning included) to the end
    prompt_tokens: int | None = None
    completion_tokens: int | None = None  # reasoning tokens included
    truncated: bool = False  # cut off by max_tokens
    error: str | None = None

    @property
    def tok_per_s(self) -> float | None:
        """Generation speed, excluding prompt processing (the time before the first token)."""
        gen_s = self.gen_s if self.gen_s is not None else self.total_s - (self.ttft_s or 0)
        return self.completion_tokens / gen_s if self.completion_tokens and gen_s > 0 else None


@dataclass
class Question:
    q: str
    group: str = ""
    # page ids that answer it; [] = the docs don't cover it; None = not specified
    expect: list[str] | None = None


@dataclass
class Item:
    question: Question
    sources: list[Source]
    runs: dict[str, Run] = field(default_factory=dict)  # model -> run; empty if no sources
    error: str | None = None  # the search failed

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
    for n, i in enumerate(yaml.safe_load(path.read_text()) or [], 1):
        if not isinstance(i, dict):
            i = {"q": str(i)}
        if "q" not in i:
            raise ValueError(f"{path}, item {n}: no 'q' field (quote a plain question that contains ': ')")
        expect = i.get("expect", i.get("relevant"))
        expect = None if expect is None else [str(e) for e in expect]
        questions.append(Question(str(i["q"]), str(i.get("group") or ""), expect))
    return [q for q in questions if group is None or q.group == group]


def generate(llm: LLM, model: str, messages: list[dict], **kwargs) -> Run:
    run, info, parts = Run(model), {}, []
    started = time.perf_counter()
    try:
        for piece in llm.stream_chat(messages, model=model, info=info, **kwargs):
            if run.ttft_s is None:
                run.ttft_s = time.perf_counter() - started
            parts.append(piece)
    except LLMError as exc:
        run.error = str(exc)
    ended = time.perf_counter()
    run.total_s = ended - started
    if "first_delta_at" in info:
        run.gen_s = ended - info["first_delta_at"]
    run.text = "".join(parts)
    usage = info.get("usage") or {}
    run.prompt_tokens = usage.get("prompt_tokens")
    run.completion_tokens = usage.get("completion_tokens")
    run.truncated = info.get("finish_reason") == "length"
    if not run.error and not run.text.strip():
        run.error = "empty answer"
    return run


def run_bench(
    searcher: Searcher,
    llm: LLM,
    questions: list[Question],
    models: list[str],
    mode: Mode = "hybrid",
    rerank: bool = True,
    log: Callable[[str], None] = lambda _: None,
    on_item: Callable[[list[Item]], None] = lambda _: None,
    lang: str = "en",
) -> list[Item]:
    """Run every question on every model. `on_item` gets the items so far after each question,
    so that a report survives an interruption. A model that fails to warm up stops the bench."""
    for model in models:  # load / warm up, not measured
        log(f"warming up {model}")
        warm = generate(llm, model, [{"role": "user", "content": "hi"}], max_tokens=1)
        if warm.error and warm.error != "empty answer":
            raise LLMError(f"{model}: {warm.error}")
    items = []
    for i, question in enumerate(questions):
        try:
            hits = searcher.search(question.q, k=10, mode=mode, rerank=rerank)  # same as the UI
        except Exception as exc:  # one broken question must not cost the runs done so far
            items.append(Item(question, [], error=f"search failed: {exc}"))
            log(f"{i + 1:>2} search failed: {exc}")
            on_item(items)
            continue
        sources, context = build_context(hits)
        item = Item(question, sources)
        items.append(item)
        if item.expected_found is False:
            log(f"{i + 1:>2} expected page is not among the sources")
        if not sources:
            log(f"{i + 1:>2} no sources, skipped: {question.q}")
        else:
            messages = build_messages(question.q, context, lang)
            limit = {} if lang == "en" else {"max_tokens": answer_max_tokens(config.LLM_MAX_TOKENS, lang)}
            # alternate the order so that no model always runs right after another one
            for model in models if i % 2 == 0 else models[::-1]:
                run = item.runs[model] = generate(llm, model, messages, **limit)
                log(f"{i + 1:>2} {model}: {describe(run)}")
        on_item(items)
    return items


def describe(run: Run) -> str:
    if run.error:
        return f"error: {run.error}"
    speed = f"{run.tok_per_s:.0f} tok/s" if run.tok_per_s else "? tok/s"
    cut = ", cut off at max_tokens" if run.truncated else ""
    return (f"first token {fmt(run.ttft_s)} s, total {run.total_s:.1f} s, "
            f"{run.completion_tokens or '?'} tok, {speed}{cut}")


def summarize(items: list[Item], model: str) -> dict:
    runs = [it.runs[model] for it in items if model in it.runs]
    ok = [r for r in runs if not r.error]

    def stat(values: list, fn) -> float | None:
        values = [v for v in values if v is not None]
        return fn(values) if values else None

    # Speed over all answers together (tokens / time), not a mean of per-answer ratios that
    # gives a 20-token answer the same weight as a 500-token one.
    timed = [(r.completion_tokens, r.gen_s if r.gen_s is not None else r.total_s - (r.ttft_s or 0))
             for r in ok if r.completion_tokens]
    gen_time = sum(s for _, s in timed)
    return {
        "model": model,
        "answers": len(ok),
        "errors": len(runs) - len(ok),
        "truncated": sum(r.truncated for r in ok),
        "ttft_avg": stat([r.ttft_s for r in ok], statistics.mean),
        "ttft_max": stat([r.ttft_s for r in ok], max),
        "total_avg": stat([r.total_s for r in ok], statistics.mean),
        "total_max": stat([r.total_s for r in ok], max),
        "total_sum": sum(r.total_s for r in ok),
        "tok_per_s": sum(t for t, _ in timed) / gen_time if gen_time > 0 else None,
        "out_tokens_avg": stat([r.completion_tokens for r in ok], statistics.mean),
    }


def fmt(value: float | None, spec: str = ".1f") -> str:
    return "–" if value is None else format(value, spec)


def summary_rows(items: list[Item], models: list[str]) -> list[list[str]]:
    rows = [["Model", "First token avg / max, s", "Answer avg / max, s", "Tokens/s",
             "Answer tokens", "Total, s", "Cut off", "Errors"]]
    for model in models:
        s = summarize(items, model)
        rows.append([
            model,
            f"{fmt(s['ttft_avg'])} / {fmt(s['ttft_max'])}",
            f"{fmt(s['total_avg'])} / {fmt(s['total_max'])}",
            fmt(s["tok_per_s"], ".0f"),
            fmt(s["out_tokens_avg"], ".0f"),
            fmt(s["total_sum"], ".0f"),
            str(s["truncated"]),
            str(s["errors"]),
        ])
    return rows


def retrieval_line(items: list[Item]) -> str:
    checked = [it.expected_found for it in items if it.expected_found is not None]
    uncovered = sum(1 for it in items if it.question.expect == [])
    line = f"Expected page among the sources: {sum(checked)}/{len(checked)}"
    return line + (f"; {uncovered} question(s) not covered by the docs" if uncovered else "")


def as_markdown_block(text: str) -> str:
    """Model output inside the report: an unclosed ``` fence would swallow the rest of the report."""
    if text.count("```") % 2:
        text += "\n```"
    return text


def render_report(items: list[Item], models: list[str], settings: str, partial: bool = False) -> str:
    rows = summary_rows(items, models)
    lines = [
        f"# LLM bench {time.strftime('%Y-%m-%d %H:%M')}" + (" (interrupted)" if partial else ""),
        "",
        settings,
        "",
        "## Summary",
        "",
        "| " + " | ".join(rows[0]) + " |",
        "|" + "---|" * len(rows[0]),
        *("| " + " | ".join(r) + " |" for r in rows[1:]),
        "",
        "Tokens/s: all answer tokens (reasoning included) over the generation time, after the first token.",
        "",
        retrieval_line(items),
        "",
        "## Answers",
    ]
    for i, item in enumerate(items, 1):
        q = item.question
        lines += ["", f"### {i}. {q.q}", ""]
        tags = [f"group: {q.group}"] if q.group else []
        if q.expect == []:
            tags.append("expected page: none (not covered by the docs)")
        elif q.expect:
            tags.append(f"expected page: {'found' if item.expected_found else '**missing**'}")
        if tags:
            lines += [" · ".join(tags), ""]
        if item.error:
            lines.append(f"**{item.error}**")
            continue
        if not item.sources:
            lines.append("No relevant sources: the LLM was not called.")
            continue
        lines.append("Sources: " + " · ".join(f"[{s.n}] {s.title}" for s in item.sources))
        for model in models:
            run = item.runs[model]
            lines += ["", f"#### {model}: {describe(run)}", "", as_markdown_block(run.text) or "(empty)"]
    return "\n".join(lines) + "\n"


def default_report_path() -> Path:
    return config.DATA_DIR / "bench" / f"{time.strftime('%Y%m%d-%H%M%S')}.md"
