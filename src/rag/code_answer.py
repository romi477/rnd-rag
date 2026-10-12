"""Answers about a failed job: a traceback and/or a question -> numbered code sources -> LLM answer with [n] citations.

The evidence is collected by exact lookups first (the frames of the traceback, their overrides, the error codes) and by
semantic search last, then packed into a token budget in that order: the innermost frames in full, the callers as
signature, docstring and the lines around the call, the rest until the budget is used.
"""

from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath

from rag import config
from rag.answer import (
    answer_in,
    answer_max_tokens,
    build_context,
    check_language,
    clean_text,
    language_rule,
    stream_llm,
)
from rag.chunking import approx_tokens
from rag.code_graph import CodeGraph, MethodRef
from rag.code_lookup import (
    COLUMNS,
    FrameUnit,
    find_error_code,
    named_definitions,
    related_methods,
    resolve_frames,
    scope_modules,
    sync_problems,
    whole_unit,
)
from rag.code_search import CodeSearcher
from rag.code_store import CodeStore
from rag.llm import LLM
from rag.sources.odoo_module import split_lines, unit_key
from rag.store import sql_str
from rag.traceback import ParsedTraceback, parse_traceback, semantic_query, version_note

CODE_SYSTEM_PROMPT_TEMPLATE = """\
You are a support engineer for a family of Odoo e-commerce connector modules. You get a failure (the exception \
and the frames of a Python traceback from a failed background job), numbered sources (code excerpts with their file \
path and line range, error-code registry entries, documentation) and a question.

Rules:
1. Explain the cause from the sources: name the function and the line where it failed and which condition raised \
the error. Frames are listed innermost first; the innermost frame in our code usually shows where the condition was \
checked, but the cause may be in the data, the settings or the other system.
2. Say what kind of problem it is: configuration or data on the client's side, an error returned by the e-commerce \
platform, a missing feature, or a probable defect in the code. Do not call it a defect unless the sources show it.
3. Give concrete next steps (settings to check, data to fix, what to send to support). Mention menu paths and \
setting names only if they are in the sources.
4. If the failure comes from outside the indexed code (Odoo core, the platform's API, a library) and the sources do \
not explain it, say so plainly and describe only what the message itself states.
5. Cite sources with their numbers in square brackets right after the statement, e.g. [1] or [2][3]. Never cite a \
number that is not in the sources. Write locations as `path:line`.
6. Line numbers belong to the indexed version. If a note says the client runs another version, say the code may differ.
7. Do not invent code, settings, error codes, versions or links. Do not suggest anything that changes data unless the \
sources describe it.
8. {language_rule} Start with the cause in one or two sentences, then details as a short list. Be concise."""


def code_system_prompt(lang: str = "en") -> str:
    rule = language_rule(
        lang, "Always answer in English.",
        "Always answer in {language}, although the sources are in English. Keep code, file paths, function and setting "
        "names and error messages exactly as they are: do not translate them.",
    )
    return CODE_SYSTEM_PROMPT_TEMPLATE.format(language_rule=rule)


CODE_SYSTEM_PROMPT = code_system_prompt("en")

NO_CODE_SOURCES_BY_LANG = {
    "en": "Nothing relevant was found in the indexed code for this failure.",
    "es": "No se encontró nada relevante en el código indexado para este fallo.",
    "de": "Im indexierten Code wurde für diesen Fehler nichts Relevantes gefunden.",
    "pl": "W zaindeksowanym kodzie nie znaleziono nic istotnego dla tego błędu.",
    "ru": "В проиндексированном коде не найдено ничего подходящего для этой ошибки.",
}
NO_CODE_SOURCES = NO_CODE_SOURCES_BY_LANG["en"]
DEFAULT_QUESTION = "Why did this fail and what should be done?"
ROLE_TEXT = {
    "frame": "frame of the traceback (innermost first)",
    "caller": "caller frame of the traceback (signature, docstring and the lines around the call)",
    "override": "another definition of the same method",
    "abstract": "abstract declaration of the adapter method",
    "definition": "definition of a function the question names",
    "error code": "error-code registry entry or a place that raises it",
    "search": "search result",
}
FRAME_MAX_TOKENS = 800  # a longer innermost frame is cut to a window around the failing line
FRAME_WINDOW = 40  # lines on each side of the failing line
SEARCH_K_WITH_FRAMES = 2  # search hits next to found frames (or named functions): a generic message pulls in unrelated code
DEFINITIONS_FULL = 2  # named functions quoted in full (the others as signature and docstring)
DEFINITION_MAX_TOKENS = 1500  # a longer definition is cut to its beginning
NO_TRACEBACK_NOTE = (
    "No traceback was given: this is a question about the code. Explain what the code does, step by step in the "
    "order it runs, from the sources; if the part asked about is not among them, say so.\n"
)
CALLER_FULL_TOKENS = 150  # a shorter caller is quoted whole
CALLER_WINDOW = 8  # lines on each side of the call in a caller
SEARCH_PIECE_CHARS = 1200
CODE_PIECE_CHARS = 1600


@dataclass
class Piece:
    role: str
    unit_id: str
    module: str
    module_version: str
    path: str
    line_start: int
    line_end: int
    label: str
    text: str
    name: str = ""  # the function / class the piece is about


@dataclass
class CodeSource:
    n: int
    role: str
    module: str
    module_version: str
    path: str
    line_start: int
    line_end: int
    label: str
    kind: str = "code"


@dataclass
class Evidence:
    tb: ParsedTraceback | None = None
    frames: list[FrameUnit] = field(default_factory=list)  # own frames, innermost first
    pieces: list[Piece] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def external_llm_blocked() -> str | None:
    """Why code may not go to the configured LLM, or None. The permission to send documentation to the Claude API
    does not cover source code."""
    if config.LLM_PROVIDER == "anthropic" and not config.CODE_EXTERNAL_LLM:
        return (
            "Code answers are off for the Claude provider: source code would leave this machine. Use a local LLM "
            "(RAG_LLM_PROVIDER=openai) or set RAG_CODE_EXTERNAL_LLM=1 if sending code to the Claude API is allowed."
        )
    return None


def evidence_dict(ev: Evidence) -> dict:
    """The evidence as JSON for the UI: what was parsed, where each frame landed, the pieces quoted to the LLM."""
    tb = ev.tb
    return {
        "exception": {"name": tb.exception.name, "message": tb.exception.message} if tb and tb.exception else None,
        "error_codes": tb.error_codes if tb else [],
        "notes": ev.notes,
        "frames": [
            {
                "module": f.module, "path": f.path, "line": f.frame.line, "func": f.frame.func, "found": f.found,
                "name": f.name, "line_start": f.line_start, "line_end": f.line_end, "name_matches": f.name_matches,
            }
            for f in ev.frames
        ],
        "pieces": [asdict(p) for p in ev.pieces],
    }


def signature_and_doc(text: str) -> str:
    """A function's decorators, signature and docstring with the body replaced by `...`; anything else: first lines."""
    lines = split_lines(text)
    try:
        node = ast.parse(textwrap.dedent(text)).body[0]
    except (SyntaxError, IndexError):
        return "\n".join(lines[:5])
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return "\n".join(lines[:5])
    first = node.body[0]
    is_doc = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    end = first.end_lineno if is_doc else first.lineno - 1
    return "\n".join([*lines[:end], " " * (node.col_offset + 4) + "..."])


def clip(text: str, max_tokens: int) -> str:
    """The beginning of `text` up to about `max_tokens`, cut at a line end and marked."""
    if approx_tokens(text) <= max_tokens:
        return text
    return text[: max_tokens * 4].rsplit("\n", 1)[0] + "\n    # ... (the rest of the function is not shown)"


def window(text: str, first_line: int, failing_line: int, radius: int) -> str:
    """`radius` lines on each side of the failing line (`first_line` is the file line of the text's first line).

    A failing line outside the text (the client runs another version, so a frame matched by function name carries
    its own line number) is moved to the nearest end: the window then shows the start or the end of the function."""
    lines = split_lines(text)
    at = min(max(failing_line - first_line, 0), len(lines) - 1)
    lo, hi = max(0, at - radius), min(len(lines), at + radius + 1)
    if lo == 0 and hi == len(lines):
        return text
    return "\n".join([*(["# ..."] if lo else []), *lines[lo:hi], *(["# ..."] if hi < len(lines) else [])])


def overlaps(pieces: list[Piece], module: str, path: str, start: int, end: int) -> bool:
    return any(p.module == module and p.path == path and p.line_start <= end and start <= p.line_end for p in pieces)


def unit_text(store: CodeStore, ref: MethodRef) -> dict | None:
    rows = store.rows_where(f"unit_id = {sql_str(unit_key(ref.module, ref.path, ref.line_start, 'method'))}", COLUMNS)
    return whole_unit(rows) if rows else None


def gather(
    store: CodeStore,
    graph: CodeGraph,
    question: str,
    traceback_text: str,
    searcher: CodeSearcher | None,
    modules: list[str] | None = None,
    client_version: str = "",
    search_k: int = 6,
    inner: int | None = None,
) -> Evidence:
    """Collect the evidence: frames, their overrides, error codes, then search hits."""
    inner = config.CODE_INNER_FRAMES if inner is None else inner
    ev = Evidence()
    tb = parse_traceback(traceback_text, set(graph.modules)) if traceback_text.strip() else None
    ev.tb = tb if tb and (tb.frames or tb.exception) else None
    pieces = ev.pieces
    if ev.tb:
        frames = ev.frames = resolve_frames(store, ev.tb)
        visible = {f.module for f in ev.tb.own_frames}
        for fu in frames:
            if not fu.found:
                ev.notes.append(f"{fu.module}/{fu.path}:{fu.frame.line} (in {fu.frame.func}) is not in the indexed code")
            elif fu.drifted:
                ev.notes.append(
                    f"{fu.module}/{fu.path}:{fu.frame.line} (in {fu.frame.func}) was matched by function name: the line "
                    f"number differs from the indexed version, the function is at lines {fu.line_start}-{fu.line_end}"
                )
        found = [fu for fu in frames if fu.found]
        for fu in found:
            if overlaps(pieces, fu.module, fu.path, fu.line_start, fu.line_end):
                continue
            pieces.append(frame_piece(fu, "frame" if len(pieces) < inner else "caller"))
        for fu in found:
            related = related_methods(graph, fu, visible)
            for role, refs in (("override", related.chain), ("abstract", related.abstract)):
                for ref in refs:
                    if overlaps(pieces, ref.module, ref.path, ref.line_start, ref.line_end):
                        continue
                    if row := unit_text(store, ref):
                        pieces.append(Piece(
                            role, row["unit_id"], row["module"], row["module_version"], row["path"], row["line_start"],
                            row["line_end"], f"{ref.class_name}.{ref.name}", signature_and_doc(row["text"]), ref.name,
                        ))
        scope = scope_modules(graph, visible)
        for code in ev.tb.error_codes:
            for u in find_error_code(store, code, scope):
                if not overlaps(pieces, u.module, u.path, u.line_start, u.line_end):
                    pieces.append(Piece(
                        "error code", u.unit_id, u.module, u.module_version, u.path, u.line_start, u.line_end,
                        f"{code} in {u.name}", u.text[:CODE_PIECE_CHARS], u.name,
                    ))
        problems = sync_problems(graph, store, visible)
        for module in sorted(visible):
            if note := version_note(client_version, graph.modules[module].version):
                ev.notes.append(f"{module}: {note}")
            if module in problems:
                ev.notes.append(f"{module}: the index and the graph differ ({problems[module]})")
    # With a module chosen, a named function is looked up in it and in what it depends on: the base module's method
    # (`_import_one_product` of `integration` for `integration_shopify`) is part of how the chosen module behaves.
    defined = named_definitions(graph, question, scope_modules(graph, modules) if modules else None)
    for i, ref in enumerate(defined):
        if overlaps(pieces, ref.module, ref.path, ref.line_start, ref.line_end) or not (row := unit_text(store, ref)):
            continue
        text = clip(row["text"], DEFINITION_MAX_TOKENS) if i < DEFINITIONS_FULL else signature_and_doc(row["text"])
        owner = f"{ref.model}.{ref.name}" if ref.model else f"{ref.class_name}.{ref.name}"
        pieces.append(Piece(
            "definition", row["unit_id"], row["module"], row["module_version"], row["path"], row["line_start"],
            row["line_end"], owner, text, ref.name,
        ))
    query = " ".join(filter(None, [question.strip(), semantic_query(ev.tb) if ev.tb else ""]))
    if searcher and query:
        k = min(search_k, SEARCH_K_WITH_FRAMES) if defined or any(fu.found for fu in ev.frames) else search_k
        for hit in searcher.search(query, k=k, modules=modules or None):
            if not overlaps(pieces, hit.module, hit.path, hit.line_start, hit.line_end):
                owner = f"{hit.model}.{hit.name}" if hit.model and hit.kind == "method" else hit.name
                pieces.append(Piece(
                    "search", hit.unit_id, hit.module, hit.module_version, hit.path, hit.line_start, hit.line_end,
                    f"{hit.kind} {owner}", hit.text[:SEARCH_PIECE_CHARS], hit.name,
                ))
    return ev


def frame_piece(fu: FrameUnit, role: str) -> Piece:
    owner = f"{fu.model}.{fu.name}" if fu.model else fu.name
    if role == "caller" and fu.kind == "method" and approx_tokens(fu.text) > CALLER_FULL_TOKENS:
        # The cause is often the call itself (its arguments): keep those lines next to the signature.
        text = signature_and_doc(fu.text) + "\n    # ...\n" + window(fu.text, fu.line_start, fu.frame.line, CALLER_WINDOW)
    elif approx_tokens(fu.text) > FRAME_MAX_TOKENS:
        text = window(fu.text, fu.line_start, fu.frame.line, FRAME_WINDOW)
    else:
        text = fu.text
    return Piece(
        role, fu.unit_id, fu.module, fu.module_version, fu.path, fu.line_start, fu.line_end,
        f"{owner}, line {fu.frame.line} in the traceback", text, fu.name,
    )


FENCE_LANGUAGE = {".py": "python", ".xml": "xml"}  # a README or an .rst file is plain text


def code_fence(text: str) -> str:
    """A fence longer than any run of backticks in `text`, so that text from a README or a comment cannot close it."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def build_code_context(pieces: list[Piece], max_tokens: int = config.CODE_CONTEXT_TOKENS) -> tuple[list[CodeSource], str]:
    """Number the pieces in order and pack them into the budget: a piece that does not fit is cut when it is
    the first or leaves room for 150 tokens, otherwise skipped (a smaller one later may still fit)."""
    sources: list[CodeSource] = []
    blocks: list[str] = []
    used = 0
    for p in pieces:
        text, room = clean_text(p.text), max_tokens - used
        if approx_tokens(text) > room:
            if room < 150 and sources:
                continue
            text = text[: max(room, 1) * 4].rsplit("\n", 1)[0] + "\n# ... truncated"
        used += approx_tokens(text)
        n = len(sources) + 1
        sources.append(CodeSource(n, p.role, p.module, p.module_version, p.path, p.line_start, p.line_end, p.label))
        version = f", version {p.module_version}" if p.module_version else ""
        fence = code_fence(text)
        blocks.append(
            f"[{n}] {p.module}/{p.path}:{p.line_start}-{p.line_end}  {p.label}\n"
            f"Role: {ROLE_TEXT.get(p.role, p.role)}{version}\n{fence}{FENCE_LANGUAGE.get(PurePosixPath(p.path).suffix, '')}"
            f"\n{text}\n{fence}"
        )
    return sources, "\n\n".join(blocks)


def failure_summary(ev: Evidence) -> str:
    if not ev.tb and not ev.notes:
        return ""
    lines = ["Failure:"]
    if ev.tb and ev.tb.exception:
        lines.append(f"{ev.tb.exception.name}: {ev.tb.exception.message[:1500]}")
    if ev.tb and ev.tb.own_frames:
        lines.append("Frames in the indexed modules, innermost first:")
        lines += [f"- {f.module}/{f.rel}:{f.line} in {f.func}" for f in ev.tb.own_frames]
    if ev.tb and ev.tb.error_codes:
        lines.append("Error codes: " + ", ".join(ev.tb.error_codes))
    lines += [f"Note: {n}" for n in ev.notes]
    return "\n".join(lines)


def build_code_messages(question: str, ev: Evidence, context: str, lang: str = "en") -> list[dict]:
    parts = [p for p in (failure_summary(ev), f"Sources:\n\n{context}") if p]
    return [
        {"role": "system", "content": code_system_prompt(lang)},
        {
            "role": "user",
            "content": "\n\n".join(parts)
            + f"\n\n---\n\n{'' if ev.tb else NO_TRACEBACK_NOTE}Question: {question.strip() or DEFAULT_QUESTION}{answer_in(lang)}",
        },
    ]


def stream_code_answer(
    question: str, ev: Evidence, llm: LLM, model: str | None = None, pages=None, lang: str = "en"
) -> Iterator[dict]:
    """Events: sources -> delta* -> done, or an error event. `pages` (Confluence hits) continue the numbering;
    `lang` is the language of the answer."""
    sources, context = build_code_context(ev.pieces)
    listed = [asdict(s) for s in sources]
    if pages:
        doc_sources, doc_context = build_context(pages, start=len(sources) + 1)
        listed += [asdict(s) for s in doc_sources]
        if doc_context:
            context = "\n\n".join(filter(None, [context, "Documentation:\n\n" + doc_context]))
    yield {"type": "sources", "sources": listed}
    if not listed:
        yield {"type": "delta", "text": NO_CODE_SOURCES_BY_LANG[check_language(lang)]}
        yield {"type": "done", "model": None, "took_ms": 0}
        return
    yield from stream_llm(
        build_code_messages(question, ev, context, lang), llm, model, answer_max_tokens(config.CODE_LLM_MAX_TOKENS, lang)
    )
