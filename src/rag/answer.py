"""Grounded answer generation: top search results -> numbered sources -> LLM answer with [n] citations."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from rag import config
from rag.chunking import approx_tokens
from rag.llm import LLMClient, LLMError

if TYPE_CHECKING:  # rag.search pulls in torch: not needed to build a prompt
    from rag.search import PageHit

SYSTEM_PROMPT = """\
You are a support assistant that answers questions about product documentation.
You get numbered sources (documentation excerpts) and a question.

Rules:
1. The sources come from a search engine: some may be about another product or topic. First decide \
which sources actually address the question and use only those. Do not mix up products.
2. Use only facts from the relevant sources. If they do not answer the question, say so plainly \
("The documentation does not cover this.") and, if useful, point to the closest source.
3. Cite the sources you used with their numbers in square brackets right after the statement, \
e.g. [1] or [2][3]. Never cite a number that is not in the sources.
4. Start with a direct answer in one or two sentences, then add steps or details as a short list \
when that helps.
5. Keep menu paths, setting names, field names, error messages and code exactly as in the sources; \
format them as `code` or **bold**. Write menu paths with "→"; never use LaTeX.
6. Do not invent settings, menu paths, versions, prices or links.
7. Answer in the language of the question. Be concise."""

NO_SOURCES = "Nothing relevant was found in the documentation for this question."
EMPTY_ANSWER = "The model returned no answer text (a reasoning model may have spent max_tokens on thinking)."


@dataclass
class Source:
    n: int
    page_id: str
    title: str
    url: str


def clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_context(
    pages: list[PageHit],
    max_pages: int = config.LLM_CONTEXT_PAGES,
    max_tokens: int = config.LLM_CONTEXT_TOKENS,
) -> tuple[list[Source], str]:
    """Number the top pages as sources and pack their matched chunks into a token budget.

    Pages flagged `low_relevance` by the reranker are left out.
    """
    sources: list[Source] = []
    blocks: list[str] = []
    used = 0
    for page in [p for p in pages if not p.low_relevance][:max_pages]:
        parts = []
        for chunk in page.chunks:
            text = clean_text(chunk.text)
            size = approx_tokens(text)
            if used + size > max_tokens and (blocks or parts):
                break
            section = " > ".join(chunk.section_path)
            parts.append(f"(Section: {section})\n{text}" if section else text)
            used += size
        if not parts:
            break
        n = len(sources) + 1
        sources.append(Source(n, page.page_id, page.title, page.url))
        path = " > ".join(page.breadcrumbs) or page.product
        blocks.append(f"[{n}] {page.title}\nPath: {path}\n\n" + "\n\n".join(parts))
    return sources, "\n\n---\n\n".join(blocks)


def build_messages(query: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{context}\n\n---\n\nQuestion: {query}"},
    ]


def stream_answer(
    query: str, pages: list[PageHit], llm: LLMClient, model: str | None = None
) -> Iterator[dict]:
    """Events: sources -> delta* -> done, or an error event instead of done.

    `model` overrides the client's default (the model loaded in LM Studio). `done.truncated` is
    true when the answer was cut off by max_tokens.
    """
    sources, context = build_context(pages)
    yield {"type": "sources", "sources": [asdict(s) for s in sources]}
    if not sources:
        yield {"type": "delta", "text": NO_SOURCES}
        yield {"type": "done", "model": None, "took_ms": 0}
        return

    started, info, answered = time.perf_counter(), {}, False
    try:
        model = model or llm.resolve_model()
        for text in llm.stream_chat(build_messages(query, context), model=model, info=info):
            answered = answered or bool(text.strip())
            yield {"type": "delta", "text": text}
    except LLMError as exc:
        yield {"type": "error", "message": str(exc)}
        return
    if not answered:
        yield {"type": "error", "message": EMPTY_ANSWER}
        return
    yield {
        "type": "done",
        "model": model,
        "took_ms": round((time.perf_counter() - started) * 1000),
        "truncated": info.get("finish_reason") == "length",
    }
