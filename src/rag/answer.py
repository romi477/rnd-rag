"""Grounded answer generation: top search results -> numbered sources -> LLM answer with [n] citations."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from rag import config
from rag.chunking import approx_tokens
from rag.llm import LLM, LLMError

if TYPE_CHECKING:  # rag.search pulls in torch: not needed to build a prompt
    from rag.search import PageHit

SYSTEM_PROMPT_TEMPLATE = """\
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
7. {language_rule}"""

ENGLISH_RULE = "Always answer in English, like the documentation. Be concise."


def check_language(lang: str | None = None) -> str:
    """The language code of a request: `ru`, `RU`, `ru-RU` and `ru_RU` are Russian; nothing means the default
    (`RAG_ANSWER_LANGUAGE`, read the same way). Raises ValueError for a language that is not supported."""
    named = (lang or "").strip()
    raw = named or config.ANSWER_LANGUAGE  # the configured default is written like any other code
    code = raw.strip().lower().replace("_", "-").split("-")[0]
    if code not in config.ANSWER_LANGUAGES:
        source = "" if named else " (RAG_ANSWER_LANGUAGE)"
        raise ValueError(f"Unsupported language {raw!r}{source}, expected one of {', '.join(config.ANSWER_LANGUAGES)}")
    return code


def language_name(lang: str) -> str:
    return config.ANSWER_LANGUAGES[check_language(lang)][0]


def language_rule(lang: str, english: str, other: str) -> str:
    """The prompt's last rule: `english` as it always was, or `other` (with {language}) for another language."""
    code = check_language(lang)
    return english if code == "en" else other.format(language=language_name(code))


def answer_max_tokens(base: int, lang: str) -> int:
    """The answer's token limit for a language: `base` times the language's factor (other languages need more tokens)."""
    return round(base * config.LANGUAGE_TOKEN_FACTOR.get(check_language(lang), 1.0))


def answer_in(lang: str) -> str:
    """The reminder after the question for a language other than English (the sources are English)."""
    return "" if check_language(lang) == "en" else f"\n\nAnswer in {language_name(lang)}."


def system_prompt(lang: str = "en") -> str:
    rule = language_rule(
        lang, ENGLISH_RULE,
        "Always answer in {language}, although the sources are in English. Keep menu paths, setting names, field names, "
        "error messages and code exactly as in the sources: do not translate them. Be concise.",
    )
    return SYSTEM_PROMPT_TEMPLATE.format(language_rule=rule)


SYSTEM_PROMPT = system_prompt("en")

NO_SOURCES_BY_LANG = {
    "en": "Nothing relevant was found in the documentation for this question.",
    "es": "No se encontró nada relevante en la documentación para esta pregunta.",
    "de": "Zu dieser Frage wurde in der Dokumentation nichts Relevantes gefunden.",
    "pl": "W dokumentacji nie znaleziono nic istotnego w tej sprawie.",
    "ru": "В документации не найдено ничего подходящего для этого вопроса.",
}
NO_SOURCES = NO_SOURCES_BY_LANG["en"]
EMPTY_ANSWER = "The model returned no answer text (a reasoning model may have spent max_tokens on thinking)."


@dataclass
class Source:
    n: int
    page_id: str
    title: str
    url: str
    kind: str = "page"  # code answers number their code sources too: {"kind": "code", ...}


def clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_context(
    pages: list[PageHit],
    max_pages: int = config.LLM_CONTEXT_PAGES,
    max_tokens: int = config.LLM_CONTEXT_TOKENS,
    start: int = 1,
) -> tuple[list[Source], str]:
    """Number the top pages as sources and pack their matched chunks into a token budget.

    Pages flagged `low_relevance` by the reranker are left out. `start` is the number of the first source
    (a code answer that also quotes the docs continues its own numbering).
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
        n = start + len(sources)
        sources.append(Source(n, page.page_id, page.title, page.url))
        path = " > ".join(page.breadcrumbs) or page.product
        blocks.append(f"[{n}] {page.title}\nPath: {path}\n\n" + "\n\n".join(parts))
    return sources, "\n\n---\n\n".join(blocks)


def token_usage(usage: dict | None) -> dict | None:
    """Token counts as the LLM server reported them (UI debug mode): prompt, answer and, when the
    server tells, the reasoning tokens included in the answer count."""
    if not usage or usage.get("prompt_tokens") is None:
        return None
    counts = {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage.get("completion_tokens")}
    details = usage.get("completion_tokens_details") or {}
    if details.get("reasoning_tokens"):
        counts["reasoning_tokens"] = details["reasoning_tokens"]
    return counts


def build_messages(query: str, context: str, lang: str = "en") -> list[dict]:
    return [
        {"role": "system", "content": system_prompt(lang)},
        {"role": "user", "content": f"Sources:\n\n{context}\n\n---\n\nQuestion: {query}{answer_in(lang)}"},
    ]


def stream_llm(
    messages: list[dict], llm: LLM, model: str | None = None, max_tokens: int | None = None
) -> Iterator[dict]:
    """Events after the sources: delta* -> done, or an error event instead of done.

    `model` overrides the client's default (the model loaded in LM Studio, or claude-sonnet-5-5). `done.truncated` is
    true when the answer was cut off by max_tokens.
    """
    started, info, answered, first_token_ms = time.perf_counter(), {}, False, None
    try:
        model = model or llm.resolve_model()
        limit = {} if max_tokens is None else {"max_tokens": max_tokens}  # none: the client's own default
        for text in llm.stream_chat(messages, model=model, info=info, **limit):
            if first_token_ms is None:  # the first answer text, as the bench measures it
                first_token_ms = round((time.perf_counter() - started) * 1000)
            answered = answered or bool(text.strip())
            yield {"type": "delta", "text": text}
    except LLMError as exc:
        yield {"type": "error", "message": str(exc)}
        return
    if not answered:
        yield {"type": "error", "message": EMPTY_ANSWER}
        return
    done = {
        "type": "done",
        "model": model,
        "took_ms": round((time.perf_counter() - started) * 1000),
        "truncated": info.get("finish_reason") == "length",
    }
    if first_token_ms is not None:
        done["first_token_ms"] = first_token_ms
    if usage := token_usage(info.get("usage")):
        done["usage"] = usage
    yield done


def stream_answer(
    query: str, pages: list[PageHit], llm: LLM, model: str | None = None, lang: str = "en"
) -> Iterator[dict]:
    """Events: sources -> delta* -> done, or an error event instead of done. `lang`: the language of the answer."""
    sources, context = build_context(pages)
    yield {"type": "sources", "sources": [asdict(s) for s in sources]}
    if not sources:
        yield {"type": "delta", "text": NO_SOURCES_BY_LANG[check_language(lang)]}
        yield {"type": "done", "model": None, "took_ms": 0}
        return
    # English keeps the client's default limit; another language gets more tokens for the same answer
    max_tokens = None if check_language(lang) == "en" else answer_max_tokens(config.LLM_MAX_TOKENS, lang)
    yield from stream_llm(build_messages(query, context, lang), llm, model, max_tokens)
