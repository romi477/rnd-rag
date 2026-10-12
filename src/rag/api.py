"""HTTP API; also serves the static UI from src/web/.  Run with `rag serve`."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from rag import config
from rag.answer import stream_answer
from rag.llm import LLMError, make_llm
from rag.search import MODES, PageHit, Searcher

state: dict = {}
log = logging.getLogger(__name__)
# Models on MPS are not safe to call from several threadpool workers at once.
lock = threading.Lock()


def warm_up(searcher: Searcher) -> None:
    """Load both models before the first search (rerank is on by default in the UI); requests
    that come earlier wait on the lock instead of the server refusing connections."""
    with lock:
        searcher.embedder.embed_query("warm up")
        searcher.reranker.score("warm up", ["warm up"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    searcher = Searcher()
    state["searcher"] = searcher
    state["llm"] = make_llm()
    threading.Thread(target=warm_up, args=(searcher,), daemon=True).start()
    yield
    state.clear()


# No /docs or /redoc: they load scripts from a CDN, and everything here stays local.
app = FastAPI(title="rnd-rag", lifespan=lifespan, docs_url=None, redoc_url=None)
# Only answer to requests addressed to this machine by name: a page on another site that rebinds
# its DNS name to 127.0.0.1 would otherwise read the docs. Bind to another interface? Add its name.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.ALLOWED_HOSTS)


@app.middleware("http")
async def revalidate_ui(request: Request, call_next):
    """The UI files have no build step to version them: make browsers check for a newer copy
    (a cheap 304 via ETag) instead of mixing a cached app.js with a new index.html."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


def run_search(
    q: str, k: int, mode: str, product: str | None, rerank: bool
) -> tuple[list[PageHit], int]:
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    searcher: Searcher = state["searcher"]
    started = time.perf_counter()
    with lock:
        hits = searcher.search(q, k=k, mode=mode, product=product or None, rerank=rerank)
    return hits, round((time.perf_counter() - started) * 1000)


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, max_length=500),
    k: int = Query(10, ge=1, le=50),
    mode: str = "hybrid",
    product: str | None = None,
    rerank: bool = False,
) -> dict:
    hits, took_ms = run_search(q, k, mode, product, rerank)
    return {"query": q, "took_ms": took_ms, "results": [asdict(h) for h in hits]}


# Start times of the answers of the last hour, for RAG_ANSWERS_PER_HOUR.
answer_times: deque[float] = deque()
answer_times_lock = threading.Lock()


def take_answer_slot() -> bool:
    limit = config.ANSWERS_PER_HOUR
    if limit <= 0:
        return True
    now = time.monotonic()
    with answer_times_lock:
        while answer_times and now - answer_times[0] > 3600:
            answer_times.popleft()
        if len(answer_times) >= limit:
            return False
        answer_times.append(now)
        return True


@app.get("/api/answer")
def answer(
    q: str = Query(..., min_length=1, max_length=500),
    k: int = Query(10, ge=1, le=50),
    mode: str = "hybrid",
    product: str | None = None,
    rerank: bool = True,  # the relevance cut-off keeps unrelated pages out of the prompt
    model: str | None = Query(None, max_length=200, description="LLM id; default: RAG_LLM_MODEL, the model loaded in LM Studio, or claude-sonnet-5-5"),
) -> StreamingResponse:
    """Search + LLM answer as NDJSON events: results, sources, delta*, done | error.

    The search runs once: its results are the first event, the top pages become the sources.
    """
    if not take_answer_slot():
        raise HTTPException(429, f"Too many LLM answers: the limit is {config.ANSWERS_PER_HOUR} per hour (RAG_ANSWERS_PER_HOUR)")
    if model:
        try:  # the Claude client reuses its model list for a few minutes: no request per answer
            known = {m["id"] for m in state["llm"].list_models()}
        except LLMError:
            known = None  # server down: the answer stream reports it
        if known is not None and model not in known:
            raise HTTPException(400, f"Unknown LLM model {model!r}")
    hits, took_ms = run_search(q, k, mode, product, rerank)

    def events() -> Iterator[str]:
        first = {"type": "results", "query": q, "took_ms": took_ms, "results": [asdict(h) for h in hits]}
        yield json.dumps(first, ensure_ascii=False) + "\n"
        # The LLM runs outside the model lock.
        try:
            for event in stream_answer(q, hits, state["llm"], model=model or None):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:  # the headers are sent: report it in the stream instead of cutting it off
            log.exception("Answer failed")
            yield json.dumps({"type": "error", "message": f"Internal error: {exc}"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/api/llm")
def llm_status() -> dict:
    """Whether answer generation is available, the default model, all chat models and the provider
    ("openai": LM Studio or another OpenAI-compatible server; "anthropic": the Claude API)."""
    return state["llm"].status() | {"provider": config.LLM_PROVIDER}


@app.get("/api/products")
def products() -> list[str]:
    return state["searcher"].products()


@app.get("/api/pages")
def pages() -> dict:
    """Every indexed page with its Confluence link, sorted by product and path."""
    pages = state["searcher"].pages()
    return {"count": len(pages), "pages": pages}


# Mounted last so that /api/* routes take precedence.
app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")
