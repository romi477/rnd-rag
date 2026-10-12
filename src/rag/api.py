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
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from rag import config
from rag.answer import check_language, stream_answer
from rag.code_answer import evidence_dict, external_llm_blocked, gather, stream_code_answer
from rag.code_graph import CodeGraph, graph_path
from rag.code_lookup import sync_problems
from rag.code_search import CodeSearcher
from rag.code_store import CodeStore
from rag.llm import LLMError, make_llm
from rag.search import MODES, PageHit, Searcher
from rag.store import store_path
from rag.traceback import semantic_query

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
    check_language(None)  # a bad RAG_ANSWER_LANGUAGE stops the server here, not at the first answer
    searcher = Searcher()
    state["searcher"] = searcher
    state["llm"] = make_llm()
    # The code corpus is optional (experimental) and may be added while the server runs: see code_searcher().
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


def check_model(model: str | None) -> None:
    if not model:
        return
    try:  # the Claude client reuses its model list for a few minutes: no request per answer
        known = {m["id"] for m in state["llm"].list_models()}
    except LLMError:
        known = None  # server down: the answer stream reports it
    if known is not None and model not in known:
        raise HTTPException(400, f"Unknown LLM model {model!r}")


def requested_language(lang: str | None) -> str:
    try:
        return check_language(lang)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/languages")
def languages() -> dict:
    """The languages an answer can be written in, and the server's default."""
    return {
        "default": check_language(None),
        "languages": [{"code": c, "name": n, "native": nat} for c, (n, nat) in config.ANSWER_LANGUAGES.items()],
    }


@app.get("/api/answer")
def answer(
    q: str = Query(..., min_length=1, max_length=500),
    k: int = Query(10, ge=1, le=50),
    mode: str = "hybrid",
    product: str | None = None,
    rerank: bool = True,  # the relevance cut-off keeps unrelated pages out of the prompt
    model: str | None = Query(None, max_length=200, description="LLM id; default: RAG_LLM_MODEL, the model loaded in LM Studio, or claude-sonnet-5-5"),
    lang: str | None = Query(None, max_length=10, description="Answer language: en, es, de, pl or ru (default: RAG_ANSWER_LANGUAGE)"),
) -> StreamingResponse:
    """Search + LLM answer as NDJSON events: results, sources, delta*, done | error.

    The search runs once: its results are the first event, the top pages become the sources.
    """
    language = requested_language(lang)  # before the slot: a rejected request must not use up the hourly limit
    check_model(model)
    if not take_answer_slot():
        raise HTTPException(429, f"Too many LLM answers: the limit is {config.ANSWERS_PER_HOUR} per hour (RAG_ANSWERS_PER_HOUR)")
    hits, took_ms = run_search(q, k, mode, product, rerank)

    def events() -> Iterator[str]:
        first = {"type": "results", "query": q, "took_ms": took_ms, "results": [asdict(h) for h in hits]}
        yield json.dumps(first, ensure_ascii=False) + "\n"
        # The LLM runs outside the model lock.
        try:
            for event in stream_answer(q, hits, state["llm"], model=model or None, lang=language):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:  # the headers are sent: report it in the stream instead of cutting it off
            log.exception("Answer failed")
            yield json.dumps({"type": "error", "message": f"Internal error: {exc}"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


def code_searcher() -> CodeSearcher:
    """The code searcher, opened on first use: `rag module add` may create the code index after the server started."""
    searcher = state.get("code_searcher")
    if searcher is None:
        embedder = state["searcher"].embedder
        store = CodeStore(store_path(embedder.model_name))
        if not store.exists():
            raise HTTPException(503, "No code index: add modules with `rag module add`")
        searcher = state["code_searcher"] = CodeSearcher(embedder, store)
    return searcher


@app.get("/api/code/modules")
def code_modules() -> dict:
    """Indexed Odoo addons: version, git ref they were read at, number of methods, and `problem` when the search
    index does not hold the version the graph describes (null otherwise)."""
    searcher = code_searcher()
    graph = CodeGraph.load(graph_path())
    problems = sync_problems(graph, searcher.store)
    methods: dict[str, int] = {}
    for m in graph.methods:
        methods[m.module] = methods.get(m.module, 0) + 1
    return {"modules": [
        {"name": n, "version": graph.modules[n].version, "origin": graph.modules[n].origin,
         "ref": graph.modules[n].ref, "methods": methods.get(n, 0), "problem": problems.get(n)}
        for n in graph.order()
    ]}


@app.get("/api/code/search")
def code_search(
    q: str = Query(..., min_length=1, max_length=500),
    k: int = Query(10, ge=1, le=50),
    mode: str = "hybrid",
    module: list[str] = Query(default=[]),
    kind: list[str] = Query(default=[]),
) -> dict:
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    searcher = code_searcher()
    started = time.perf_counter()
    with lock:
        hits = searcher.search(q, k=k, mode=mode, modules=module or None, kinds=kind or None)
    return {"query": q, "took_ms": round((time.perf_counter() - started) * 1000), "results": [asdict(h) for h in hits]}


class CodeAnswerRequest(BaseModel):
    question: str = Field("", max_length=500)
    traceback: str = Field("", max_length=30000)  # a failed job's traceback as pasted
    modules: list[str] = Field(default_factory=list, max_length=20)  # search only these modules
    client_version: str = Field("", max_length=50)  # module version on the client's server
    docs: bool = False  # also quote Confluence pages
    answer: bool = True  # False: only the evidence (the `results` event), no LLM, no use of the hourly limit
    model: str | None = Field(None, max_length=200)
    language: str | None = Field(None, max_length=10)  # answer language: en, es, de, pl or ru


@app.post("/api/code/answer")
def code_answer(req: CodeAnswerRequest) -> StreamingResponse:
    """Explain a failed job from its traceback and/or a question, as NDJSON events:
    results (evidence), sources, delta*, done | error. With `answer: false` only `results`."""
    if not req.question.strip() and not req.traceback.strip():
        raise HTTPException(400, "Give a question or a traceback")
    searcher = code_searcher()
    language = requested_language(req.language)
    if req.answer:
        if reason := external_llm_blocked():
            raise HTTPException(403, reason)
        check_model(req.model)  # before the slot: a rejected request must not use up the hourly limit
        if not take_answer_slot():
            raise HTTPException(429, f"Too many LLM answers: the limit is {config.ANSWERS_PER_HOUR} per hour (RAG_ANSWERS_PER_HOUR)")
    graph = CodeGraph.load(graph_path())
    started = time.perf_counter()
    with lock:
        ev = gather(
            searcher.store, graph, req.question, req.traceback, searcher, req.modules or None, req.client_version.strip()
        )
        pages = None
        if req.docs:
            query = semantic_query(ev.tb) if ev.tb else req.question
            pages = state["searcher"].search(query or req.question, k=config.LLM_CONTEXT_PAGES, rerank=True)
    took_ms = round((time.perf_counter() - started) * 1000)

    def events() -> Iterator[str]:
        first = {"type": "results", "took_ms": took_ms} | evidence_dict(ev)
        yield json.dumps(first, ensure_ascii=False) + "\n"
        if not req.answer:
            return
        try:  # the LLM runs outside the model lock
            for event in stream_code_answer(req.question, ev, state["llm"], model=req.model or None, pages=pages, lang=language):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:
            log.exception("Code answer failed")
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
