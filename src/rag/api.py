"""HTTP API; also serves the static UI from src/web/.  Run with `rag serve`."""

from __future__ import annotations

import base64
import hmac
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from rag import config
from rag.answer import check_language, stream_answer
from rag.code_answer import evidence_dict, external_llm_blocked, gather, stream_code_answer
from rag.chunking import load_page
from rag.code_graph import CodeGraph, graph_path
from rag.code_lookup import sync_problems
from rag.code_search import CodeSearcher
from rag.code_store import CodeStore
from rag.llm import ClaudeClient, LLMError, claude_key_set, make_llm
from rag.search import MODES, PageHit, Searcher
from rag.store import store_path
from rag.translate import check_target, stream_translation
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


def token_given(header: str) -> str | None:
    """The token in an `Authorization: Basic` header: the password of `user:password`, or the whole value
    when it has no colon. None for any other kind of header or one that does not decode."""
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "basic":
        return None
    try:
        decoded = base64.b64decode(value.strip(), validate=True).decode()
    except (ValueError, UnicodeDecodeError):
        return None
    return decoded.partition(":")[2] if ":" in decoded else decoded


@app.middleware("http")
async def require_token(request: Request, call_next):
    """With RAG_API_TOKEN set, the UI and every endpoint need it (HTTP Basic: the browser asks once and then
    sends it with each request, so the page's own fetch calls need no code). Read per request, not at import."""
    expected = config.API_TOKEN
    if expected:
        given = token_given(request.headers.get("authorization", ""))
        if given is None or not hmac.compare_digest(given.encode(), expected.encode()):
            return Response(
                "Authentication required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="rnd-rag", charset="UTF-8"'}
            )
    return await call_next(request)


# Only answer to requests addressed to this machine by name: a page on another site that rebinds
# its DNS name to 127.0.0.1 would otherwise read the docs. Bind to another interface? Add its name.
# Added after the token check, so it runs first.
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


def check_model(model: str | None, llm=None) -> None:
    """400 for a model the LLM client does not offer (`llm`: the answer client unless another is given)."""
    if not model:
        return
    try:  # the Claude client reuses its model list for a few minutes: no request per answer
        known = {m["id"] for m in (llm or state["llm"]).list_models()}
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
    ("openai": LM Studio or another OpenAI-compatible server; "anthropic": the Claude API); `translate`: whether
    the Browse translation can run (it always uses the Claude API, so it needs the key)."""
    return state["llm"].status() | {"provider": config.LLM_PROVIDER} | translate_status()


def translate_status() -> dict:
    """What the Browse translation can use: `translate` (the Claude key is set), its default model and the models it offers
    (just the default when the list cannot be fetched)."""
    ready = claude_key_set()
    models: list[str] = []
    if ready:
        try:
            models = [m["id"] for m in translator().list_models()]
        except LLMError:
            models = [config.TRANSLATE_MODEL]
    return {"translate": ready, "translate_model": config.TRANSLATE_MODEL, "translate_models": models}


@app.get("/api/products")
def products() -> list[str]:
    return state["searcher"].products()


def created_dates() -> dict[str, str]:
    """page id -> when the page was created in Confluence, from the synced metadata (the index doesn't hold it)."""
    dates = {}
    for path in config.RAW_DIR.glob(f"{config.SPACE_KEY}/pages/*.json"):
        try:
            dates[path.stem] = json.loads(path.read_text()).get("created") or ""
        except (OSError, ValueError):
            continue  # a broken copy: the page just has no date
    return dates


@app.get("/api/pages")
def pages() -> dict:
    """Every indexed page with its Confluence link, sorted by product and path; `created` is the page's
    creation date in Confluence ("" when unknown)."""
    created = created_dates()
    pages = [p | {"created": created.get(p["page_id"], "")} for p in state["searcher"].pages()]
    return {"count": len(pages), "pages": pages}


def synced_page(page_id: str):
    """One synced article, from `raw-data/` (the server holds no Confluence token). The id is digits only, so it
    cannot name another path."""
    meta = next(iter(config.RAW_DIR.glob(f"*/pages/{page_id}.json")), None)
    if meta is None:
        raise HTTPException(404, f"No synced page {page_id}")
    try:
        return load_page(meta)
    except (OSError, ValueError, KeyError) as exc:
        log.warning("Broken synced page %s: %s", meta, exc)
        raise HTTPException(500, f"The synced copy of page {page_id} is unreadable: run the sync again") from exc


@app.get("/api/pages/{page_id}")
def page(page_id: str = Path(..., pattern=r"^[0-9]{1,20}$")) -> dict:
    """One article as the sync stored it (Markdown, without the title heading), for the Browse preview. `updated` is
    the page's last change at the time of the sync."""
    found = synced_page(page_id)
    return {
        "page_id": found.id, "title": found.title, "url": found.url, "breadcrumbs": found.breadcrumbs,
        "product": found.product, "updated": found.updated, "created": found.created, "labels": found.labels,
        "markdown": found.body,
    }


def image_type(path) -> str | None:
    """The media type of a raster picture, from the first bytes of the file (some pasted pictures have no extension in
    their title). None for anything else: an SVG can carry a script, other attachments are not pictures."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return None
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


@app.get("/api/pages/{page_id}/images/{name}")
def page_image(page_id: str = Path(..., pattern=r"^[0-9]{1,20}$"), name: str = Path(..., max_length=300)) -> FileResponse:
    """A picture of an article, by the file name the page shows (`[image: name.png]` in its Markdown): the copy the sync
    downloaded with `--attachments`. The name is looked up in the page's `_index.json`; it never builds a path, and a file
    outside the page's folder is not served."""
    for index_path in config.RAW_DIR.glob(f"*/attachments/{page_id}/_index.json"):
        try:
            entries = list(json.loads(index_path.read_text()).values())
        except (OSError, ValueError, AttributeError):
            continue
        folder = index_path.parent.resolve()
        for entry in entries:
            if isinstance(entry, dict) and entry.get("title") == name and isinstance(entry.get("name"), str):
                path = (folder / entry["name"]).resolve()
                media = image_type(path) if path.parent == folder and path.is_file() else None
                if media:
                    return FileResponse(path, media_type=media, headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-cache"})
    raise HTTPException(404, f"No synced image {name!r} for page {page_id}")


def translator() -> ClaudeClient:
    """The client that translates articles: always the Claude API (Haiku), whatever RAG_LLM_PROVIDER is."""
    if "translator" not in state:
        state["translator"] = ClaudeClient(model=config.TRANSLATE_MODEL, reasoning="none")
    return state["translator"]


@app.get("/api/pages/{page_id}/translate")
def translate_page(
    page_id: str = Path(..., pattern=r"^[0-9]{1,20}$"),
    lang: str = Query(..., min_length=2, max_length=10, description="Language to translate into: es, de, pl or ru"),
    model: str | None = Query(None, max_length=200, description="Claude model (default: RAG_TRANSLATE_MODEL)"),
) -> StreamingResponse:
    """The article translated by AI as NDJSON events: title?, delta*, done | error (the Browse preview shows them as
    they come). Counts toward RAG_ANSWERS_PER_HOUR: every translation is billed by the Claude API."""
    try:
        code = check_target(lang)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    found = synced_page(page_id)
    check_model(model, translator())
    if not take_answer_slot():
        raise HTTPException(429, f"Too many LLM requests: the limit is {config.ANSWERS_PER_HOUR} per hour (RAG_ANSWERS_PER_HOUR)")

    def events() -> Iterator[str]:
        try:
            for event in stream_translation(found.title, found.body, translator(), code, model=model or config.TRANSLATE_MODEL):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:  # the headers are sent: report it in the stream instead of cutting it off
            log.exception("Translation failed")
            yield json.dumps({"type": "error", "message": f"Internal error: {exc}"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


# Mounted last so that /api/* routes take precedence.
app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")
