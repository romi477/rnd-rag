"""HTTP API; also serves the static UI from src/web/.  Run with `rag serve`."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from rag import config
from rag.answer import stream_answer
from rag.llm import LLMClient
from rag.search import MODES, PageHit, Searcher

state: dict = {}
# Models on MPS are not safe to call from several threadpool workers at once.
lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    searcher = Searcher()
    # Load the models before the first request (rerank is on by default in the UI).
    searcher.embedder.embed_query("warm up")
    searcher.reranker.score("warm up", ["warm up"])
    state["searcher"] = searcher
    state["llm"] = LLMClient()
    yield
    state.clear()


app = FastAPI(title="rnd-rag", lifespan=lifespan)


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


@app.get("/api/answer")
def answer(
    q: str = Query(..., min_length=1, max_length=500),
    k: int = Query(10, ge=1, le=50),
    mode: str = "hybrid",
    product: str | None = None,
    rerank: bool = False,
    model: str | None = Query(None, description="LLM id; default: the model loaded in LM Studio"),
) -> StreamingResponse:
    """Search + LLM answer as NDJSON events: results, sources, delta*, done | error.

    The search runs once: its results are the first event, the top pages become the sources.
    """
    hits, took_ms = run_search(q, k, mode, product, rerank)

    def events() -> Iterator[str]:
        first = {"type": "results", "query": q, "took_ms": took_ms, "results": [asdict(h) for h in hits]}
        yield json.dumps(first, ensure_ascii=False) + "\n"
        # The LLM runs outside the model lock.
        for event in stream_answer(q, hits, state["llm"], model=model or None):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/api/llm")
def llm_status() -> dict:
    """Whether answer generation is available, the default model and all chat models."""
    return state["llm"].status()


@app.get("/api/products")
def products() -> list[str]:
    return state["searcher"].products()


# Mounted last so that /api/* routes take precedence.
app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")
