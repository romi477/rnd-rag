"""HTTP API; also serves the static UI from src/web/.  Run with `rag serve`."""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from rag import config
from rag.search import MODES, Searcher

state: dict = {}
# Models on MPS are not safe to call from several threadpool workers at once.
lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    searcher = Searcher()
    searcher.embedder.embed_query("warm up")  # load the model before the first request
    state["searcher"] = searcher
    yield
    state.clear()


app = FastAPI(title="rnd-rag", lifespan=lifespan)


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, max_length=500),
    k: int = Query(10, ge=1, le=50),
    mode: str = "hybrid",
    product: str | None = None,
    rerank: bool = False,
) -> dict:
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    searcher: Searcher = state["searcher"]
    started = time.perf_counter()
    with lock:
        hits = searcher.search(q, k=k, mode=mode, product=product or None, rerank=rerank)
    return {
        "query": q,
        "took_ms": round((time.perf_counter() - started) * 1000),
        "results": [asdict(h) for h in hits],
    }


@app.get("/api/products")
def products() -> list[str]:
    return state["searcher"].products()


# Mounted last so that /api/* routes take precedence.
app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")
