"""Incremental indexing: synced pages -> chunks -> embeddings -> LanceDB."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rag import config
from rag.chunking import chunk_page, load_pages
from rag.embedding import Embedder
from rag.store import Store, store_path


@dataclass
class IndexStats:
    pages: int = 0
    updated: int = 0
    removed: int = 0
    chunks_written: int = 0
    total_chunks: int = 0
    full_rebuild: bool = False


def index_space(
    space: str = config.SPACE_KEY,
    embedder: Embedder | None = None,
    store: Store | None = None,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> IndexStats:
    embedder = embedder or Embedder()
    store = store or Store(store_path(embedder.model_name))
    stats = IndexStats()

    if force or not store.exists() or store.meta.get("model") != embedder.model_name:
        log(f"Full rebuild with {embedder.model_name}")
        store.reset(embedder.model_name, embedder.dim)
        stats.full_rebuild = True
    indexed = store.page_versions()

    pages = load_pages(config.RAW_DIR / space)
    stats.pages = len(pages)
    current = {p.id for p in pages}

    chunks, updated_ids = [], []
    for page in pages:
        if indexed.get(page.id) == page.version:
            continue
        page_chunks = chunk_page(page)
        if not page_chunks and page.id not in indexed:
            continue  # empty page that was never indexed
        chunks.extend(page_chunks)
        updated_ids.append(page.id)

    removed = [pid for pid in indexed if pid not in current]
    if chunks:
        log(f"Embedding {len(chunks)} chunks from {len(updated_ids)} pages...")
        vectors = embedder.embed_documents([c.embed_text for c in chunks])
        store.upsert_pages(chunks, vectors)
    # pages that became empty: drop their old chunks
    store.delete_pages([pid for pid in updated_ids if pid not in {c.page_id for c in chunks}] + removed)

    if updated_ids or removed:
        store.rebuild_fts()

    stats.updated, stats.removed = len(updated_ids), len(removed)
    stats.chunks_written, stats.total_chunks = len(chunks), store.count()
    return stats
