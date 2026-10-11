"""Incremental indexing: synced pages -> chunks -> embeddings -> LanceDB."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rag import config
from rag.chunking import Page, chunk_page, chunk_settings, load_pages
from rag.embedding import Embedder
from rag.store import Store, store_path

# Refuse to drop more indexed pages than this in one run, unless `force`: a wrong space, an empty
# or half-synced raw-data/ must not silently empty the index.
MAX_REMOVED_SHARE = 0.25


class IndexingError(RuntimeError):
    pass


@dataclass
class IndexStats:
    pages: int = 0
    updated: int = 0
    removed: int = 0
    chunks_written: int = 0
    total_chunks: int = 0
    full_rebuild: bool = False


def is_current(page: Page, state: dict | None) -> bool:
    """Indexed at this version, with the same metadata (a renamed parent changes the breadcrumbs)."""
    return state is not None and (state["page_version"], state["title"], state["breadcrumbs"], state["labels"]) == (
        page.version, page.title, page.breadcrumbs, page.labels
    )


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

    space_dir = config.RAW_DIR / space
    pages = load_pages(space_dir)
    if not pages:
        raise IndexingError(f"No synced pages in {space_dir}: check the space key and RAG_RAW_DIR, or run the sync")
    meta = store.meta
    if store.exists() and meta.get("space", space) != space and not force:
        raise IndexingError(
            f"The index at {store.path} holds space '{meta['space']}', not '{space}'; use --force to replace it"
        )

    settings = chunk_settings()
    if not store.exists() or meta.get("model") != embedder.model_name:
        log(f"New index with {embedder.model_name}")
        store.reset(embedder.model_name, embedder.dim)
        stats.full_rebuild = True
    elif force or meta.get("chunking") != settings:
        # Rebuilt in place: the new rows replace the old ones in one commit, so a running server
        # keeps answering from the old index until then.
        log("Full rebuild" + ("" if force else f": chunking changed ({meta.get('chunking')} -> {settings})"))
        stats.full_rebuild = True
    indexed = {} if stats.full_rebuild else store.page_states()
    current = {p.id for p in pages}
    stats.pages = len(pages)

    chunks, updated_ids = [], []
    for page in pages:
        if is_current(page, indexed.get(page.id)):
            continue
        page_chunks = chunk_page(page)
        if not page_chunks and page.id not in indexed:
            continue  # empty page that was never indexed
        chunks.extend(page_chunks)
        updated_ids.append(page.id)

    removed = [pid for pid in indexed if pid not in current]
    if len(removed) > MAX_REMOVED_SHARE * len(indexed) and not force:
        raise IndexingError(
            f"{len(removed)} of {len(indexed)} indexed pages are gone from {space_dir}; "
            "refusing to drop them (half-synced raw-data/?), use --force to accept"
        )

    vectors = []
    if chunks:
        log(f"Embedding {len(chunks)} chunks from {len(updated_ids)} pages...")
        too_long = embedder.count_truncated([c.embed_text for c in chunks]) if hasattr(embedder, "count_truncated") else 0
        if too_long:
            log(f"Warning: {too_long} chunks are longer than the model's input and get truncated")
        vectors = embedder.embed_documents([c.embed_text for c in chunks])
    # Pages that became empty or disappeared lose their chunks in the same commit.
    store.upsert_pages(chunks, vectors, delete=updated_ids + removed, replace_all=stats.full_rebuild)
    if updated_ids or removed or stats.full_rebuild or not store.has_fts():
        store.rebuild_fts()
    store.write_meta(space=space, chunking=settings)

    stats.updated, stats.removed = len(updated_ids), len(removed)
    stats.chunks_written, stats.total_chunks = len(chunks), store.count()
    return stats
