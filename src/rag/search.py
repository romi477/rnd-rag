"""Retrieval: hybrid (BM25 + vector, weighted RRF), vector-only or BM25-only; optional
cross-encoder rerank; results grouped by page."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rag import config
from rag.embedding import Embedder, Reranker
from rag.store import Store, sql_str, store_path

Mode = Literal["hybrid", "vector", "fts"]
MODES: tuple[Mode, ...] = ("hybrid", "vector", "fts")

RRF_K = 60

COLUMNS = [
    "chunk_id", "page_id", "title", "product", "breadcrumbs", "section_path",
    "url", "updated", "labels", "text", "embed_text",
]  # fmt: skip


@dataclass
class ChunkHit:
    chunk_id: str
    section_path: list[str]
    text: str
    score: float
    # How the chunk was scored, for debugging: {"vector"|"fts": {"rank", "score"}, "rrf",
    # "retrieval_rank" (position before rerank), "rerank"}; only the keys that apply.
    signals: dict = field(default_factory=dict)


@dataclass
class PageHit:
    page_id: str
    title: str
    url: str
    product: str
    breadcrumbs: list[str]
    updated: str
    labels: list[str]
    score: float
    chunks: list[ChunkHit] = field(default_factory=list)
    # Set only after rerank: the best chunk scored below the relevance cut-off.
    low_relevance: bool = False


class Searcher:
    def __init__(
        self,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        store: Store | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.store = store or Store(store_path(self.embedder.model_name))
        self._reranker = reranker
        if not self.store.exists():
            raise RuntimeError(f"No index at {self.store.path}; run `rag index` first")

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def products(self) -> list[str]:
        return self.store.products()

    def search(
        self,
        query: str,
        k: int = 10,
        mode: Mode = "hybrid",
        product: str | None = None,
        rerank: bool = False,
        candidates: int = 40,
        chunks_per_page: int = 3,
        fts_weight: float = config.HYBRID_FTS_WEIGHT,
        rerank_candidates: int = config.RERANK_CANDIDATES,
        min_score: float = config.RERANK_MIN_SCORE,
    ) -> list[PageHit]:
        """Pages ranked by their best chunk.

        With `rerank`, the top `rerank_candidates` chunks are re-scored by the cross-encoder
        (0..1) and pages whose best chunk is below `min_score` are flagged `low_relevance`.
        """
        query = query.strip()
        if not query:
            return []
        if mode == "hybrid":
            rows = self.hybrid(query, product, candidates, fts_weight)
        else:
            rows = self.retrieve(query, mode, product, candidates)
        for i, row in enumerate(rows, 1):
            row["signals"]["retrieval_rank"] = i
        if rerank and rows:
            rows = rows[:rerank_candidates]
            scores = self.reranker.score(query, [r["embed_text"] for r in rows])
            for row, score in zip(rows, scores, strict=True):
                row["score"] = row["signals"]["rerank"] = score
            rows.sort(key=lambda r: r["score"], reverse=True)
        pages = group_by_page(rows, k, chunks_per_page)
        if rerank:
            for page in pages:
                page.low_relevance = page.score < min_score
        return pages

    def retrieve(
        self, query: str, mode: Literal["vector", "fts"], product: str | None, limit: int
    ) -> list[dict]:
        table = self.store.table
        if mode == "fts":
            q = table.search(query, query_type="fts")
        else:
            vector = self.embedder.embed_query(query)
            q = table.search(vector, query_type="vector").distance_type("cosine")
        if product:
            q = q.where(f"product = {sql_str(product)}", prefilter=True)
        score_col = "_score" if mode == "fts" else "_distance"
        rows = q.select(COLUMNS + [score_col]).limit(limit).to_list()
        for rank, row in enumerate(rows, 1):
            row["score"] = row["_score"] if mode == "fts" else 1.0 - row["_distance"]
            row["signals"] = {mode: {"rank": rank, "score": row["score"]}}
        return rows

    def hybrid(
        self, query: str, product: str | None, limit: int, fts_weight: float
    ) -> list[dict]:
        """Weighted reciprocal rank fusion: score = 1/(K+rank_vec) + w * 1/(K+rank_fts)."""
        fused: dict[str, dict] = {}
        for mode, weight in (("vector", 1.0), ("fts", fts_weight)):
            if weight <= 0:
                continue
            for rank, row in enumerate(self.retrieve(query, mode, product, limit), 1):
                entry = fused.setdefault(row["chunk_id"], row | {"score": 0.0, "signals": {}})
                entry["signals"].update(row["signals"])
                entry["score"] += weight / (RRF_K + rank)
        for entry in fused.values():
            entry["signals"]["rrf"] = entry["score"]
        return sorted(fused.values(), key=lambda r: r["score"], reverse=True)[:limit]


def group_by_page(rows: list[dict], k: int, chunks_per_page: int) -> list[PageHit]:
    """Rows must be sorted by score desc; a page's score is its best chunk's score."""
    pages: dict[str, PageHit] = {}
    for row in rows:
        page = pages.get(row["page_id"])
        if page is None:
            if len(pages) >= k:
                continue
            page = pages[row["page_id"]] = PageHit(
                page_id=row["page_id"],
                title=row["title"],
                url=row["url"],
                product=row["product"],
                breadcrumbs=row["breadcrumbs"],
                updated=row["updated"],
                labels=row["labels"],
                score=row["score"],
            )
        if len(page.chunks) < chunks_per_page:
            page.chunks.append(
                ChunkHit(row["chunk_id"], row["section_path"], row["text"], row["score"], row["signals"])
            )
    return list(pages.values())
