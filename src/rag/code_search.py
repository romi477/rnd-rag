"""Search over the `code_units` table: hybrid (BM25 + vector, weighted RRF), vector-only or BM25-only.

Units are not grouped by page: a hit is one unit (its best part). No rerank yet; the relevance cut-off for code is
a separate calibration (see the stage plan).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from rag import config
from rag.code_store import CodeStore
from rag.embedding import Embedder
from rag.search import FTS_SYNTAX, MODES, RRF_K
from rag.store import sql_str, store_path

log = logging.getLogger(__name__)

COLUMNS = [
    "row_id", "unit_id", "kind", "module", "module_version", "origin", "model", "name", "path", "line_start",
    "line_end", "text", "embed_text", "extra", "part", "parts",
]  # fmt: skip


@dataclass
class CodeHit:
    unit_id: str
    kind: str
    module: str
    module_version: str
    origin: str
    model: str
    name: str
    path: str
    line_start: int
    line_end: int
    text: str
    score: float
    extra: dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)


class CodeSearcher:
    def __init__(self, embedder: Embedder | None = None, store: CodeStore | None = None):
        self.embedder = embedder or Embedder()
        self.store = store or CodeStore(store_path(self.embedder.model_name))
        if not self.store.exists():
            raise RuntimeError(f"No code index at {self.store.path}; add a module with `rag module add` first")

    def search(
        self,
        query: str,
        k: int = 10,
        mode: str = "hybrid",
        modules: list[str] | None = None,
        kinds: list[str] | None = None,
        candidates: int = 40,
        fts_weight: float = config.HYBRID_FTS_WEIGHT,
    ) -> list[CodeHit]:
        if mode not in MODES:
            raise ValueError(f"Unknown search mode {mode!r}, expected one of {MODES}")
        query = query.strip()
        if not query:
            return []
        where = " AND ".join(
            f"{col} IN ({', '.join(sql_str(v) for v in values)})"
            for col, values in (("module", modules), ("kind", kinds))
            if values
        )
        if mode == "hybrid":
            rows = self.hybrid(query, where, candidates, fts_weight)
        else:
            rows = self.retrieve(query, mode, where, candidates)
        hits: dict[str, CodeHit] = {}
        for row in rows:  # sorted by score: the first part of a unit is its best
            if row["unit_id"] in hits:
                continue
            if len(hits) >= k:
                break
            hits[row["unit_id"]] = CodeHit(
                row["unit_id"], row["kind"], row["module"], row["module_version"], row["origin"], row["model"],
                row["name"], row["path"], row["line_start"], row["line_end"], row["text"], row["score"],
                json.loads(row["extra"] or "{}"), row["signals"],
            )
        return list(hits.values())

    def retrieve(self, query: str, mode: str, where: str, limit: int) -> list[dict]:
        table = self.store.table
        if mode == "fts":
            words = FTS_SYNTAX.sub(" ", query).strip()
            if not words:
                return []
            q = table.search(words, query_type="fts")
        else:
            q = table.search(self.embedder.embed_query(query), query_type="vector").distance_type("cosine")
        if where:
            q = q.where(where, prefilter=True)
        score_col = "_score" if mode == "fts" else "_distance"
        rows = q.select(COLUMNS + [score_col]).limit(limit).to_list()
        for rank, row in enumerate(rows, 1):
            row["score"] = row["_score"] if mode == "fts" else 1.0 - row["_distance"]
            row["signals"] = {mode: {"rank": rank, "score": row["score"]}}
        return rows

    def hybrid(self, query: str, where: str, limit: int, fts_weight: float) -> list[dict]:
        """Weighted reciprocal rank fusion: score = 1/(K+rank_vec) + w * 1/(K+rank_fts)."""
        fused: dict[str, dict] = {}
        for mode, weight in (("vector", 1.0), ("fts", fts_weight)):
            if weight <= 0:
                continue
            try:
                rows = self.retrieve(query, mode, where, limit)
            except ValueError as exc:  # e.g. no BM25 index while it is being rebuilt
                if mode != "fts":
                    raise
                log.warning("BM25 search failed, using vector results only: %s", exc)
                continue
            for rank, row in enumerate(rows, 1):
                entry = fused.setdefault(row["row_id"], row | {"score": 0.0, "signals": {}})
                entry["signals"].update(row["signals"])
                entry["score"] += weight / (RRF_K + rank)
        for entry in fused.values():
            entry["signals"]["rrf"] = entry["score"]
        return sorted(fused.values(), key=lambda r: r["score"], reverse=True)[:limit]
