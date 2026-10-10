"""Page-level retrieval metrics over the golden set (eval/queries.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag.search import Mode, Searcher


@dataclass
class EvalResult:
    name: str
    n: int = 0
    hits: dict[int, int] = field(default_factory=lambda: {1: 0, 3: 0, 5: 0, 10: 0})
    rr_sum: float = 0.0
    misses: list[tuple[str, int | None]] = field(default_factory=list)  # (query, rank or None)

    def hit(self, k: int) -> float:
        return self.hits[k] / self.n if self.n else 0.0

    @property
    def mrr(self) -> float:
        return self.rr_sum / self.n if self.n else 0.0


def load_queries(path: Path) -> list[dict]:
    items = yaml.safe_load(path.read_text()) or []
    return [{"q": i["q"], "relevant": [str(r) for r in i["relevant"]]} for i in items]


def evaluate(
    searcher: Searcher,
    queries: list[dict],
    mode: Mode = "hybrid",
    rerank: bool = False,
    name: str | None = None,
    **search_kwargs,
) -> EvalResult:
    result = EvalResult(name or (f"{mode}+rerank" if rerank else mode))
    for item in queries:
        pages = searcher.search(item["q"], k=10, mode=mode, rerank=rerank, **search_kwargs)
        ids = [p.page_id for p in pages]
        rank = next((i + 1 for i, pid in enumerate(ids) if pid in item["relevant"]), None)
        result.n += 1
        if rank:
            result.rr_sum += 1 / rank
            for k in result.hits:
                result.hits[k] += rank <= k
        if rank != 1:
            result.misses.append((item["q"], rank))
    return result
