import numpy as np
import pytest

from rag.chunking import Page, chunk_page
from rag.search import RRF_K, Searcher, group_by_page
from rag.store import Store, sql_str

VOCAB = ["printer", "label", "shopify", "invoice"]


class FakeEmbedder:
    """Bag-of-keywords vectors: enough to make vector search deterministic."""

    model_name = "fake"
    dim = len(VOCAB)

    def _vec(self, text: str) -> np.ndarray:
        v = np.array([text.lower().count(w) for w in VOCAB], dtype=np.float32) + 0.01
        return v / np.linalg.norm(v)

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, query):
        return self._vec(query)


def page(pid: str, title: str, body: str, product: str, version: int = 1) -> Page:
    return Page(pid, title, version, f"https://x/{pid}", "2026-01-01", ["Home", product], [], body)


def vectors(chunks):
    return FakeEmbedder().embed_documents([c.embed_text for c in chunks])


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "db")
    s.reset("fake", FakeEmbedder.dim)
    emb = FakeEmbedder()
    pages = [
        page("1", "Printer setup", "Install the printer client. Printer printer.", "Direct Print"),
        page("2", "Label sizes", "Configure label size for the Zebra label printer.", "ZPL Label Designer"),
        page("3", "Shopify orders", "Import Shopify orders and invoice them. Error 403 Forbidden.", "E-commerce"),
    ]
    chunks = [c for p in pages for c in chunk_page(p, max_tokens=500, min_tokens=80)]
    s.upsert_pages(chunks, emb.embed_documents([c.embed_text for c in chunks]))
    s.rebuild_fts()
    return s


def test_upsert_replaces_page_chunks_and_versions(store):
    assert store.page_versions() == {"1": 1, "2": 1, "3": 1}
    updated = chunk_page(page("1", "Printer setup", "New text about printer.", "Direct Print", 2))
    store.upsert_pages(updated, FakeEmbedder().embed_documents([c.embed_text for c in updated]))
    assert store.page_versions()["1"] == 2
    assert store.count() == 3


def test_delete_pages(store):
    store.delete_pages(["2"])
    assert set(store.page_versions()) == {"1", "3"}


def test_products(store):
    assert store.products() == ["Direct Print", "E-commerce", "ZPL Label Designer"]


def test_pages_lists_each_page_once(store):
    long = page("4", "Long printer guide", "\n\n".join(f"## Part {i}\n\n" + "printer " * 300 for i in range(3)), "Direct Print")
    chunks = chunk_page(long)
    assert len(chunks) > 1
    store.upsert_pages(chunks, vectors(chunks))
    pages = store.pages()
    assert [p["page_id"] for p in pages] == ["4", "1", "3", "2"]  # by product, then title
    assert pages[1] == {"page_id": "1", "title": "Printer setup", "url": "https://x/1", "product": "Direct Print",
                        "breadcrumbs": ["Direct Print"], "updated": "2026-01-01", "labels": []}


@pytest.mark.parametrize("mode", ["fts", "vector", "hybrid"])
def test_search_modes_find_expected_page(store, mode):
    searcher = Searcher(FakeEmbedder(), store=store)
    hits = searcher.search("shopify invoice", k=3, mode=mode)
    assert hits[0].page_id == "3"


def test_keyword_only_term_found_by_hybrid(store):
    # "403" is invisible to the fake embedder; BM25 must bring it in.
    searcher = Searcher(FakeEmbedder(), store=store)
    hits = searcher.search("403 forbidden", k=1, mode="hybrid", fts_weight=1.0)
    assert hits[0].page_id == "3"


def test_product_filter(store):
    searcher = Searcher(FakeEmbedder(), store=store)
    hits = searcher.search("printer", k=5, mode="hybrid", product="ZPL Label Designer")
    assert [h.page_id for h in hits] == ["2"]


class FakeReranker:
    """Scores a chunk by the share of query words it contains (0..1)."""

    def score(self, query, passages):
        words = query.lower().split()
        return [sum(w in p.lower() for w in words) / len(words) for p in passages]


def test_rerank_flags_pages_below_cutoff(store):
    searcher = Searcher(FakeEmbedder(), reranker=FakeReranker(), store=store)
    hits = searcher.search("shopify invoice orders", k=3, rerank=True, min_score=0.5, rerank_candidates=10)
    assert hits[0].page_id == "3" and not hits[0].low_relevance
    assert all(h.low_relevance for h in hits[1:])
    # without rerank nothing is flagged: hybrid scores carry no absolute meaning
    assert not any(h.low_relevance for h in searcher.search("shopify invoice orders", k=3))


def test_rerank_candidates_limit(store):
    searcher = Searcher(FakeEmbedder(), reranker=FakeReranker(), store=store)
    assert len(searcher.search("printer label shopify", k=10, rerank=True, rerank_candidates=1)) == 1


def test_signals_explain_the_score(store):
    searcher = Searcher(FakeEmbedder(), reranker=FakeReranker(), store=store)
    best = searcher.search("shopify invoice", k=3, mode="hybrid", rerank=True, rerank_candidates=10)[0].chunks[0]
    sig = best.signals
    assert set(sig) == {"vector", "fts", "rrf", "retrieval_rank", "rerank"}
    assert sig["vector"]["rank"] == 1 and sig["fts"]["rank"] == 1 and sig["retrieval_rank"] == 1
    assert sig["rerank"] == best.score
    assert set(searcher.search("shopify", k=1, mode="fts")[0].chunks[0].signals) == {"fts", "retrieval_rank"}


def test_quoted_query_does_not_break_bm25(store):
    searcher = Searcher(FakeEmbedder(), store=store)
    for mode in ("fts", "hybrid"):
        assert searcher.search('"shopify invoice"', k=3, mode=mode)[0].page_id == "3"
    assert searcher.search('""', mode="fts") == []


def test_unknown_mode_is_an_error(store):
    with pytest.raises(ValueError, match="Unknown search mode"):
        Searcher(FakeEmbedder(), store=store).search("printer", mode="fulltext")


def test_hybrid_falls_back_to_vector_without_bm25_index(store):
    store.table.drop_index(store.table.list_indices()[0].name)
    assert not store.has_fts()
    searcher = Searcher(FakeEmbedder(), store=store)
    assert searcher.search("shopify invoice", k=3, mode="hybrid")[0].page_id == "3"
    with pytest.raises(ValueError):
        searcher.search("shopify", mode="fts")


def test_upsert_with_bad_vectors_changes_nothing(store):
    before = store.page_versions()
    new = chunk_page(page("1", "Printer setup", "Changed printer text.", "Direct Print", 2))
    bad = vectors(new)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="Bad vectors"):
        store.upsert_pages(new, bad)
    with pytest.raises(ValueError, match="Bad vectors"):
        store.upsert_pages(new, vectors(new)[:, :2])
    assert store.page_versions() == before and store.count() == 3


def test_upsert_replaces_and_deletes_in_one_commit(store):
    version = store.table.version
    new = chunk_page(page("1", "Printer setup", "Changed printer text.", "Direct Print", 2))
    store.upsert_pages(new, vectors(new), delete=["2"])
    assert store.table.version == version + 1
    assert store.page_versions() == {"1": 2, "3": 1}
    store.upsert_pages([], [], replace_all=True)
    assert store.count() == 0


def test_reset_creates_bm25_index(tmp_path):
    s = Store(tmp_path / "db")
    s.reset("fake", FakeEmbedder.dim)
    assert s.has_fts() and s.meta == {"model": "fake", "dim": FakeEmbedder.dim}


def test_hybrid_score_is_weighted_rrf(store):
    searcher = Searcher(FakeEmbedder(), store=store)
    top = searcher.search("shopify invoice", k=1, mode="hybrid", fts_weight=0.5)[0]
    sig = top.chunks[0].signals
    assert sig["vector"]["rank"] == sig["fts"]["rank"] == 1
    assert top.score == pytest.approx(1 / (RRF_K + 1) + 0.5 / (RRF_K + 1))
    assert sig["vector"]["score"] > 0.9  # cosine similarity, not distance


def test_group_by_page_limits():
    rows = [{"page_id": p, "chunk_id": f"{p}:{i}", "title": p, "url": "", "product": "", "breadcrumbs": [],
             "updated": "", "labels": [], "section_path": [], "text": "", "score": s, "signals": {}}
            for p, i, s in [("a", 0, 0.9), ("b", 0, 0.8), ("a", 1, 0.7), ("a", 2, 0.6), ("c", 0, 0.5)]]
    pages = group_by_page(rows, k=2, chunks_per_page=2)
    assert [p.page_id for p in pages] == ["a", "b"]
    assert [c.chunk_id for c in pages[0].chunks] == ["a:0", "a:1"] and pages[0].score == 0.9


def test_sql_str_escapes_quotes(store):
    assert sql_str("O'Brien") == "'O''Brien'"
    searcher = Searcher(FakeEmbedder(), store=store)
    assert searcher.search("printer", product="x' OR '1'='1") == []


def test_evaluate_hit_at_k_and_mrr(store):
    from rag.evaluation import evaluate

    queries = [{"q": "shopify invoice", "relevant": ["3"]}, {"q": "zebra label", "relevant": ["9"]}]
    res = evaluate(Searcher(FakeEmbedder(), store=store), queries, mode="hybrid")
    assert res.n == 2 and res.hit(1) == 0.5 and res.mrr == 0.5
    assert res.misses == [("zebra label", None)]
