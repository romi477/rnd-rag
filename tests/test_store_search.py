import numpy as np
import pytest

from rag.chunking import Page, chunk_page
from rag.search import Searcher
from rag.store import Store

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
    chunks = [c for p in pages for c in chunk_page(p)]
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
