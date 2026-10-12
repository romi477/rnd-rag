import numpy as np
import pytest

from rag.chunking import approx_tokens
from rag.code_index import CodeIndexError, index_module, split_unit, unit_row
from rag.code_search import CodeSearcher
from rag.code_store import CodeStore, identifier_words
from rag.sources.odoo_module import CodeUnit

VOCAB = ["order", "sync", "carrier", "client"]


class FakeEmbedder:
    """Bag-of-keywords vectors, as in test_store_search."""

    model_name = "fake"
    dim = len(VOCAB)

    def _vec(self, text: str) -> np.ndarray:
        v = np.array([text.lower().count(w) for w in VOCAB], dtype=np.float32) + 0.01
        return v / np.linalg.norm(v)

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, query):
        return self._vec(query)


def unit(kind="method", module="demo_core", model="sale.order", name="action_sync", text="def action_sync(self):\n    return 1",
         path="models/order.py", start=10, **extra) -> CodeUnit:
    n_lines = len(text.splitlines())
    return CodeUnit(
        kind, module, "19.0.1.0.0", "connector", model, name, path, start, start + n_lines - 1, text,
        f"{module} > {model} > {name}\n{text}", extra,
    )


def long_method(n_lines: int) -> CodeUnit:
    body = "\n".join(f"    value_{i} = compute(order, {i})" for i in range(n_lines))
    return unit(text=f"def big(self, order):\n{body}\n    return order", name="big", start=100)


# --- splitting ---------------------------------------------------------------------------------------------------


def test_small_unit_is_not_split():
    u = unit()
    assert split_unit(u, 500) == [u]


def test_big_unit_is_split_into_parts_that_fit_and_cover_every_line():
    u = long_method(200)
    parts = split_unit(u, 200)
    assert len(parts) > 1
    assert all(approx_tokens(p.embed_text) <= 200 for p in parts)
    assert all(p.embed_text.startswith("demo_core > sale.order > big") for p in parts)
    assert [p.extra["part"] for p in parts] == list(range(1, len(parts) + 1))
    assert all(p.extra["parts"] == len(parts) for p in parts)
    assert parts[0].line_start == u.line_start and parts[-1].line_end == u.line_end
    for a, b in zip(parts, parts[1:]):
        assert b.line_start == a.line_end + 1
    assert "\n".join(p.text for p in parts) == u.text
    assert len({p.id for p in parts}) == len(parts)


def test_split_prefers_a_blank_line():
    lines = [f"    step_{i} = {i}" for i in range(30)]
    lines[14] = ""
    u = unit(text="def f(self):\n" + "\n".join(lines), name="f")
    first = split_unit(u, 120)[0]
    assert first.text.endswith("step_13 = 13")


def test_one_huge_line_is_cut_by_characters():
    u = unit(text="x = '" + "a" * 5000 + "'", name="blob")
    parts = split_unit(u, 200)
    assert len(parts) > 1 and all(approx_tokens(p.embed_text) <= 200 for p in parts)


def test_generated_text_keeps_the_whole_range():
    card = unit(kind="model", name="sale.order", text="\n".join(f"- field_{i} (Char)" for i in range(300)))
    parts = split_unit(card, 200)
    assert len(parts) > 1
    assert {(p.line_start, p.line_end) for p in parts} == {(card.line_start, card.line_end)}


def test_unit_row_carries_extra_as_json_and_part_ids():
    row = unit_row(unit(calls_super=True))
    assert row["row_id"] == row["unit_id"] == "demo_core:models/order.py:10:method"
    assert '"calls_super": true' in row["extra"]


# --- store and search --------------------------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    s = CodeStore(tmp_path / "db")
    s.reset("fake", FakeEmbedder.dim)
    return s


@pytest.fixture
def indexed(store):
    emb = FakeEmbedder()
    index_module(store, emb, "demo_core", [
        unit(text="def action_sync(self):\n    '''Push the order to the client.'''\n    return super().action_sync()"),
        unit(name="get_carrier", model="delivery.carrier", text="def get_carrier(self):\n    return self.carrier_id",
             path="models/carrier.py", start=5),
    ], log=lambda m: None)
    index_module(store, emb, "demo_shop", [
        unit(module="demo_shop", name="get_orders", model="", kind="method",
             text="def get_orders(self, since):\n    return self.client.orders(since)", path="api/client.py", start=7),
    ], log=lambda m: None)
    return store


def searcher(store) -> CodeSearcher:
    return CodeSearcher(FakeEmbedder(), store=store)


def test_index_module_counts_rows_per_module(indexed):
    assert indexed.count() == 3
    assert indexed.modules() == {"demo_core": 2, "demo_shop": 1}


def test_reindexing_a_module_replaces_only_its_rows(indexed):
    index_module(indexed, FakeEmbedder(), "demo_core", [unit(name="only_one")], log=lambda m: None)
    assert indexed.modules() == {"demo_core": 1, "demo_shop": 1}


def test_delete_module(indexed):
    indexed.delete_module("demo_shop")
    assert indexed.modules() == {"demo_core": 2}


def test_vector_search_finds_by_meaning_words(indexed):
    hits = searcher(indexed).search("carrier", mode="vector", k=3)
    assert hits[0].name == "get_carrier"
    assert hits[0].signals["vector"]["rank"] == 1


def test_fts_matches_a_snake_case_identifier_part(indexed):
    hits = searcher(indexed).search("get_orders", mode="fts")
    assert hits and hits[0].name == "get_orders"
    assert hits[0].path == "api/client.py" and hits[0].line_start == 7


def test_hybrid_reports_both_signals_and_rrf(indexed):
    hit = searcher(indexed).search("order sync", mode="hybrid")[0]
    assert {"vector", "fts", "rrf"} <= set(hit.signals)
    assert hit.name == "action_sync"


def test_module_and_kind_filters(indexed):
    s = searcher(indexed)
    assert {h.module for h in s.search("order", modules=["demo_shop"])} == {"demo_shop"}
    assert s.search("order", kinds=["view"]) == []


def test_parts_of_one_unit_come_back_once(store):
    index_module(store, FakeEmbedder(), "demo_core", [long_method(120)], max_tokens=150, log=lambda m: None)
    assert store.count() > 1
    hits = searcher(store).search("compute order", mode="vector", k=10)
    assert len(hits) == 1 and hits[0].name == "big"


def test_unknown_mode_is_a_value_error(indexed):
    with pytest.raises(ValueError):
        searcher(indexed).search("x", mode="nope")


def test_changed_index_version_asks_for_a_rebuild(indexed):
    indexed.write_meta(code_index_version=-1)
    with pytest.raises(CodeIndexError, match="re-add"):
        index_module(indexed, FakeEmbedder(), "demo_core", [unit()], log=lambda m: None)


def test_identifier_words_splits_camel_case_only():
    assert identifier_words("integrationApiExport _ensure_settings ShopifyAPIClient") == "integration api export shopify client"
    assert identifier_words("plain words and snake_case") == ""


def test_fts_matches_camel_case_names_by_their_words_and_the_reverse(store):
    index_module(store, FakeEmbedder(), "demo_core", [
        unit(name="exportInventoryNow", text="def exportInventoryNow(self):\n    return 1", start=1),
        unit(name="export_stock_later", text="def export_stock_later(self):\n    return 2", start=9),
    ], log=lambda m: None)
    s = searcher(store)
    assert s.search("export inventory now", mode="fts")[0].name == "exportInventoryNow"
    assert s.search("exportStockLater", mode="fts")[0].name == "export_stock_later"
