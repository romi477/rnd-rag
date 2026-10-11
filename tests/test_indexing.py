import json

import pytest

from rag import indexing
from rag.indexing import IndexingError, index_space
from rag.store import Store
from test_store_search import FakeEmbedder


def write_page(space_dir, pid, title, body, version=1, crumbs=("Home", "Printing")):
    pages = space_dir / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    meta = {"id": pid, "title": title, "url": f"https://x/{pid}", "version": version,
            "updated": "2026-01-01", "breadcrumbs": list(crumbs), "labels": []}
    (pages / f"{pid}.json").write_text(json.dumps(meta))
    (pages / f"{pid}.md").write_text(f"---\nid: \"{pid}\"\n---\n# {title}\n\n{body}\n")


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(indexing.config, "RAW_DIR", tmp_path / "raw")
    space = tmp_path / "raw" / "sp"
    for i in range(1, 6):
        write_page(space, str(i), f"Page {i}", f"Printer text number {i}.")
    return space


def run(tmp_path, **kwargs):
    store = Store(tmp_path / "db")
    return store, index_space("sp", embedder=FakeEmbedder(), store=store, log=lambda _: None, **kwargs)


def test_second_run_changes_nothing(raw, tmp_path):
    store, first = run(tmp_path)
    assert first.full_rebuild and first.updated == 5 and store.has_fts()
    assert store.meta["space"] == "sp" and store.meta["chunking"]["version"] >= 1
    _, second = run(tmp_path)
    assert (second.updated, second.removed, second.full_rebuild) == (0, 0, False)


def test_renamed_parent_reindexes_without_new_version(raw, tmp_path):
    run(tmp_path)
    write_page(raw, "2", "Page 2", "Printer text number 2.", crumbs=("Home", "Printing v2"))
    store, stats = run(tmp_path)
    assert stats.updated == 1
    assert store.table.search().where("page_id = '2'").to_list()[0]["product"] == "Printing v2"


def test_wrong_or_empty_raw_dir_does_not_wipe_the_index(raw, tmp_path):
    store, _ = run(tmp_path)
    with pytest.raises(IndexingError, match="No synced pages"):
        index_space("typo", embedder=FakeEmbedder(), store=store, log=lambda _: None)
    for pid in "1234":
        (raw / "pages" / f"{pid}.json").unlink()
    with pytest.raises(IndexingError, match="4 of 5 indexed pages are gone"):
        run(tmp_path)
    assert store.count() == 5
    _, stats = run(tmp_path, force=True)
    assert stats.removed == 0 and store.count() == 1  # force: rebuilt from what is there


def test_another_space_needs_force(raw, tmp_path):
    run(tmp_path)
    write_page(raw.parent / "other", "9", "Other", "Printer text.")
    store = Store(tmp_path / "db")
    with pytest.raises(IndexingError, match="holds space 'sp'"):
        index_space("other", embedder=FakeEmbedder(), store=store, log=lambda _: None)


def test_chunking_change_triggers_rebuild(raw, tmp_path, monkeypatch):
    run(tmp_path)
    monkeypatch.setattr(indexing, "chunk_settings", lambda: {"version": 99, "max_tokens": 1, "min_tokens": 0})
    store, stats = run(tmp_path)
    assert stats.full_rebuild and stats.updated == 5
    assert store.meta["chunking"]["version"] == 99


def test_missing_bm25_index_is_rebuilt(raw, tmp_path):
    store, _ = run(tmp_path)
    store.table.drop_index(store.table.list_indices()[0].name)
    _, stats = run(tmp_path)
    assert stats.updated == 0 and store.has_fts()


def test_removed_and_emptied_pages_lose_their_chunks(raw, tmp_path):
    run(tmp_path)
    (raw / "pages" / "5.json").unlink()
    write_page(raw, "4", "Page 4", "[image: only.png]", version=2)
    store, stats = run(tmp_path)
    assert stats.removed == 1 and set(store.page_versions()) == {"1", "2", "3"}
