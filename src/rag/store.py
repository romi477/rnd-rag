"""LanceDB storage for chunks: one table, atomic upsert by page, native BM25 index on embed_text."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import FTS

from rag import config
from rag.chunking import Chunk

TABLE = "chunks"
FTS_FIELD = "embed_text"


def chunk_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("page_id", pa.string()),
            pa.field("page_version", pa.int64()),
            pa.field("title", pa.string()),
            pa.field("product", pa.string()),
            pa.field("breadcrumbs", pa.list_(pa.string())),
            pa.field("section_path", pa.list_(pa.string())),
            pa.field("url", pa.string()),
            pa.field("updated", pa.string()),
            pa.field("labels", pa.list_(pa.string())),
            pa.field("text", pa.string()),
            pa.field("embed_text", pa.string()),
            pa.field("n_tokens", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def store_path(model: str) -> Path:
    """Each embedding model gets its own index directory, so models can be compared side by side."""
    return config.DATA_DIR / "lancedb" / model.replace("/", "__")


def sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def page_filter(page_ids: list[str]) -> str:
    return f"page_id IN ({', '.join(sql_str(i) for i in page_ids)})"


class Store:
    def __init__(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = lancedb.connect(path)
        self.meta_path = path / "index_meta.json"

    # --- metadata -----------------------------------------------------------------

    @property
    def meta(self) -> dict:
        return json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {}

    def write_meta(self, **values) -> None:
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.meta | values, indent=1))
        tmp.replace(self.meta_path)

    def exists(self) -> bool:
        return TABLE in self.db.list_tables().tables

    @property
    def table(self):
        return self.db.open_table(TABLE)

    def reset(self, model: str, dim: int) -> None:
        """Drop and recreate the table (with an empty BM25 index) for a (new) embedding model."""
        if self.exists():
            self.db.drop_table(TABLE)
        self.db.create_table(TABLE, schema=chunk_schema(dim))
        self.rebuild_fts()
        self.meta_path.unlink(missing_ok=True)
        self.write_meta(model=model, dim=dim)

    def has_fts(self) -> bool:
        return self.exists() and any(
            i.index_type == "FTS" and FTS_FIELD in i.columns for i in self.table.list_indices()
        )

    # --- reads --------------------------------------------------------------------

    def page_versions(self) -> dict[str, int]:
        return {pid: s["page_version"] for pid, s in self.page_states().items()}

    def page_states(self) -> dict[str, dict]:
        """Per indexed page: the version and the metadata that can change without a new version."""
        if not self.exists():
            return {}
        cols = ["page_id", "page_version", "title", "breadcrumbs", "labels"]
        return {r.pop("page_id"): r for r in self.table.to_arrow().select(cols).to_pylist()}

    def count(self) -> int:
        return self.table.count_rows() if self.exists() else 0

    def products(self) -> list[str]:
        if not self.exists():
            return []
        return sorted(set(self.table.to_arrow().column("product").to_pylist()))

    # --- writes -------------------------------------------------------------------

    def delete_pages(self, page_ids: list[str]) -> None:
        if page_ids:
            self.table.delete(page_filter(page_ids))

    def upsert_pages(
        self, chunks: list[Chunk], vectors: np.ndarray, delete: list[str] = (), replace_all: bool = False
    ) -> None:
        """Replace all chunks of the pages in `chunks` and drop the pages in `delete`, in one commit.

        The new rows are built and checked first, so a bad vector or an interruption leaves the
        table as it was; readers never see a page missing. `replace_all` drops every other page.
        """
        table = self.table
        dim = table.schema.field("vector").type.list_size
        vectors = np.asarray(vectors, dtype=np.float32).reshape(len(chunks), -1) if chunks else vectors
        if chunks and (vectors.shape != (len(chunks), dim) or not np.isfinite(vectors).all()):
            raise ValueError(f"Bad vectors: shape {vectors.shape}, expected ({len(chunks)}, {dim}), all finite")
        rows = [asdict(c) | {"vector": v.tolist()} for c, v in zip(chunks, vectors, strict=True)]
        data = pa.Table.from_pylist(rows, schema=table.schema)
        pages = sorted({c.page_id for c in chunks} | set(delete))
        if not pages and not replace_all:
            return
        merge = table.merge_insert("chunk_id").when_matched_update_all().when_not_matched_insert_all()
        merge = merge.when_not_matched_by_source_delete(None if replace_all else page_filter(pages))
        merge.execute(data)

    def rebuild_fts(self) -> None:
        self.table.create_index(
            FTS_FIELD,
            config=FTS(language="English", stem=True, remove_stop_words=True),
            replace=True,
        )
