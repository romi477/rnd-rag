"""LanceDB storage for chunks: one table, upsert by page, native BM25 index on embed_text."""

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

    def exists(self) -> bool:
        return TABLE in self.db.list_tables().tables

    @property
    def table(self):
        return self.db.open_table(TABLE)

    def reset(self, model: str, dim: int) -> None:
        """Drop and recreate the table for a (new) embedding model."""
        if self.exists():
            self.db.drop_table(TABLE)
        self.db.create_table(TABLE, schema=chunk_schema(dim))
        self.meta_path.write_text(json.dumps({"model": model, "dim": dim}, indent=1))

    # --- reads --------------------------------------------------------------------

    def page_versions(self) -> dict[str, int]:
        if not self.exists():
            return {}
        rows = self.table.to_arrow().select(["page_id", "page_version"]).to_pylist()
        return {r["page_id"]: r["page_version"] for r in rows}

    def count(self) -> int:
        return self.table.count_rows() if self.exists() else 0

    def products(self) -> list[str]:
        if not self.exists():
            return []
        return sorted(set(self.table.to_arrow().column("product").to_pylist()))

    # --- writes -------------------------------------------------------------------

    def delete_pages(self, page_ids: list[str]) -> None:
        if page_ids:
            ids = ", ".join(sql_str(i) for i in page_ids)
            self.table.delete(f"page_id IN ({ids})")

    def upsert_pages(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Replace all chunks of the pages present in `chunks`."""
        self.delete_pages(sorted({c.page_id for c in chunks}))
        if not chunks:
            return
        rows = [asdict(c) | {"vector": v.tolist()} for c, v in zip(chunks, vectors, strict=True)]
        self.table.add(pa.Table.from_pylist(rows, schema=self.table.schema))

    def rebuild_fts(self) -> None:
        self.table.create_index(
            FTS_FIELD,
            config=FTS(language="English", stem=True, remove_stop_words=True),
            replace=True,
        )
