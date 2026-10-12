"""LanceDB table `code_units`: Odoo addon knowledge units, next to the Confluence `chunks` table.

Same directory as the chunks of the embedding model, own metadata file (`code_index_meta.json`). Rows are replaced a
whole module at a time, in one commit, so a running search never sees a module half-written.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import FTS

from rag.store import sql_str

log = logging.getLogger(__name__)

TABLE = "code_units"
FTS_FIELD = "fts_text"  # embed_text plus the words of camelCase names; the vectors come from embed_text alone
META_FILE = "code_index_meta.json"


IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def identifier_words(text: str) -> str:
    """Lowercase words of the camelCase identifiers in `text`, in order, once each.

    The BM25 tokenizer already splits snake_case and dotted names; it does not split camelCase, so
    `ShopifyAPIClient` would never match "shopify api client". snake_case identifiers add nothing here.
    """
    words: dict[str, None] = {}
    for ident in IDENTIFIER.findall(text):
        parts = CAMEL_BOUNDARY.split(ident)
        if len(parts) > 1:
            words.update(dict.fromkeys(w.lower() for p in parts for w in p.split("_") if w))
    return " ".join(words)


def code_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("row_id", pa.string()),  # unit id plus the part number of a split unit
            pa.field("unit_id", pa.string()),
            pa.field("kind", pa.string()),
            pa.field("module", pa.string()),
            pa.field("module_version", pa.string()),
            pa.field("origin", pa.string()),
            pa.field("model", pa.string()),
            pa.field("name", pa.string()),
            pa.field("path", pa.string()),
            pa.field("line_start", pa.int32()),
            pa.field("line_end", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("embed_text", pa.string()),
            pa.field("fts_text", pa.string()),
            pa.field("extra", pa.string()),  # JSON: calls_super, adapter, depends...
            pa.field("part", pa.int32()),
            pa.field("parts", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


class CodeStore:
    def __init__(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = lancedb.connect(path)
        self.meta_path = path / META_FILE

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
        """Drop and recreate the table (with an empty BM25 index) for an embedding model."""
        if self.exists():
            self.db.drop_table(TABLE)
        self.db.create_table(TABLE, schema=code_schema(dim))
        self.rebuild_fts()
        self.meta_path.unlink(missing_ok=True)
        self.write_meta(model=model, dim=dim)

    def has_fts(self) -> bool:
        return self.exists() and any(
            i.index_type == "FTS" and FTS_FIELD in i.columns for i in self.table.list_indices()
        )

    def rebuild_fts(self) -> None:
        self.table.create_index(
            FTS_FIELD, config=FTS(language="English", stem=True, remove_stop_words=True), replace=True
        )

    def count(self) -> int:
        return self.table.count_rows() if self.exists() else 0

    def modules(self) -> dict[str, int]:
        """Module name -> number of rows (only the module column is read, not the vectors)."""
        if not self.exists():
            return {}
        table = self.table
        if not (n := table.count_rows()):
            return {}
        names = table.search().select(["module"]).limit(n).to_arrow().column("module").to_pylist()
        return dict(sorted(Counter(names).items()))

    def indexed_modules(self) -> dict[str, dict]:
        """Module name -> {"version", "ref"} as they were when its rows were written (`code_index_meta.json`)."""
        return self.meta.get("modules", {})

    def set_indexed_module(self, module: str, version: str, ref: str) -> None:
        self.write_meta(modules=self.indexed_modules() | {module: {"version": version, "ref": ref}})

    def replace_module(self, module: str, rows: list[dict], vectors: np.ndarray) -> None:
        """Replace every row of `module` with `rows`, in one commit, after checking the vectors."""
        table = self.table
        dim = table.schema.field("vector").type.list_size
        if not rows:
            self.delete_module(module)
            return
        vectors = np.asarray(vectors, dtype=np.float32).reshape(len(rows), -1)
        if vectors.shape != (len(rows), dim) or not np.isfinite(vectors).all():
            raise ValueError(f"Bad vectors: shape {vectors.shape}, expected ({len(rows)}, {dim}), all finite")
        data = pa.Table.from_pylist(
            [r | {"vector": v.tolist()} for r, v in zip(rows, vectors, strict=True)], schema=table.schema
        )
        merge = table.merge_insert("row_id").when_matched_update_all().when_not_matched_insert_all()
        merge.when_not_matched_by_source_delete(f"module = {sql_str(module)}").execute(data)

    def rows_where(self, where: str, columns: list[str], limit: int = 1000) -> list[dict]:
        """Rows matching a SQL filter, no ranking: exact lookups by module, path, line or text. A result cut off at
        `limit` is logged: the caller would otherwise not know rows are missing."""
        if not self.exists():
            return []
        rows = self.table.search().where(where).select(columns).limit(limit).to_list()
        if len(rows) >= limit:
            log.warning("rows_where: stopped at %d rows for %s", limit, where[:120])
        return rows

    def delete_module(self, module: str) -> None:
        if not self.exists():
            return
        self.table.delete(f"module = {sql_str(module)}")
        self.write_meta(modules={k: v for k, v in self.indexed_modules().items() if k != module})
