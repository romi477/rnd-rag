"""Project paths and tunables. Values come from env (.env) with sane defaults."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

RAW_DIR = Path(os.environ.get("RAG_RAW_DIR", ROOT / "raw-data" / "confluence"))
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", ROOT / "data"))
EVAL_FILE = Path(os.environ.get("RAG_EVAL_FILE", ROOT / "eval" / "queries.yaml"))
WEB_DIR = Path(os.environ.get("RAG_WEB_DIR", ROOT / "src" / "web"))  # static UI served by api.py

SPACE_KEY = os.environ.get("CONFLUENCE_SPACE_KEY", "pd")

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# Hybrid search: weight of the BM25 list relative to the vector list in RRF fusion.
HYBRID_FTS_WEIGHT = float(os.environ.get("RAG_HYBRID_FTS_WEIGHT", 0.3))

# Chunking, in approximate tokens (see chunking.approx_tokens).
CHUNK_MAX_TOKENS = int(os.environ.get("RAG_CHUNK_MAX_TOKENS", 500))
CHUNK_MIN_TOKENS = int(os.environ.get("RAG_CHUNK_MIN_TOKENS", 80))
