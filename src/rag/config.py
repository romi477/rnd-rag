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
BENCH_FILE = Path(os.environ.get("RAG_BENCH_FILE", ROOT / "eval" / "bench.yaml"))  # questions for `rag bench`
WEB_DIR = Path(os.environ.get("RAG_WEB_DIR", ROOT / "src" / "web"))  # static UI served by api.py
# Host names the API answers to (Host header); add the machine's name when binding beyond localhost.
ALLOWED_HOSTS = os.environ.get("RAG_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")

SPACE_KEY = os.environ.get("CONFLUENCE_SPACE_KEY", "pd")

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# Hybrid search: weight of the BM25 list relative to the vector list in RRF fusion.
HYBRID_FTS_WEIGHT = float(os.environ.get("RAG_HYBRID_FTS_WEIGHT", 0.3))

# Rerank: how many fused chunks the cross-encoder re-scores, and the relevance cut-off (0..1).
# Pages below the cut-off are flagged low_relevance (collapsed in the UI, not sent to the LLM).
RERANK_CANDIDATES = int(os.environ.get("RAG_RERANK_CANDIDATES", 10))
RERANK_MIN_SCORE = float(os.environ.get("RAG_RERANK_MIN_SCORE", 0.1))

# Chunking, in approximate tokens (see chunking.approx_tokens).
CHUNK_MAX_TOKENS = int(os.environ.get("RAG_CHUNK_MAX_TOKENS", 500))
CHUNK_MIN_TOKENS = int(os.environ.get("RAG_CHUNK_MIN_TOKENS", 80))

# Answer generation: any OpenAI-compatible server (LM Studio by default).
LLM_BASE_URL = os.environ.get("RAG_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "")  # empty: the LLM currently loaded in LM Studio
# Sent as `reasoning_effort`; "none" turns off thinking for reasoning models. Empty: not sent.
LLM_REASONING = os.environ.get("RAG_LLM_REASONING", "none")
LLM_TEMPERATURE = float(os.environ.get("RAG_LLM_TEMPERATURE", 0.2))
LLM_MAX_TOKENS = int(os.environ.get("RAG_LLM_MAX_TOKENS", 1000))
# How much of the search result goes into the prompt.
LLM_CONTEXT_PAGES = int(os.environ.get("RAG_LLM_CONTEXT_PAGES", 5))
LLM_CONTEXT_TOKENS = int(os.environ.get("RAG_LLM_CONTEXT_TOKENS", 4000))
