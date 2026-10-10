"""Local embedding and reranking models (sentence-transformers on Apple MPS / CPU)."""

from __future__ import annotations

from functools import cached_property

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag import config

# Query-side instructions for models that expect one but don't ship it in `model.prompts`.
QUERY_PROMPTS = {
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}
MAX_SEQ_LENGTH = 1024  # chunks are <= ~600 tokens; caps memory for long-context models


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Embedder:
    def __init__(self, model_name: str = config.EMBED_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device or default_device()

    @cached_property
    def model(self) -> SentenceTransformer:
        kwargs = {}
        if "qwen3-embedding" in self.model_name.lower():
            kwargs["processor_kwargs"] = {"padding_side": "left"}
        model = SentenceTransformer(self.model_name, device=self.device, **kwargs)
        model.max_seq_length = min(model.max_seq_length or MAX_SEQ_LENGTH, MAX_SEQ_LENGTH)
        return model

    @property
    def dim(self) -> int:
        return self.model.get_embedding_dimension()

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > batch_size,
        ).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        kwargs = {}
        if "query" in (self.model.prompts or {}):
            kwargs["prompt_name"] = "query"
        elif self.model_name in QUERY_PROMPTS:
            kwargs["prompt"] = QUERY_PROMPTS[self.model_name]
        return self.model.encode(
            query, normalize_embeddings=True, convert_to_numpy=True, **kwargs
        ).astype(np.float32)


class Reranker:
    def __init__(self, model_name: str = config.RERANK_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device or default_device()

    @cached_property
    def model(self) -> CrossEncoder:
        return CrossEncoder(self.model_name, device=self.device, max_length=MAX_SEQ_LENGTH)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict([(query, p) for p in passages], batch_size=16)
        return [float(s) for s in scores]
