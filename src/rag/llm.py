"""Minimal client for an OpenAI-compatible chat server (LM Studio, Ollama, vLLM...).

Only what answer generation needs: pick a model, stream a chat completion.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator

import httpx

from rag import config

EMBEDDING_HINTS = ("embed", "rerank")


class LLMError(RuntimeError):
    pass


def split_lines(chunks: Iterable[str]) -> Iterator[str]:
    """Split a text stream on "\n" only (SSE lines), unlike str.splitlines(), which also breaks on
    U+2028 and friends inside the JSON of a delta."""
    buf = ""
    for chunk in chunks:
        buf += chunk
        *lines, buf = buf.split("\n")
        for line in lines:
            yield line.removesuffix("\r")
    if buf:
        yield buf


def parse_sse(lines: Iterable[str], info: dict | None = None) -> Iterator[str]:
    """Yield answer text from an OpenAI-style SSE stream.

    `reasoning_content` deltas (a model "thinking" before answering) are skipped.
    If `info` is given, it gets `finish_reason` ("stop", or "length" when cut off by max_tokens),
    `usage` (token counts, when the server reports them) and `first_delta_at` (perf_counter time of
    the first generated token, reasoning included: generation speed is measured from there).
    A stream that ends without [DONE] or a finish reason was cut off: LLMError.
    """
    info = {} if info is None else info
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue
        if chunk.get("error"):
            raise LLMError(str(chunk["error"]))
        if chunk.get("usage"):
            info["usage"] = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                info["finish_reason"] = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                info.setdefault("first_delta_at", time.perf_counter())
            if delta.get("content"):
                yield delta["content"]
    if not info.get("finish_reason"):
        raise LLMError("The LLM server closed the stream before the answer was complete")


class LLMClient:
    def __init__(
        self,
        base_url: str = config.LLM_BASE_URL,
        model: str = config.LLM_MODEL,
        reasoning: str = config.LLM_REASONING,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.reasoning = reasoning
        # trust_env=False: the prompts carry internal docs, never send them via a system proxy
        self.http = httpx.Client(timeout=httpx.Timeout(timeout, connect=3.0), trust_env=False)

    def list_models(self) -> list[dict]:
        """Chat models on the server as {"id", "loaded"}; `loaded` is None when the server can't tell.

        LM Studio's /api/v0/models reports load state; other servers only have /v1/models.
        LM Studio loads a not-loaded model on its first request (just-in-time), which takes a while.
        """
        root = self.base_url.removesuffix("/v1")
        try:
            resp = self.http.get(f"{root}/api/v0/models")
            if resp.status_code == 200:
                return [
                    {"id": m["id"], "loaded": m.get("state") == "loaded"}
                    for m in resp.json().get("data", [])
                    if m.get("type") in ("llm", "vlm")
                ]
            resp = self.http.get(f"{self.base_url}/models")
            resp.raise_for_status()
            return [
                {"id": m["id"], "loaded": None}
                for m in resp.json().get("data", [])
                if not any(h in m["id"].lower() for h in EMBEDDING_HINTS)
            ]
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM server not reachable at {self.base_url}: {exc}") from exc
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise LLMError(f"Unexpected model list from {self.base_url}: {exc!r}") from exc

    def resolve_model(self, models: list[dict] | None = None) -> str:
        """Configured model, else the first loaded chat model, else the first listed one."""
        if self.model:
            return self.model
        models = self.list_models() if models is None else models
        for m in models:
            if m["loaded"]:
                return m["id"]
        if models:
            return models[0]["id"]
        raise LLMError("No chat model available on the LLM server")

    def status(self) -> dict:
        try:
            models = self.list_models()
            return {"available": True, "model": self.resolve_model(models), "models": models, "error": None}
        except LLMError as exc:
            return {"available": False, "model": None, "models": [], "error": str(exc)}

    def stream_chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS,
        info: dict | None = None,
    ) -> Iterator[str]:
        """Stream answer text. Pass a dict as `info` to get `finish_reason` and `usage` filled in."""
        payload = {
            "model": model or self.resolve_model(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if self.reasoning:
            payload["reasoning_effort"] = self.reasoning
        if info is not None:
            payload["stream_options"] = {"include_usage": True}
        try:
            with self.http.stream("POST", f"{self.base_url}/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise LLMError(f"LLM server error {resp.status_code}: {resp.text[:300]}")
                yield from parse_sse(split_lines(resp.iter_text()), info)
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
