"""Minimal LLM clients: an OpenAI-compatible chat server (LM Studio, Ollama, vLLM...) or the Claude API.

Only what answer generation needs: list and pick a model, stream a chat completion. Both clients have
the same interface; `make_llm()` picks one by `RAG_LLM_PROVIDER`.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Iterator
from functools import cached_property

import anthropic
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

    def describe(self) -> str:
        return (f"OpenAI-compatible server {self.base_url} · temperature {config.LLM_TEMPERATURE} · "
                f"reasoning {self.reasoning or 'default'}")

    def list_models(self, fresh: bool = False) -> list[dict]:
        """Chat models on the server as {"id", "loaded"}; `loaded` is None when the server can't tell.
        Always asked fresh (`fresh` is for the shared interface): the load state changes.

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


# --- Claude API ------------------------------------------------------------------------

CLAUDE_DEFAULT_MODEL = "claude-sonnet-5-5"
# Thinking and fallback options differ per model (see the Claude API docs); ids match their dated
# variants too (`same_model`). The lowest thinking setting where it exists ("disabled" is a 400 there):
BETWEEN_TOOLS_MODELS = ("claude-sonnet-5-5",)
# Models that still take thinking "disabled" (at the default effort):
NO_THINKING_MODELS = ("claude-haiku-5-5",)
# Models that take the server-side refusal fallback (a declined answer is retried on another model):
FALLBACK_MODELS = ("claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Thinking counts toward max_tokens: room for it on top of the answer when adaptive thinking may run.
THINKING_HEADROOM = 4000
MODELS_TTL = 300.0  # seconds the model list is reused by `list_models()` (status() always refreshes)
STOP_TO_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}


def same_model(model: str, ids: tuple[str, ...]) -> bool:
    return any(model == i or model.startswith(i + "-") for i in ids)


def thinking_params(model: str, reasoning: str) -> dict:
    """`RAG_LLM_REASONING` as Claude request fields: "none" = as little thinking as the model allows,
    "low" / "medium" / "high"... = the effort level (adaptive thinking), "" = the model's default."""
    if not reasoning:
        return {}
    if reasoning == "none":
        if same_model(model, BETWEEN_TOOLS_MODELS):
            return {"thinking": {"type": "between_tools"}}
        if same_model(model, NO_THINKING_MODELS):
            return {"thinking": {"type": "disabled"}}
        return {"output_config": {"effort": "low"}}
    return {"output_config": {"effort": reasoning}}


def claude_error(exc: Exception) -> str:
    """The API's own message (e.g. "Your credit balance is too low...") rather than the SDK's wrapper."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict) and body["error"].get("message"):
        return body["error"]["message"]
    return getattr(exc, "message", None) or str(exc)


def as_llm_error(exc: Exception) -> LLMError:
    if isinstance(exc, anthropic.APIStatusError):
        return LLMError(f"Claude API error {exc.status_code}: {claude_error(exc)}")
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMError(f"Claude API not reachable: {exc}")
    return LLMError(f"Claude API: {exc}")


def claude_key_set() -> bool:
    """Whether the Claude API can be called: without a key the SDK only fails on the first request, with a TypeError."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


class ClaudeClient:
    """The Claude API through the official SDK, with the same interface as LLMClient.

    `temperature` is accepted for that interface but not sent: current Claude models reject
    non-default sampling values.
    """

    def __init__(self, model: str = config.LLM_MODEL, reasoning: str = config.LLM_REASONING,
                 allowed: list[str] = config.CLAUDE_MODELS, timeout: float = 120.0, client=None):
        self.model = model
        self.reasoning = reasoning
        self.allowed = allowed  # empty: every model the key can use
        self.timeout = timeout
        self._models: tuple[float, list[dict]] | None = None  # (fetched at, list)
        if client is not None:
            self.__dict__["client"] = client  # tests: a fake SDK client

    @cached_property
    def client(self) -> anthropic.Anthropic:
        # Checked here: without a key the SDK only fails on the first request, with a TypeError.
        if not claude_key_set():
            raise LLMError("ANTHROPIC_API_KEY is not set: add it to .env (RAG_LLM_PROVIDER=anthropic)")
        return anthropic.Anthropic(timeout=self.timeout)

    def describe(self) -> str:
        return f"Claude API · reasoning {self.reasoning or 'default'}"

    def list_models(self, fresh: bool = False) -> list[dict]:
        """The allowed models (`RAG_CLAUDE_MODELS`, in that order, plus `RAG_LLM_MODEL`) the key can use;
        all of them when nothing is allowed explicitly. `loaded` is None: nothing to load on an API.
        Reused for MODELS_TTL seconds unless `fresh`, so checking a requested model costs no request."""
        if not fresh and self._models and time.monotonic() - self._models[0] < MODELS_TTL:
            return self._models[1]
        try:
            ids = [m.id for m in self.client.models.list()]
        except anthropic.AnthropicError as exc:
            raise as_llm_error(exc) from exc
        if self.allowed:
            wanted = list(dict.fromkeys([*self.allowed, *([self.model] if self.model else [])]))
            ids = [m for m in wanted if m in ids]
        models = [{"id": m, "loaded": None} for m in ids]
        self._models = (time.monotonic(), models)
        return models

    def resolve_model(self, models: list[dict] | None = None) -> str:
        return self.model or CLAUDE_DEFAULT_MODEL

    def status(self) -> dict:
        try:
            models = self.list_models(fresh=True)
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
        """Stream answer text; `info` gets `finish_reason` ("stop" / "length"), `usage` in OpenAI terms
        (`prompt_tokens`, `completion_tokens`) and `first_delta_at` (first generated token, thinking
        included, as for LLMClient). A refusal is an LLMError."""
        info = {} if info is None else info
        model = model or self.resolve_model()
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        params = thinking_params(model, self.reasoning)
        if params.get("thinking", {}).get("type") not in ("between_tools", "disabled"):
            max_tokens += THINKING_HEADROOM  # adaptive thinking may run and would eat the answer's budget
        if same_model(model, FALLBACK_MODELS):
            params |= {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        try:
            with self.client.beta.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[m for m in messages if m["role"] != "system"],
                **params,
            ) as stream:
                for event in stream:
                    if event.type in ("text", "thinking"):
                        info.setdefault("first_delta_at", time.perf_counter())
                    if event.type == "text" and event.text:
                        yield event.text
                final = stream.get_final_message()
        except anthropic.AnthropicError as exc:
            raise as_llm_error(exc) from exc
        usage = final.usage
        info["usage"] = {
            "prompt_tokens": usage.input_tokens + (usage.cache_read_input_tokens or 0)
            + (usage.cache_creation_input_tokens or 0),
            "completion_tokens": usage.output_tokens,
        }
        if final.stop_reason == "refusal":
            category = getattr(final.stop_details, "category", None)
            raise LLMError(f"The model declined to answer{f' ({category})' if category else ''}")
        info["finish_reason"] = STOP_TO_FINISH.get(final.stop_reason, final.stop_reason)


LLM = LLMClient | ClaudeClient


def make_llm(model: str = config.LLM_MODEL, provider: str = config.LLM_PROVIDER) -> LLM:
    """The client for `RAG_LLM_PROVIDER`: "openai" (LM Studio and other OpenAI-compatible servers) or "anthropic"."""
    if provider == "openai":
        return LLMClient(model=model)
    if provider == "anthropic":
        return ClaudeClient(model=model)
    raise ValueError(f"Unknown RAG_LLM_PROVIDER {provider!r}, expected 'openai' or 'anthropic'")
