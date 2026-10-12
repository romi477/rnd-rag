"""ClaudeClient on a fake SDK client: no network, no API key."""

from types import SimpleNamespace as NS

import anthropic
import pytest

from rag.llm import THINKING_HEADROOM, ClaudeClient, LLMClient, LLMError, make_llm, same_model, thinking_params

MESSAGES = [{"role": "system", "content": "Rules."}, {"role": "user", "content": "Sources... Question: q"}]


def final(stop_reason="end_turn", category=None, out=12):
    usage = NS(input_tokens=100, output_tokens=out, cache_read_input_tokens=None, cache_creation_input_tokens=None)
    return NS(stop_reason=stop_reason, stop_details=NS(category=category) if category else None, usage=usage)


class FakeStream:
    def __init__(self, events, message):
        self.events, self.message = events, message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.message


class FakeSDK:
    """Records the stream() arguments; `error` is raised instead of streaming."""

    def __init__(self, events=(), message=None, error=None, models=("claude-sonnet-5-5", "claude-haiku-5-5")):
        self.calls, self.error = [], error
        self.events = list(events) or [NS(type="thinking", thinking=""), NS(type="text", text="Use "),
                                       NS(type="text", text="the installer [1].")]
        self.message = message or final()
        self.beta = NS(messages=NS(stream=self.stream))
        self.model_ids, self.list_calls = models, 0
        self.models = NS(list=self.list_models)

    def list_models(self):
        self.list_calls += 1
        return [NS(id=m) for m in self.model_ids]

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return FakeStream(self.events, self.message)


def api_error(status: int, message: str) -> anthropic.APIStatusError:
    """An SDK error as the API returns it, without building an HTTP response."""
    exc = anthropic.BadRequestError.__new__(anthropic.BadRequestError)
    exc.status_code, exc.message = status, f"Error code: {status}"
    exc.body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    return exc


def test_streams_text_with_system_prompt_apart_and_no_temperature():
    sdk = FakeSDK()
    info = {}
    text = "".join(ClaudeClient(client=sdk, reasoning="none").stream_chat(MESSAGES, info=info))
    assert text == "Use the installer [1]."
    call = sdk.calls[0]
    assert call["model"] == "claude-sonnet-5-5" and call["system"] == "Rules."
    assert call["messages"] == [MESSAGES[1]] and "temperature" not in call
    assert call["thinking"] == {"type": "between_tools"}
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert info["finish_reason"] == "stop" and info["usage"] == {"prompt_tokens": 100, "completion_tokens": 12}
    assert "first_delta_at" in info


def test_max_tokens_means_truncated():
    info = {}
    list(ClaudeClient(client=FakeSDK(message=final("max_tokens"))).stream_chat(MESSAGES, info=info))
    assert info["finish_reason"] == "length"


def test_refusal_is_an_error():
    with pytest.raises(LLMError, match=r"declined to answer \(cyber\)"):
        list(ClaudeClient(client=FakeSDK(message=final("refusal", "cyber"))).stream_chat(MESSAGES))


def test_api_error_keeps_the_api_message():
    sdk = FakeSDK(error=api_error(400, "Your credit balance is too low to access the Anthropic API."))
    with pytest.raises(LLMError, match="Claude API error 400: Your credit balance is too low"):
        list(ClaudeClient(client=sdk).stream_chat(MESSAGES))


def test_haiku_runs_without_thinking_or_fallback():
    sdk = FakeSDK()
    list(ClaudeClient(client=sdk, reasoning="none").stream_chat(MESSAGES, model="claude-haiku-5-5"))
    call = sdk.calls[0]
    assert call["thinking"] == {"type": "disabled"} and "fallbacks" not in call


def test_other_models_get_low_effort():
    sdk = FakeSDK()
    list(ClaudeClient(client=sdk, reasoning="none").stream_chat(MESSAGES, model="claude-opus-4-8"))
    call = sdk.calls[0]
    assert call["output_config"] == {"effort": "low"} and "thinking" not in call and "fallbacks" not in call


def test_thinking_params():
    assert thinking_params("claude-sonnet-5-5", "") == {}
    assert thinking_params("claude-sonnet-5-5", "medium") == {"output_config": {"effort": "medium"}}
    assert thinking_params("claude-opus-5-5", "none") == {"output_config": {"effort": "low"}}


def test_models_and_status():
    llm = ClaudeClient(client=FakeSDK(), model="")
    assert llm.status() == {"available": True, "model": "claude-sonnet-5-5", "error": None,
                            "models": [{"id": "claude-sonnet-5-5", "loaded": None},
                                       {"id": "claude-haiku-5-5", "loaded": None}]}
    assert ClaudeClient(client=FakeSDK(), model="claude-opus-5-5").resolve_model() == "claude-opus-5-5"


def test_model_list_is_limited_to_the_allowed_ones():
    sdk = FakeSDK(models=("claude-opus-5-5", "claude-haiku-5-5", "claude-sonnet-5-5", "claude-sonnet-4-6"))
    ids = lambda llm: [m["id"] for m in llm.list_models()]
    allowed = ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-gone-1"]
    assert ids(ClaudeClient(client=sdk, model="", allowed=allowed)) == ["claude-sonnet-5-5", "claude-haiku-5-5"]
    assert ids(ClaudeClient(client=sdk, model="claude-opus-5-5", allowed=allowed))[-1] == "claude-opus-5-5"
    assert len(ids(ClaudeClient(client=sdk, model="", allowed=[]))) == 4


def test_make_llm_picks_the_provider():
    assert isinstance(make_llm(provider="openai"), LLMClient)
    assert isinstance(make_llm(provider="anthropic"), ClaudeClient)
    with pytest.raises(ValueError, match="RAG_LLM_PROVIDER"):
        make_llm(provider="nope")


def test_missing_key_is_an_llm_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    llm = ClaudeClient(model="")
    status = llm.status()
    assert status["available"] is False and "ANTHROPIC_API_KEY is not set" in status["error"]
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        list(llm.stream_chat(MESSAGES))


def test_model_list_is_reused_but_status_refreshes_it():
    sdk = FakeSDK()
    llm = ClaudeClient(client=sdk, model="")
    llm.list_models()
    llm.list_models()
    assert sdk.list_calls == 1
    llm.status()
    assert sdk.list_calls == 2


def test_adaptive_thinking_gets_room_in_max_tokens():
    sdk = FakeSDK()
    list(ClaudeClient(client=sdk, reasoning="medium").stream_chat(MESSAGES, max_tokens=1000))
    list(ClaudeClient(client=sdk, reasoning="none").stream_chat(MESSAGES, max_tokens=1000))
    assert [c["max_tokens"] for c in sdk.calls] == [1000 + THINKING_HEADROOM, 1000]


def test_dated_model_ids_match_their_family():
    assert same_model("claude-sonnet-5-5-20261101", ("claude-sonnet-5-5",))
    assert not same_model("claude-sonnet-5", ("claude-sonnet-5-5",))
