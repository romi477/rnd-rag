import json
import time

import httpx
import pytest

from rag.answer import EMPTY_ANSWER, NO_SOURCES, build_context, build_messages, stream_answer
from rag.llm import LLMClient, LLMError, parse_sse, split_lines
from rag.search import ChunkHit, PageHit


def page(pid: str, title: str, texts: list[str], product: str = "Printing") -> PageHit:
    chunks = [ChunkHit(f"{pid}:{i}", ["Setup"] if i else [], t, 1.0) for i, t in enumerate(texts)]
    return PageHit(pid, title, f"https://x/{pid}", product, [product], "2026-01-01", [], 1.0, chunks)


def sse(*events) -> list[str]:
    return [f"data: {json.dumps(e)}" for e in events] + ["data: [DONE]"]


def delta(**d) -> dict:
    return {"choices": [{"delta": d}]}


# --- SSE parsing ------------------------------------------------------------------


def test_parse_sse_yields_content_and_skips_reasoning():
    lines = ["", ": keep-alive"] + sse(
        delta(role="assistant", reasoning_content="thinking..."),
        delta(content="Hello"),
        delta(content=" world"),
    )
    assert "".join(parse_sse(lines)) == "Hello world"


def test_parse_sse_collects_usage_and_finish_reason():
    info = {}
    lines = sse(
        delta(content="Hi"),
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 1}},
    )
    assert "".join(parse_sse(lines, info)) == "Hi"
    assert info["usage"] == {"prompt_tokens": 12, "completion_tokens": 1} and info["finish_reason"] == "length"
    assert info["first_delta_at"] > 0


def test_parse_sse_cut_off_stream_is_an_error():
    lines = [f"data: {json.dumps(delta(content='Half an ans'))}"]  # no finish_reason, no [DONE]
    with pytest.raises(LLMError, match="before the answer was complete"):
        list(parse_sse(lines))
    # a finish reason without [DONE] is a complete answer
    assert "".join(parse_sse([f"data: {json.dumps({'choices': [{'delta': {'content': 'ok'}, 'finish_reason': 'stop'}]})}"])) == "ok"


def test_parse_sse_tolerates_null_fields():
    lines = sse({"error": None, "choices": None}, delta(content="ok"))
    assert "".join(parse_sse(lines)) == "ok"


def test_split_lines_keeps_unicode_line_separators():
    text = "Line\u2028 and\u0085 more"
    body = f"data: {json.dumps(delta(content=text), ensure_ascii=False)}\n\ndata: [DONE]\n"
    lines = list(split_lines([body[:20], body[20:]]))  # split across network chunks
    assert "".join(parse_sse(lines)) == text


def test_parse_sse_stops_at_done_and_raises_on_error():
    assert list(parse_sse(["data: [DONE]", f"data: {json.dumps(delta(content='late'))}"])) == []
    with pytest.raises(LLMError):
        list(parse_sse([f"data: {json.dumps({'error': 'model crashed'})}"]))


# --- context building -------------------------------------------------------------


def test_build_context_numbers_pages_and_cleans_text():
    pages = [page("1", "Install", ["Run the installer.\n\n\n\n", "Then restart."]), page("2", "Labels", ["Label text."])]
    sources, context = build_context(pages, max_pages=5, max_tokens=1000)
    assert [(s.n, s.page_id) for s in sources] == [(1, "1"), (2, "2")]
    assert context.startswith("[1] Install\nPath: Printing\n\nRun the installer.")
    assert "(Section: Setup)\nThen restart." in context
    assert "[2] Labels" in context and "installer.\n\n(Section" in context


def test_build_context_respects_page_and_token_limits():
    pages = [page(str(i), f"P{i}", ["x" * 400]) for i in range(1, 6)]  # 100 tokens each
    assert len(build_context(pages, max_pages=3, max_tokens=10_000)[0]) == 3
    assert len(build_context(pages, max_pages=5, max_tokens=250)[0]) == 2
    # the first chunk always goes in, even if it alone exceeds the budget
    assert len(build_context(pages, max_pages=5, max_tokens=10)[0]) == 1


def test_build_messages_puts_question_after_sources():
    user = build_messages("How?", "[1] A")[1]["content"]
    assert user.index("[1] A") < user.index("Question: How?")


# --- answer events ----------------------------------------------------------------


class FakeLLM:
    def __init__(self, pieces=(), error=None, finish="stop"):
        self.pieces, self.error, self.messages, self.finish = pieces, error, None, finish

    def resolve_model(self):
        return "fake-model"

    def stream_chat(self, messages, model=None, info=None):
        self.messages = messages
        if self.error:
            raise LLMError(self.error)
        if info is not None and self.pieces:
            info["first_delta_at"] = time.perf_counter()
        yield from self.pieces
        if info is not None:
            info["finish_reason"] = self.finish
            info["usage"] = {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280,
                             "completion_tokens_details": {"reasoning_tokens": 0}}


def test_stream_answer_event_sequence():
    llm = FakeLLM(["Use the ", "installer [1]."])
    events = list(stream_answer("How to install?", [page("1", "Install", ["Run it."])], llm))
    assert [e["type"] for e in events] == ["sources", "delta", "delta", "done"]
    assert events[0]["sources"][0]["title"] == "Install"
    assert events[-1]["model"] == "fake-model" and events[-1]["truncated"] is False
    assert events[-1]["usage"] == {"prompt_tokens": 1200, "completion_tokens": 80}
    assert events[-1]["first_token_ms"] >= 0
    assert "Question: How to install?" in llm.messages[1]["content"]


def test_stream_answer_flags_truncated_and_empty_answers():
    hits = [page("1", "A", ["t"])]
    assert list(stream_answer("q", hits, FakeLLM(["Cut"], finish="length")))[-1]["truncated"] is True
    assert list(stream_answer("q", hits, FakeLLM([])))[-1] == {"type": "error", "message": EMPTY_ANSWER}


def test_stream_answer_without_results_skips_llm():
    llm = FakeLLM(["should not be used"])
    events = list(stream_answer("q", [], llm))
    assert [e["type"] for e in events] == ["sources", "delta", "done"]
    assert events[1]["text"] == NO_SOURCES and llm.messages is None


def test_stream_answer_reports_llm_error():
    events = list(stream_answer("q", [page("1", "A", ["t"])], FakeLLM(error="server down")))
    assert events[-1] == {"type": "error", "message": "server down"}


# --- LLM client against a mocked OpenAI-compatible server -------------------------


def client(handler) -> LLMClient:
    llm = LLMClient(base_url="http://llm.test/v1", model="", reasoning="none")
    llm.http = httpx.Client(transport=httpx.MockTransport(handler))
    return llm


def test_resolve_model_prefers_loaded_lmstudio_llm():
    def handler(request):
        assert request.url.path == "/api/v0/models"
        return httpx.Response(200, json={"data": [
            {"id": "nomic-embed", "type": "embeddings", "state": "loaded"},
            {"id": "big-llm", "type": "llm", "state": "not-loaded"},
            {"id": "gemma", "type": "vlm", "state": "loaded"},
        ]})
    assert client(handler).resolve_model() == "gemma"


def test_resolve_model_falls_back_to_openai_models_list():
    def handler(request):
        if request.url.path == "/api/v0/models":
            return httpx.Response(404)
        return httpx.Response(200, json={"data": [{"id": "text-embedding-x"}, {"id": "llama"}]})
    assert client(handler).resolve_model() == "llama"


def test_list_models_and_status_with_lmstudio():
    def handler(request):
        return httpx.Response(200, json={"data": [
            {"id": "nomic-embed", "type": "embeddings", "state": "loaded"},
            {"id": "qwen-27b", "type": "llm", "state": "not-loaded"},
            {"id": "gemma", "type": "vlm", "state": "loaded"},
        ]})
    status = client(handler).status()
    assert status["models"] == [{"id": "qwen-27b", "loaded": False}, {"id": "gemma", "loaded": True}]
    assert status["available"] and status["model"] == "gemma"


def test_resolve_model_without_loaded_models_takes_first():
    assert LLMClient(model="").resolve_model([{"id": "a", "loaded": False}, {"id": "b", "loaded": False}]) == "a"


def test_stream_answer_uses_requested_model():
    llm = FakeLLM(["ok"])
    events = list(stream_answer("q", [page("1", "A", ["t"])], llm, model="qwen-27b"))
    assert events[-1]["model"] == "qwen-27b"


def test_status_when_server_is_down():
    def handler(request):
        raise httpx.ConnectError("refused")
    status = client(handler).status()
    assert status["available"] is False and "not reachable" in status["error"]


def test_status_with_unexpected_model_list():
    status = client(lambda request: httpx.Response(200, text="<html>not json</html>")).status()
    assert status["available"] is False and "Unexpected model list" in status["error"]
    status = client(lambda request: httpx.Response(200, json={"data": [{"name": "no id"}]})).status()
    assert status["available"] is False


def test_stream_chat_sends_reasoning_off_and_streams_text():
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        body = "\n\n".join(sse(delta(content="Hi"), delta(content="!"))) + "\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    llm = client(handler)
    assert "".join(llm.stream_chat([{"role": "user", "content": "hey"}], model="m")) == "Hi!"
    assert sent["reasoning_effort"] == "none" and sent["stream"] is True and sent["model"] == "m"
    assert "stream_options" not in sent  # only requested together with info


def test_stream_chat_requests_usage():
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        body = "\n\n".join(sse(delta(content="Hi"), {"choices": [], "usage": {"completion_tokens": 1}}))
        return httpx.Response(200, text=body + "\n\n", headers={"content-type": "text/event-stream"})

    info = {}
    assert "".join(client(handler).stream_chat([], model="m", info=info)) == "Hi"
    assert sent["stream_options"] == {"include_usage": True} and info["usage"] == {"completion_tokens": 1}


def test_stream_chat_http_error_becomes_llm_error():
    llm = client(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(LLMError, match="500"):
        list(llm.stream_chat([], model="m"))


def test_low_relevance_pages_are_not_sources():
    weak = page("2", "Weak", ["Unrelated."])
    weak.low_relevance = True
    sources, context = build_context([page("1", "Strong", ["Answer."]), weak])
    assert [s.page_id for s in sources] == ["1"] and "Weak" not in context


def test_only_low_relevance_pages_skips_llm():
    weak = page("1", "Weak", ["Unrelated."])
    weak.low_relevance = True
    llm = FakeLLM(["should not be used"])
    events = list(stream_answer("q", [weak], llm))
    assert events[1]["text"] == NO_SOURCES and llm.messages is None
