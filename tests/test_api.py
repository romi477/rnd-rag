import json

import pytest
from fastapi.testclient import TestClient

from rag import api
from rag.llm import LLMError
from rag.search import ChunkHit, PageHit


class FakeSearcher:
    def __init__(self):
        self.calls = []

    def search(self, q, **kwargs):
        self.calls.append(kwargs)
        return [PageHit("1", "Install", "https://x/1", "Printing", ["Printing"], "2026", [], 0.9,
                        [ChunkHit("1:0", [], "Run the installer.", 0.9)])]

    def products(self):
        return ["Printing"]

    def pages(self):
        return [{"page_id": "1", "title": "Install", "url": "https://x/1", "product": "Printing",
                 "breadcrumbs": ["Printing"], "updated": "2026", "labels": []}]


class FakeLLM:
    def __init__(self, models=("m1",), down=False):
        self.models, self.down = models, down

    def list_models(self):
        if self.down:
            raise LLMError("down")
        return [{"id": m, "loaded": True} for m in self.models]

    def resolve_model(self):
        return self.models[0]

    def status(self):
        return {"available": True, "model": "m1", "models": self.list_models(), "error": None}

    def stream_chat(self, messages, model=None, info=None):
        self.messages = messages
        yield "Use the installer [1]."
        info["finish_reason"] = "stop"


@pytest.fixture
def client(monkeypatch):
    # no lifespan: the real one would load the models
    monkeypatch.setattr(api, "state", {"searcher": FakeSearcher(), "llm": FakeLLM()})
    monkeypatch.setattr(api, "answer_times", api.deque())
    return TestClient(api.app, base_url="http://127.0.0.1:8000")


def events(resp) -> list[dict]:
    return [json.loads(line) for line in resp.text.splitlines() if line]


def test_search_and_products(client):
    res = client.get("/api/search", params={"q": "install"})
    assert res.status_code == 200 and res.json()["results"][0]["page_id"] == "1"
    assert client.get("/api/products").json() == ["Printing"]
    assert client.get("/api/pages").json() == {"count": 1, "pages": api.state["searcher"].pages()}


def test_bad_mode_is_400(client):
    assert client.get("/api/search", params={"q": "x", "mode": "fulltext"}).status_code == 400


def test_answer_streams_events_with_rerank_on_by_default(client):
    res = client.get("/api/answer", params={"q": "install"})
    assert [e["type"] for e in events(res)] == ["results", "sources", "delta", "done"]
    assert api.state["searcher"].calls[-1]["rerank"] is True


def test_answer_rejects_unknown_model(client):
    assert client.get("/api/answer", params={"q": "x", "model": "nope"}).status_code == 400
    assert client.get("/api/answer", params={"q": "x", "model": "m1"}).status_code == 200


def test_answer_with_llm_server_down_still_streams_an_error(client):
    api.state["llm"] = FakeLLM(down=True)
    assert client.get("/api/answer", params={"q": "x", "model": "m1"}).status_code == 200


def test_foreign_host_header_is_rejected(client):
    assert client.get("/api/products", headers={"Host": "attacker.example"}).status_code == 400
    assert client.get("/api/products", headers={"Host": "localhost:8000"}).status_code == 200


def test_no_docs_pages(client):
    assert client.get("/docs").status_code == 404 and client.get("/redoc").status_code == 404


def test_ui_files_are_revalidated(client):
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert "cache-control" not in client.get("/api/products").headers


def test_favicon_is_served_and_linked(client):
    icon = client.get("/favicon.svg")
    assert icon.status_code == 200 and icon.headers["content-type"].startswith("image/svg+xml")
    assert 'rel="icon" href="favicon.svg"' in client.get("/").text


def test_answers_per_hour_limit(client, monkeypatch):
    monkeypatch.setattr(api.config, "ANSWERS_PER_HOUR", 2)
    codes = [client.get("/api/answer", params={"q": "x"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_unexpected_llm_failure_ends_the_stream_with_an_error(client):
    class Broken(FakeLLM):
        def stream_chat(self, messages, model=None, info=None):
            raise TypeError("boom")
            yield

    api.state["llm"] = Broken()
    assert events(client.get("/api/answer", params={"q": "x"}))[-1] == {"type": "error", "message": "Internal error: boom"}
