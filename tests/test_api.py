import json

import pytest
from fastapi.testclient import TestClient

from rag import api, config
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

    def stream_chat(self, messages, model=None, info=None, **kwargs):
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


def test_search_and_products(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")  # nothing synced: no creation dates
    res = client.get("/api/search", params={"q": "install"})
    assert res.status_code == 200 and res.json()["results"][0]["page_id"] == "1"
    assert client.get("/api/products").json() == ["Printing"]
    pages = [p | {"created": ""} for p in api.state["searcher"].pages()]
    assert client.get("/api/pages").json() == {"count": 1, "pages": pages}


def test_pages_carry_the_creation_date_from_the_synced_metadata(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "SPACE_KEY", "demo")
    write_synced_page(tmp_path / "raw", page_id="1", created="2024-05-06T07:08:09.000Z")
    assert client.get("/api/pages").json()["pages"][0]["created"] == "2024-05-06T07:08:09.000Z"
    assert client.get("/api/pages/1").json()["created"] == "2024-05-06T07:08:09.000Z"


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
        def stream_chat(self, messages, model=None, info=None, **kwargs):
            raise TypeError("boom")
            yield

    api.state["llm"] = Broken()
    assert events(client.get("/api/answer", params={"q": "x"}))[-1] == {"type": "error", "message": "Internal error: boom"}


def write_synced_page(raw_dir, page_id="123", body="Intro.\n\n[image: a.png]\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n",
                      created=None):
    pages = raw_dir / "demo" / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    meta = {"id": page_id, "title": "Install", "space": "demo", "url": "https://x/1", "version": 3,
            "updated": "2026-01-02T03:04:05.000Z", "breadcrumbs": ["Home", "Printing"], "labels": ["setup"]}
    if created:
        meta["created"] = created
    (pages / f"{page_id}.json").write_text(json.dumps(meta))
    (pages / f"{page_id}.md").write_text(f"---\nid: \"{page_id}\"\ntitle: \"Install\"\n---\n# Install\n\n{body}")


def test_page_returns_the_synced_markdown_without_front_matter_and_title(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    write_synced_page(tmp_path / "raw")
    res = client.get("/api/pages/123")
    assert res.status_code == 200
    page = res.json()
    assert page["markdown"].startswith("Intro.") and "| A | B |" in page["markdown"] and "id:" not in page["markdown"]
    assert (page["title"], page["url"], page["product"], page["labels"]) == ("Install", "https://x/1", "Printing", ["setup"])
    assert page["breadcrumbs"] == ["Printing"] and page["updated"].startswith("2026-01-02")


def test_page_is_404_when_not_synced_and_the_id_cannot_name_a_path(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    write_synced_page(tmp_path / "raw")
    assert client.get("/api/pages/999").status_code == 404
    for bad in ("..", "12a", "%2e%2e%2f%2e%2e%2fsecret", "1" * 30):
        assert client.get(f"/api/pages/{bad}").status_code in (404, 422), bad
    assert client.get("/api/pages/..%2f..%2fetc").status_code in (404, 422)


def test_page_with_a_broken_synced_copy_is_a_500_with_a_hint(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    write_synced_page(tmp_path / "raw")
    (tmp_path / "raw" / "demo" / "pages" / "123.json").write_text("{not json")
    res = client.get("/api/pages/123")
    assert res.status_code == 500 and "sync" in res.json()["detail"]


def basic(user: str, token: str) -> dict:
    import base64

    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{token}".encode()).decode()}


@pytest.fixture
def locked(client, monkeypatch):
    monkeypatch.setattr(config, "API_TOKEN", "s3cret")
    return client


def test_no_token_configured_means_no_login(client):
    assert client.get("/api/products").status_code == 200


@pytest.mark.parametrize("path", ["/api/products", "/api/search?q=x", "/api/answer?q=x", "/api/code/modules", "/"])
def test_token_guards_api_and_ui(locked, path):
    res = locked.get(path)
    assert res.status_code == 401
    assert res.headers["WWW-Authenticate"].startswith("Basic")  # the browser asks once, then sends it with every request
    assert "Printing" not in res.text


def test_token_is_the_password_and_the_user_name_is_ignored(locked):
    assert locked.get("/api/products", headers=basic("", "s3cret")).json() == ["Printing"]
    assert locked.get("/api/products", headers=basic("anyone", "s3cret")).status_code == 200
    assert locked.get("/").status_code == 401  # the page itself is behind it too
    assert locked.get("/", headers=basic("", "s3cret")).status_code == 200


def test_decoded_header_without_a_colon_is_the_token(locked):
    import base64

    header = {"Authorization": "Basic " + base64.b64encode(b"s3cret").decode()}
    assert locked.get("/api/products", headers=header).status_code == 200


@pytest.mark.parametrize("headers", [
    basic("", "wrong"), basic("s3cret", ""), basic("", "s3cret "), {"Authorization": "Bearer s3cret"},
    {"Authorization": "Basic !!!not-base64"}, {"Authorization": "Basic"}, {"Authorization": ""},
])
def test_wrong_credentials_are_401(locked, headers):
    assert locked.get("/api/products", headers=headers).status_code == 401


def test_unauthorised_request_does_not_reach_the_handlers(locked):
    locked.get("/api/answer", params={"q": "install"})
    assert api.state["searcher"].calls == []
