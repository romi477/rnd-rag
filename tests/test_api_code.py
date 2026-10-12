import json

import pytest
from fastapi.testclient import TestClient
from test_api import FakeLLM, FakeSearcher, events
from test_code_index import FakeEmbedder
from test_code_lookup import world  # noqa: F401  (fixture)

from rag import api, config
from rag.code_graph import graph_path
from rag.code_search import CodeSearcher

TRACEBACK = """Traceback (most recent call last):
  File "/opt/project/demo_shop/models/order.py", line 8, in action_sync
    return super().action_sync()
  File "/opt/project/demo_core/models/order.py", line 8, in action_sync
    ErrorStore.raise_error(err_code='E514')
odoo.exceptions.UserError: [E514] unpaid invoice
"""


@pytest.fixture
def client(world, tmp_path, monkeypatch):  # noqa: F811
    graph, store = world
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    graph.save(graph_path())
    state = {
        "searcher": FakeSearcher(), "llm": FakeLLM(),
        "code_searcher": CodeSearcher(FakeEmbedder(), store),
    }
    monkeypatch.setattr(api, "state", state)
    monkeypatch.setattr(api, "answer_times", api.deque())
    return TestClient(api.app, base_url="http://127.0.0.1:8000")


def post(client, **body):
    return client.post("/api/code/answer", json=body)


def test_modules_lists_the_graph_with_versions_and_method_counts(client):
    res = client.get("/api/code/modules")
    assert res.status_code == 200
    by_name = {m["name"]: m for m in res.json()["modules"]}
    assert by_name["demo_shop"]["version"] == "19.0.1.0.0" and by_name["demo_shop"]["ref"] == "test"
    assert by_name["demo_core"]["methods"] > 0


def test_code_search_returns_units_with_path_and_lines(client):
    res = client.get("/api/code/search", params={"q": "action_sync", "mode": "fts"})
    assert res.status_code == 200
    first = res.json()["results"][0]
    assert first["name"] == "action_sync" and first["path"] == "models/order.py" and first["line_start"] == 7


def test_code_search_filters_by_module_and_rejects_a_bad_mode(client):
    res = client.get("/api/code/search", params={"q": "order", "module": "demo_shop"})
    assert {r["module"] for r in res.json()["results"]} == {"demo_shop"}
    assert client.get("/api/code/search", params={"q": "x", "mode": "nope"}).status_code == 400


def test_everything_is_503_without_a_code_index(client):
    api.state["code_searcher"] = None
    assert client.get("/api/code/modules").status_code == 503
    assert client.get("/api/code/search", params={"q": "x"}).status_code == 503
    assert post(client, traceback=TRACEBACK).status_code == 503


def test_answer_streams_evidence_sources_and_the_llm_text(client):
    res = post(client, traceback=TRACEBACK, question="why?")
    assert res.status_code == 200
    ev = events(res)
    assert [e["type"] for e in ev] == ["results", "sources", "delta", "done"]
    results = ev[0]
    assert results["exception"]["name"] == "odoo.exceptions.UserError" and results["error_codes"] == ["E514"]
    assert [(f["module"], f["found"]) for f in results["frames"]] == [("demo_core", True), ("demo_shop", True)]
    assert results["pieces"][0]["role"] == "frame" and "return True" in results["pieces"][0]["text"]
    assert ev[1]["sources"][0]["kind"] == "code" and ev[1]["sources"][0]["n"] == 1


def test_answer_needs_a_question_or_a_traceback(client):
    assert post(client, question="", traceback="").status_code == 400
    assert post(client, traceback="x" * 30001).status_code == 422


def test_question_only_answer_uses_search(client):
    ev = events(post(client, question="how does sync work with the client"))
    assert ev[0]["frames"] == [] and ev[0]["pieces"] and ev[0]["pieces"][0]["role"] == "search"


def test_docs_flag_adds_confluence_sources_after_the_code_ones(client):
    ev = events(post(client, traceback=TRACEBACK, docs=True))
    sources = ev[1]["sources"]
    assert sources[-1]["kind"] == "page" and sources[-1]["title"] == "Install"
    assert sources[-1]["n"] == len(sources)
    assert api.state["searcher"].calls[-1]["rerank"] is True
    assert not events(post(client, traceback=TRACEBACK))[1]["sources"][-1]["kind"] == "page"


def test_unknown_model_is_400_and_the_hourly_limit_applies(client, monkeypatch):
    assert post(client, traceback=TRACEBACK, model="nope").status_code == 400
    monkeypatch.setattr(config, "ANSWERS_PER_HOUR", 1)
    assert post(client, traceback=TRACEBACK).status_code == 200
    assert post(client, traceback=TRACEBACK).status_code == 429


def test_code_is_not_sent_to_an_external_llm_unless_allowed(client, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "CODE_EXTERNAL_LLM", False)
    res = post(client, traceback=TRACEBACK)
    assert res.status_code == 403 and "RAG_CODE_EXTERNAL_LLM" in res.json()["detail"]
    monkeypatch.setattr(config, "CODE_EXTERNAL_LLM", True)
    assert post(client, traceback=TRACEBACK).status_code == 200


def test_results_event_is_json_per_line(client):
    lines = [line for line in post(client, traceback=TRACEBACK).text.splitlines() if line]
    assert all(isinstance(json.loads(line), dict) for line in lines)
