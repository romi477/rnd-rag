import pytest
from test_answer import FakeLLM, page
from test_api import FakeLLM as ApiLLM
from test_api import FakeSearcher, events
from test_api_code import TRACEBACK, post
from test_api_code import client as code_client  # noqa: F401  (fixture)
from test_code_answer import world  # noqa: F401  (fixture)
from typer.testing import CliRunner

from rag import api, config
from rag.answer import (
    NO_SOURCES_BY_LANG,
    SYSTEM_PROMPT,
    build_messages,
    check_language,
    language_name,
    stream_answer,
    system_prompt,
)
from rag.cli import app
from rag.code_answer import (
    CODE_SYSTEM_PROMPT,
    NO_CODE_SOURCES_BY_LANG,
    Evidence,
    build_code_messages,
    code_system_prompt,
    gather,
    stream_code_answer,
)

LANGS = ["en", "es", "de", "pl", "ru"]


def test_supported_languages_in_order_with_english_and_native_names():
    assert list(config.ANSWER_LANGUAGES) == LANGS
    assert config.ANSWER_LANGUAGES["es"] == ("Spanish", "Español")
    assert config.ANSWER_LANGUAGES["ru"] == ("Russian", "Русский")


def test_check_language_normalises_and_rejects_unknown_codes():
    assert check_language("RU") == "ru" and check_language(" de-DE ") == "de" and check_language("pl_PL") == "pl"
    assert check_language(None) == check_language("") == config.ANSWER_LANGUAGE
    assert language_name("es") == "Spanish"
    with pytest.raises(ValueError, match="xx"):
        check_language("xx")


def test_english_prompts_are_exactly_what_they_were():
    assert system_prompt("en") == SYSTEM_PROMPT and "Always answer in English, like the documentation." in SYSTEM_PROMPT
    assert code_system_prompt("en") == CODE_SYSTEM_PROMPT and "Always answer in English." in CODE_SYSTEM_PROMPT
    assert "Answer in" not in build_messages("How?", "[1] A")[1]["content"]


@pytest.mark.parametrize("lang", LANGS[1:])
def test_other_languages_name_the_language_in_the_system_prompt_and_after_the_question(lang):
    name = config.ANSWER_LANGUAGES[lang][0]
    system, user = (m["content"] for m in build_messages("How?", "[1] A", lang=lang))
    assert f"Always answer in {name}" in system and "answer in English" not in system.lower()
    assert "exactly as in the sources" in system  # menu paths and error messages stay as they are
    assert user.rstrip().endswith(f"Answer in {name}.")
    assert user.index("Question: How?") < user.index(f"Answer in {name}.")


def test_code_prompt_is_translated_the_same_way(world):  # noqa: F811
    graph, store = world
    ev = gather(store, graph, "", TRACEBACK, searcher=None)
    system, user = (m["content"] for m in build_code_messages("Why?", ev, "ctx", lang="ru"))
    assert "Always answer in Russian" in system and "answer in English" not in system.lower()
    assert user.rstrip().endswith("Answer in Russian.")
    assert "Answer in" not in build_code_messages("Why?", ev, "ctx")[1]["content"]


def test_nothing_found_texts_exist_for_every_language():
    assert set(NO_SOURCES_BY_LANG) == set(NO_CODE_SOURCES_BY_LANG) == set(LANGS)
    assert all(NO_SOURCES_BY_LANG[lang].strip() for lang in LANGS)


def test_stream_answer_asks_for_the_language_and_localises_the_empty_result():
    llm = FakeLLM(["Respuesta [1]."])
    list(stream_answer("q", [page("1", "A", ["t"])], llm, lang="es"))
    assert "Spanish" in llm.messages[0]["content"]
    llm = FakeLLM(["never"])
    events_ = list(stream_answer("q", [], llm, lang="ru"))
    assert events_[1]["text"] == NO_SOURCES_BY_LANG["ru"] and llm.messages is None


def test_stream_code_answer_localises_the_empty_result():
    llm = FakeLLM(["never"])
    events_ = list(stream_code_answer("q", Evidence(), llm, lang="de"))
    assert events_[1]["text"] == NO_CODE_SOURCES_BY_LANG["de"] and llm.messages is None


# --- API ----------------------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(api, "state", {"searcher": FakeSearcher(), "llm": ApiLLM()})
    monkeypatch.setattr(api, "answer_times", api.deque())
    return TestClient(api.app, base_url="http://127.0.0.1:8000")


def test_languages_endpoint_lists_codes_names_and_the_default(client):
    body = client.get("/api/languages").json()
    assert body["default"] == config.ANSWER_LANGUAGE
    assert [(x["code"], x["name"], x["native"]) for x in body["languages"]] == [
        ("en", "English", "English"), ("es", "Spanish", "Español"), ("de", "German", "Deutsch"),
        ("pl", "Polish", "Polski"), ("ru", "Russian", "Русский"),
    ]


def test_answer_endpoint_passes_the_language_on_and_rejects_unknown_ones(client):
    assert client.get("/api/answer", params={"q": "x", "lang": "pl"}).status_code == 200
    assert "Polish" in api.state["llm"].messages[0]["content"]
    client.get("/api/answer", params={"q": "x"})
    assert "Always answer in English" in api.state["llm"].messages[0]["content"]
    res = client.get("/api/answer", params={"q": "x", "lang": "xx"})
    assert res.status_code == 400 and "xx" in res.json()["detail"]


def test_a_rejected_language_does_not_use_up_the_hourly_limit(client, monkeypatch):
    monkeypatch.setattr(config, "ANSWERS_PER_HOUR", 1)
    assert client.get("/api/answer", params={"q": "x", "lang": "xx"}).status_code == 400
    assert client.get("/api/answer", params={"q": "x", "lang": "es"}).status_code == 200


def test_code_answer_endpoint_takes_a_language(code_client):  # noqa: F811
    assert post(code_client, traceback=TRACEBACK, language="de").status_code == 200
    assert "German" in api.state["llm"].messages[0]["content"]
    assert post(code_client, traceback=TRACEBACK, language="xx").status_code == 400
    ev = events(post(code_client, traceback=TRACEBACK, language="ru", answer=False))
    assert [e["type"] for e in ev] == ["results"]


def test_cli_rejects_an_unknown_language():
    res = CliRunner().invoke(app, ["ask", "q", "--lang", "xx"])
    assert res.exit_code == 2 and "es" in res.output


# --- answer length ------------------------------------------------------------------------------


class KwargsLLM(FakeLLM):
    """FakeLLM that keeps the keyword arguments of the chat call (max_tokens)."""

    def stream_chat(self, messages, model=None, info=None, **kwargs):
        self.kwargs = kwargs
        yield from super().stream_chat(messages, model=model, info=info)


def test_languages_other_than_english_get_a_longer_token_budget():
    from rag.answer import answer_max_tokens

    assert answer_max_tokens(1000, "en") == 1000
    assert answer_max_tokens(1000, "ru") > answer_max_tokens(1000, "es") > 1000
    assert all(answer_max_tokens(1000, lang) >= 1000 for lang in LANGS)
    with pytest.raises(ValueError):
        answer_max_tokens(1000, "xx")


def test_docs_answer_in_english_keeps_the_servers_default_limit_other_languages_get_more():
    llm = KwargsLLM(["ok [1]"])
    list(stream_answer("q", [page("1", "A", ["t"])], llm))
    assert llm.kwargs == {}  # nothing passed: the client's own default, as before
    list(stream_answer("q", [page("1", "A", ["t"])], llm, lang="ru"))
    assert llm.kwargs["max_tokens"] > config.LLM_MAX_TOKENS


def test_code_answers_have_their_own_larger_limit(world, monkeypatch):  # noqa: F811
    graph, store = world
    ev = gather(store, graph, "", TRACEBACK, searcher=None)
    monkeypatch.setattr(config, "CODE_LLM_MAX_TOKENS", 2000)
    llm = KwargsLLM(["ok [1]"])
    list(stream_code_answer("q", ev, llm))
    assert llm.kwargs["max_tokens"] == 2000
    list(stream_code_answer("q", ev, llm, lang="ru"))
    assert llm.kwargs["max_tokens"] > 2000
