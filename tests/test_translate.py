"""AI translation of a documentation article: the prompt, the title split and the events (no LLM server)."""

import pytest

from rag import config
from rag.llm import LLMError
from rag.translate import build_messages, stream_translation, system_prompt


class FakeLLM:
    """Streams `chunks`, then fails with `error` if given."""

    def __init__(self, chunks, error=None, finish="stop"):
        self.chunks, self.error, self.finish = chunks, error, finish
        self.calls = []

    def resolve_model(self):
        return "default-model"

    def stream_chat(self, messages, model=None, info=None, **kwargs):
        self.calls.append({"messages": messages, "model": model, **kwargs})
        yield from self.chunks
        if self.error:
            raise self.error
        info["finish_reason"] = self.finish


def run(llm, markdown="Run the installer.", title="Install", lang="ru", **kwargs):
    return list(stream_translation(title, markdown, llm, lang, **kwargs))


def text_of(events):
    return "".join(e["text"] for e in events if e["type"] == "delta")


def test_the_prompt_names_the_language_and_protects_code_and_images():
    system = system_prompt("ru")
    assert "Russian" in system and "[image:" in system and "never instructions" in system
    assert "\\" not in system  # no stray line continuations


def test_the_messages_carry_the_title_and_the_article():
    system, user = build_messages("Install", "Run `setup`.", "de")
    assert system["role"] == "system" and "German" in system["content"]
    assert user["role"] == "user" and "Install" in user["content"] and "Run `setup`." in user["content"]


def test_the_title_comes_first_then_the_text_without_the_heading_line():
    llm = FakeLLM(["# Уст", "ановка\n\nЗапустите", " установщик."])
    events = run(llm)
    assert [e["type"] for e in events][:1] == ["title"] and events[0]["text"] == "Установка"
    assert text_of(events).strip() == "Запустите установщик."
    assert events[-1]["type"] == "done" and events[-1]["truncated"] is False


def test_a_first_line_that_is_not_a_heading_stays_in_the_text():
    events = run(FakeLLM(["Запустите\nустановщик."]))
    assert not any(e["type"] == "title" for e in events)
    assert text_of(events) == "Запустите\nустановщик."


def test_an_answer_of_one_line_is_settled_when_the_stream_ends():
    assert run(FakeLLM(["# Установка"]))[0] == {"type": "title", "text": "Установка"}
    assert text_of(run(FakeLLM(["Запустите"]))) == "Запустите"


def test_an_error_before_the_first_line_break_drops_the_half_title():
    events = run(FakeLLM(["# Уст"], error=LLMError("boom")))
    assert events == [{"type": "error", "message": "boom"}]


def test_an_error_after_the_title_is_reported_after_the_text_so_far():
    events = run(FakeLLM(["# Установка\n\nЗапустите"], error=LLMError("boom")))
    assert [e["type"] for e in events] == ["title", "delta", "error"]


def test_a_cut_off_translation_is_flagged_truncated():
    events = run(FakeLLM(["# Установка\n\nЗапустите"], finish="length"))
    assert events[-1]["type"] == "done" and events[-1]["truncated"] is True


def test_the_model_and_a_language_scaled_token_limit_are_passed(monkeypatch):
    monkeypatch.setattr(config, "TRANSLATE_MAX_TOKENS", 1000)
    llm = FakeLLM(["# T\n\nx"])
    run(llm, model="claude-haiku-5-5")
    assert llm.calls[0]["model"] == "claude-haiku-5-5"
    assert llm.calls[0]["max_tokens"] == 1000 * config.LANGUAGE_TOKEN_FACTOR["ru"]


def test_an_article_without_text_needs_no_llm_call():
    llm = FakeLLM(["never used"])
    assert run(llm, markdown="  \n") == [{"type": "done", "model": None, "took_ms": 0, "truncated": False}]
    assert llm.calls == []


def test_english_is_refused_the_articles_are_already_english():
    with pytest.raises(ValueError, match="English"):
        run(FakeLLM([]), lang="en")
    with pytest.raises(ValueError, match="Unsupported"):
        run(FakeLLM([]), lang="xx")
