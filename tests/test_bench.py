import pytest

from rag.bench import (
    Question, Run, load_questions, render_report, retrieval_line, run_bench, summarize, summary_rows,
)
from rag.llm import LLMError
from rag.search import ChunkHit, PageHit


def page(pid: str, title: str) -> PageHit:
    return PageHit(pid, title, f"https://x/{pid}", "Printing", ["Printing"], "2026-01-01", [], 1.0,
                   [ChunkHit(f"{pid}:0", [], "Some text.", 1.0)])


class FakeSearcher:
    def search(self, q, **kwargs):
        if "crash" in q:
            raise RuntimeError("index broken")
        return [] if "unknown" in q else [page("1", "Install")]


class FakeLLM:
    """Answers "<model> answer" and reports token usage; "broken" models fail on questions,
    "dead" ones already on warm-up, "silent" ones answer nothing, "long" ones hit max_tokens."""

    def __init__(self):
        self.calls = []

    def stream_chat(self, messages, model=None, info=None, **kwargs):
        self.calls.append((model, messages[-1]["content"]))
        if model == "dead" or (model == "broken" and "Question:" in messages[-1]["content"]):
            raise LLMError("model crashed")
        if model != "silent":
            yield from [f"{model} ", "answer"]
        info.update(usage={"prompt_tokens": 100, "completion_tokens": 2},
                    finish_reason="length" if model == "long" else "stop")


def qs(*texts: str) -> list[Question]:
    return [Question(t) for t in texts]


def test_load_questions_formats_and_group_filter(tmp_path):
    f = tmp_path / "q.yaml"
    f.write_text(
        '- plain question\n'
        '- q: eval item\n  relevant: ["1"]\n'
        '- q: novice item\n  group: novice\n  expect: [2]\n'
        '- q: not covered\n  expect: []\n'
        '- q: empty expect\n  expect:\n'
    )
    assert load_questions(f) == [Question("plain question"), Question("eval item", "", ["1"]),
                                 Question("novice item", "novice", ["2"]), Question("not covered", "", []),
                                 Question("empty expect")]
    assert [q.q for q in load_questions(f, "novice")] == ["novice item"]


def test_load_questions_unquoted_colon_is_a_clear_error(tmp_path):
    f = tmp_path / "q.yaml"
    f.write_text("- why do I get error: 403?\n")
    with pytest.raises(ValueError, match="item 1: no 'q' field"):
        load_questions(f)


def test_run_bench_alternates_models_and_records_runs():
    llm = FakeLLM()
    items = run_bench(FakeSearcher(), llm, qs("q1", "q2"), ["a", "b"])
    measured = [m for m, content in llm.calls if "Question:" in content]
    assert measured == ["a", "b", "b", "a"]  # warm-up calls are not counted
    run = items[0].runs["a"]
    assert run.text == "a answer" and run.completion_tokens == 2 and run.prompt_tokens == 100
    assert run.ttft_s is not None and run.total_s >= run.ttft_s and not run.error


def test_run_bench_skips_llm_without_sources_and_keeps_errors():
    items = run_bench(FakeSearcher(), FakeLLM(), qs("unknown thing", "q"), ["broken", "ok"])
    assert items[0].runs == {} and not items[0].sources
    assert items[1].runs["broken"].error == "model crashed" and items[1].runs["ok"].text == "ok answer"


def test_tok_per_s_excludes_time_to_first_token():
    assert Run("m", ttft_s=1.0, total_s=3.0, completion_tokens=100).tok_per_s == 50
    assert Run("m", total_s=1.0).tok_per_s is None


def test_summary_and_report():
    items = run_bench(FakeSearcher(), FakeLLM(), qs("q1", "unknown x"), ["a", "broken"])
    rows = summary_rows(items, ["a", "broken"])
    assert rows[1][0] == "a" and rows[1][-1] == "0"
    assert rows[2][0] == "broken" and rows[2][-1] == "1" and rows[2][1] == "– / –"
    report = render_report(items, ["a", "broken"], "settings line")
    assert "| a |" in report and "### 1. q1" in report and "a answer" in report
    assert "error: model crashed" in report and "the LLM was not called" in report


def test_expected_page_check():
    questions = [Question("q found", expect=["1"]), Question("q missing", expect=["9"]), Question("unknown q", expect=[]), Question("no expectation")]
    items = run_bench(FakeSearcher(), FakeLLM(), questions, ["a"])
    assert [it.expected_found for it in items] == [True, False, None, None]
    assert retrieval_line(items) == "Expected page among the sources: 1/2; 1 question(s) not covered by the docs"
    assert "expected page: **missing**" in render_report(items, ["a"], "")


def test_bench_keeps_results_when_a_search_fails_and_reports_progress():
    seen = []
    items = run_bench(FakeSearcher(), FakeLLM(), qs("q1", "crash me", "q3"), ["a"], on_item=lambda it: seen.append(len(it)))
    assert seen == [1, 2, 3]
    assert items[1].error == "search failed: index broken" and items[2].runs["a"].text == "a answer"
    assert "**search failed: index broken**" in render_report(items, ["a"], "", partial=True)


def test_bench_stops_when_a_model_fails_to_warm_up():
    with pytest.raises(LLMError, match="dead"):
        run_bench(FakeSearcher(), FakeLLM(), qs("q1"), ["a", "dead"])


def test_empty_and_cut_off_answers():
    items = run_bench(FakeSearcher(), FakeLLM(), qs("q1"), ["silent", "long"])
    assert items[0].runs["silent"].error == "empty answer"
    long = items[0].runs["long"]
    assert long.truncated and not long.error
    assert summarize(items, "long")["truncated"] == 1


def test_speed_is_tokens_over_total_time_not_mean_of_ratios():
    from rag.bench import Item
    item1, item2 = Item(Question("a"), []), Item(Question("b"), [])
    item1.runs["m"] = Run("m", text="x", ttft_s=0, total_s=1, gen_s=1, completion_tokens=10)
    item2.runs["m"] = Run("m", text="x", ttft_s=0, total_s=9, gen_s=9, completion_tokens=90)
    assert summarize([item1, item2], "m")["tok_per_s"] == 10


def test_report_closes_an_unclosed_code_fence():
    from rag.bench import as_markdown_block
    assert as_markdown_block("run:\n```bash\nls").endswith("ls\n```")
    assert as_markdown_block("```a```") == "```a```"
