from rag.bench import Question, Run, load_questions, render_report, retrieval_line, run_bench, summary_rows
from rag.llm import LLMError
from rag.search import ChunkHit, PageHit


def page(pid: str, title: str) -> PageHit:
    return PageHit(pid, title, f"https://x/{pid}", "Printing", ["Home", "Printing"], "2026-01-01", [], 1.0,
                   [ChunkHit(f"{pid}:0", [], "Some text.", 1.0)])


class FakeSearcher:
    def search(self, q, **kwargs):
        return [] if "unknown" in q else [page("1", "Install")]


class FakeLLM:
    """Answers "<model> answer" and reports token usage; "broken" models fail."""

    def __init__(self):
        self.calls = []

    def stream_chat(self, messages, model=None, usage=None, **kwargs):
        self.calls.append((model, messages[-1]["content"]))
        if model == "broken":
            raise LLMError("model crashed")
        yield from [f"{model} ", "answer"]
        usage.update(prompt_tokens=100, completion_tokens=2)


def qs(*texts: str) -> list[Question]:
    return [Question(t) for t in texts]


def test_load_questions_formats_and_group_filter(tmp_path):
    f = tmp_path / "q.yaml"
    f.write_text(
        '- plain question\n'
        '- q: eval item\n  relevant: ["1"]\n'
        '- q: novice item\n  group: novice\n  expect: [2]\n'
    )
    assert load_questions(f) == [Question("plain question"), Question("eval item", "", ["1"]),
                                 Question("novice item", "novice", ["2"])]
    assert [q.q for q in load_questions(f, "novice")] == ["novice item"]


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
    questions = [Question("q found", expect=["1"]), Question("q missing", expect=["9"]), Question("unknown q")]
    items = run_bench(FakeSearcher(), FakeLLM(), questions, ["a"])
    assert [it.expected_found for it in items] == [True, False, None]
    assert retrieval_line(items) == "Expected page among the sources: 1/2; 1 question(s) not covered by the docs"
    assert "expected page: **missing**" in render_report(items, ["a"], "")
