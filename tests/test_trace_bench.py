from test_answer import FakeLLM
from test_code_index import FakeEmbedder
from test_code_lookup import world  # noqa: F401  (fixture)

from rag.code_search import CodeSearcher
from rag.trace_bench import answer_checks, render_report, run_trace_bench
from rag.trace_eval import Cause, TraceCase

TB = """Traceback (most recent call last):
  File "/opt/project/demo_shop/models/order.py", line 8, in action_sync
    return super().action_sync()
  File "/opt/project/demo_core/models/order.py", line 8, in action_sync
    ErrorStore.raise_error(err_code='E514')
odoo.exceptions.UserError: [E514] unpaid invoice
"""
class Llm(FakeLLM):
    """FakeLLM that also takes the extra keywords `generate` passes (max_tokens for the warm-up)."""

    def stream_chat(self, messages, model=None, info=None, **kwargs):
        yield from super().stream_chat(messages, model=model, info=info)


CASE = TraceCase("t1", "synthetic", TB, [Cause("demo_core", "action_sync")], codes=["E514"])


def test_checks_count_valid_citations_loose_ones_and_the_cause_mention():
    checks = answer_checks("It fails in `action_sync` [1][2]. See [Source 3] and [9].", CASE, n_sources=3)
    assert checks == {"cause_named": True, "citations": 2, "loose_citations": 1, "unknown_citations": 1}
    assert answer_checks("Something else [1].", CASE, n_sources=3)["cause_named"] is False


def test_bench_runs_every_case_on_every_model_and_alternates_order(world):  # noqa: F811
    graph, store = world
    llm = Llm(["Fails in action_sync [1]."])
    order = []
    original = llm.stream_chat
    llm.stream_chat = lambda messages, model=None, info=None, **kw: (order.append(model), original(messages, model=model, info=info))[1]
    items = run_trace_bench([CASE, CASE], ["m1", "m2"], store, graph, CodeSearcher(FakeEmbedder(), store), llm)
    assert [sorted(i.runs) for i in items] == [["m1", "m2"], ["m1", "m2"]]
    assert order[2:] == ["m1", "m2", "m2", "m1"] or order[2:] == ["m2", "m1", "m1", "m2"]  # after the two warm-ups
    assert items[0].sources and items[0].runs["m1"].text.startswith("Fails in")


def test_report_has_a_summary_per_model_the_sources_and_the_answers(world):  # noqa: F811
    graph, store = world
    items = run_trace_bench([CASE], ["m1"], store, graph, None, Llm(["Fails in action_sync [1]."]))
    report = render_report(items, ["m1"], "settings line")
    assert "## Summary" in report and "| m1 |" in report
    assert "### 1. t1" in report and "Sources:" in report and "Fails in action_sync [1]." in report
    assert "cause named" in report
