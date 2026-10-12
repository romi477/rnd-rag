import pytest
import yaml
from test_code_index import FakeEmbedder
from test_code_lookup import world  # noqa: F401  (fixture)

from rag.code_search import CodeSearcher
from rag.trace_eval import Cause, TraceCase, evaluate_case, load_cases, summarize

TB = """Traceback (most recent call last):
  File "/opt/project/demo_shop/models/order.py", line 8, in action_sync
    return super().action_sync()
  File "/opt/project/demo_core/models/order.py", line 8, in action_sync
    ErrorStore.raise_error(err_code='E514')
odoo.exceptions.UserError: [E514] unpaid invoice
"""


def case(**kw) -> TraceCase:
    base = dict(id="t1", group="synthetic", traceback=TB, causes=[Cause("demo_core", "action_sync")])
    return TraceCase(**(base | kw))


def test_cause_found_at_rank_one_and_sent_to_the_llm(world):  # noqa: F811
    graph, store = world
    res = evaluate_case(case(), store, graph, None)
    assert (res.frames_found, res.frames_total) == (2, 2)
    assert res.cause_rank == 1 and res.cause_sent is True


def test_missing_cause_has_no_rank(world):  # noqa: F811
    graph, store = world
    res = evaluate_case(case(causes=[Cause("demo_core", "no_such_function")]), store, graph, None)
    assert res.cause_rank is None and res.cause_sent is False


def test_error_code_found_only_when_a_piece_carries_it(world):  # noqa: F811
    graph, store = world
    assert evaluate_case(case(codes=["E514"]), store, graph, None).codes_found is True
    assert evaluate_case(case(codes=["E999"]), store, graph, None).codes_found is False
    assert evaluate_case(case(), store, graph, None).codes_found is None  # nothing expected


def test_chain_needs_every_expected_module_among_the_cause_function_pieces(world):  # noqa: F811
    graph, store = world
    assert evaluate_case(case(chain=["demo_core", "demo_shop"]), store, graph, None).chain_ok is True
    assert evaluate_case(case(chain=["demo_core", "demo_other"]), store, graph, None).chain_ok is False
    assert evaluate_case(case(), store, graph, None).chain_ok is None


def test_question_only_case_uses_search(world):  # noqa: F811
    graph, store = world
    searcher = CodeSearcher(FakeEmbedder(), store)
    res = evaluate_case(case(traceback="", question="order sync", causes=[Cause("demo_core", "action_sync")]), store, graph, searcher)
    assert res.frames_total == 0 and res.cause_rank is not None


def test_summary_per_group_and_overall():
    from rag.trace_eval import CaseResult

    rows = [
        CaseResult("a", "real", 4, 4, 1, True, True, True),
        CaseResult("b", "real", 2, 1, 3, True, None, None),
        CaseResult("c", "drift", 2, 0, None, False, False, False),
    ]
    summary = summarize(rows)
    assert summary["real"] == {"n": 2, "frames": 5 / 6, "cause@1": 0.5, "cause@3": 1.0, "cause_sent": 1.0, "codes": 1.0, "chain": 1.0}
    assert summary["drift"]["cause@3"] == 0.0 and summary["drift"]["frames"] == 0.0
    assert summary["all"]["n"] == 3 and summary["all"]["cause@1"] == pytest.approx(1 / 3)


def test_load_cases_reads_yaml_and_rejects_a_case_without_cause(tmp_path):
    path = tmp_path / "traces.yaml"
    path.write_text(yaml.safe_dump([{
        "id": "t1", "group": "real", "traceback": TB, "cause": [{"module": "demo_core", "function": "action_sync"}],
        "codes": ["E514"], "chain": ["demo_core"], "question": "why?",
    }]))
    (c,) = load_cases(path)
    assert c.causes == [Cause("demo_core", "action_sync")] and c.codes == ["E514"] and c.chain == ["demo_core"] and c.question == "why?"
    path.write_text(yaml.safe_dump([{"id": "bad", "traceback": TB}]))
    with pytest.raises(ValueError, match="bad"):
        load_cases(path)
