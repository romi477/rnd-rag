from test_answer import FakeLLM
from test_code_index import FakeEmbedder
from test_code_lookup import world  # noqa: F401  (fixture)

from rag.chunking import approx_tokens
from rag.code_answer import (
    Piece,
    build_code_context,
    gather,
    signature_and_doc,
    stream_code_answer,
    window,
)
from rag.code_search import CodeSearcher

TRACEBACK = """Traceback (most recent call last):
  File "/opt/external-addons/queue_lib/job.py", line 5, in perform
    self.func()
  File "/opt/project/demo_shop/models/order.py", line 8, in action_sync
    return super().action_sync()
  File "/opt/project/demo_core/models/order.py", line 8, in action_sync
    ErrorStore.raise_error(err_code='E514')
odoo.exceptions.UserError: [E514] unpaid invoice
"""


def piece(n: int, text: str, role: str = "search") -> Piece:
    return Piece(role, f"u{n}:a{n}.py:1:method", f"u{n}", "19.0.1", f"a{n}.py", 1, 5, f"unit {n}", text)


# --- text helpers -----------------------------------------------------------------------------------------------


def test_signature_and_doc_keeps_signature_and_docstring_only():
    text = '@api.model\ndef f(self, a,\n      b=1):\n    """Do it."""\n    x = 1\n    return x'
    assert signature_and_doc(text) == '@api.model\ndef f(self, a,\n      b=1):\n    """Do it."""\n    ...'


def test_signature_and_doc_without_docstring():
    assert signature_and_doc("def f(self):\n    x = 1\n    return x") == "def f(self):\n    ..."


def test_signature_and_doc_falls_back_to_the_first_lines_on_a_syntax_error():
    text = "    part of a method\n" * 12
    assert signature_and_doc(text).count("\n") <= 5


def test_window_keeps_lines_around_the_failing_one_and_marks_the_cut():
    text = "\n".join(f"line {i}" for i in range(1, 101))
    cut = window(text, first_line=1000, failing_line=1049, radius=5)
    assert "line 50" in cut and "line 45" in cut and "line 55" in cut
    assert "line 10\n" not in cut and "line 90" not in cut
    assert cut.startswith("# ...") and cut.rstrip().endswith("# ...")
    assert window("a\nb", 1, 1, 5) == "a\nb"


# --- gather ------------------------------------------------------------------------------------------------------


def test_gather_orders_pieces_inner_frames_callers_overrides_then_error_codes(world):  # noqa: F811
    graph, store = world
    ev = gather(store, graph, "", TRACEBACK, searcher=None)
    assert ev.tb.exception.short_name == "UserError"
    roles = [(p.role, p.module) for p in ev.pieces]
    assert roles[0] == ("frame", "demo_core") and roles[1] == ("frame", "demo_shop")
    assert ("error code", "demo_core") in roles
    assert roles.index(("error code", "demo_core")) > 1
    assert "def action_sync(self):" in ev.pieces[0].text and "return True" in ev.pieces[0].text


def test_callers_come_as_signature_plus_the_lines_around_the_call(world, monkeypatch):  # noqa: F811
    from rag import config

    graph, store = world
    monkeypatch.setattr(config, "CODE_INNER_FRAMES", 1)
    text = (
        'Traceback (most recent call last):\n  File "/opt/project/demo_core/models/big.py", line 120, in long_one\n'
        "    value_112 = compute(self, 112)\n"
        '  File "/opt/project/demo_core/models/order.py", line 8, in action_sync\n    return True\n'
        "ValueError: boom\n"
    )
    ev = gather(store, graph, "", text, searcher=None)
    inner, caller = ev.pieces[0], ev.pieces[1]
    assert (inner.role, caller.role) == ("frame", "caller")
    assert "return True" in inner.text
    assert "def long_one(self):" in caller.text and "value_112 = compute(self, 112)" in caller.text
    assert "value_109" in caller.text and "value_115" in caller.text
    assert "value_0 = compute" not in caller.text and "value_149" not in caller.text


def test_a_frame_outside_the_index_becomes_a_note_not_a_piece(world):  # noqa: F811
    graph, store = world
    text = TRACEBACK.replace("line 8, in action_sync\n    return super", "line 999, in removed_since\n    return super")
    ev = gather(store, graph, "", text, searcher=None)
    assert any("demo_shop/models/order.py:999" in n for n in ev.notes)
    assert [p.module for p in ev.pieces if p.role.startswith("frame")] == ["demo_core"]


def test_version_note_for_each_module_in_the_traceback(world):  # noqa: F811
    graph, store = world
    ev = gather(store, graph, "", TRACEBACK, searcher=None, client_version="19.0.0.1.0")
    assert sum("line numbers may not match" in n for n in ev.notes) == 2


def test_question_only_uses_search_hits(world):  # noqa: F811
    graph, store = world
    ev = gather(store, graph, "how does sync work with the client", "", searcher=CodeSearcher(FakeEmbedder(), store))
    assert ev.tb is None and ev.pieces and {p.role for p in ev.pieces} == {"search"}


def test_search_hits_do_not_repeat_units_already_in_the_context(world):  # noqa: F811
    graph, store = world
    ev = gather(store, graph, "", TRACEBACK, searcher=CodeSearcher(FakeEmbedder(), store))
    ids = [p.unit_id for p in ev.pieces]
    assert len(ids) == len(set(ids))


def test_text_that_is_not_a_traceback_is_ignored(world):  # noqa: F811
    graph, store = world
    assert gather(store, graph, "", "just some words", searcher=None).pieces == []


# --- context and prompt ------------------------------------------------------------------------------------------


def test_context_numbers_sources_and_describes_them():
    sources, context = build_code_context([piece(1, "x = 1", "frame"), piece(2, "y = 2")], max_tokens=1000)
    assert [s.n for s in sources] == [1, 2]
    assert sources[0].kind == "code" and (sources[0].module, sources[0].path, sources[0].line_start) == ("u1", "a1.py", 1)
    assert "[1] u1/a1.py:1-5" in context and "[2] u2/a2.py:1-5" in context and "x = 1" in context


def test_context_skips_what_does_not_fit_and_truncates_the_first_oversized_piece():
    big = "z = 1\n" * 2000
    sources, context = build_code_context([piece(1, big), piece(2, "small = 1")], max_tokens=300)
    assert [s.n for s in sources] == [1, 2] or [s.module for s in sources] == ["u1"]
    assert approx_tokens(context) <= 400 and "# ... truncated" in context


def test_context_drops_a_later_piece_that_does_not_fit():
    sources, _ = build_code_context([piece(1, "a = 1\n" * 300), piece(2, "b = 2\n" * 300)], max_tokens=400)
    assert [s.module for s in sources] == ["u1"]


# --- streaming ---------------------------------------------------------------------------------------------------


def test_stream_code_answer_sends_failure_and_sources_to_the_llm(world):  # noqa: F811
    graph, store = world
    ev = gather(store, graph, "", TRACEBACK, searcher=None)
    llm = FakeLLM(["The invoice is unpaid [1]."])
    events = list(stream_code_answer("Why?", ev, llm))
    assert [e["type"] for e in events] == ["sources", "delta", "done"]
    assert events[0]["sources"][0]["kind"] == "code"
    system, user = llm.messages[0]["content"], llm.messages[1]["content"]
    assert "support engineer" in system
    assert "Failure:" in user and "[E514] unpaid invoice" in user and "Question: Why?" in user
    assert "demo_core/models/order.py:7-8" in user


def test_stream_code_answer_without_evidence_says_so_and_does_not_call_the_llm(world):  # noqa: F811
    graph, store = world
    ev = gather(store, graph, "", "nothing", searcher=None)
    llm = FakeLLM(["never"])
    events = list(stream_code_answer("Why?", ev, llm))
    assert [e["type"] for e in events] == ["sources", "delta", "done"] and llm.messages is None
    assert "Nothing relevant" in events[1]["text"]


def test_external_llm_is_blocked_for_code_unless_allowed(monkeypatch):
    from rag import config
    from rag.code_answer import external_llm_blocked

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    assert external_llm_blocked() is None
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "CODE_EXTERNAL_LLM", False)
    assert "RAG_CODE_EXTERNAL_LLM" in external_llm_blocked()
    monkeypatch.setattr(config, "CODE_EXTERNAL_LLM", True)
    assert external_llm_blocked() is None


def test_a_frame_resolved_by_name_after_a_line_shift_gets_a_note_and_a_piece(world):  # noqa: F811
    graph, store = world
    text = TRACEBACK.replace("line 8, in action_sync\n    return super", "line 999, in action_sync\n    return super")
    ev = gather(store, graph, "", text, searcher=None)
    assert any("demo_shop/models/order.py:999" in n and "function name" in n for n in ev.notes)
    assert ("frame", "demo_shop") in [(p.role, p.module) for p in ev.pieces]


def test_search_adds_fewer_hits_when_the_frames_already_point_at_the_code(world):  # noqa: F811
    from rag.code_answer import SEARCH_K_WITH_FRAMES

    graph, store = world
    searcher = CodeSearcher(FakeEmbedder(), store)
    with_frames = gather(store, graph, "", TRACEBACK, searcher=searcher)
    assert sum(p.role == "search" for p in with_frames.pieces) <= SEARCH_K_WITH_FRAMES
    assert SEARCH_K_WITH_FRAMES < 6  # the default number of hits without frames
