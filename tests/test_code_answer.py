import pytest
from test_answer import FakeLLM
from test_code_index import FakeEmbedder
from test_code_lookup import world  # noqa: F401  (fixture)

from rag.chunking import approx_tokens
from rag.code_answer import (
    Piece,
    build_code_context,
    build_code_messages,
    code_fence,
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


def test_window_of_a_line_outside_the_text_shows_the_nearest_end_not_nothing():
    # a frame matched by function name carries the client's line number, which may be beyond or before this version
    text = "\n".join(f"line {i}" for i in range(1, 51))  # a function at file lines 100..149
    after = window(text, first_line=100, failing_line=400, radius=5)
    assert "line 50" in after and "line 45" in after and "line 1\n" not in after and after.startswith("# ...")
    before = window(text, first_line=100, failing_line=20, radius=5)
    assert "line 1\n" in before and "line 6" in before and "line 50" not in before and before.rstrip().endswith("# ...")
    assert window("", 1, 5, 3) == ""


def test_window_and_signature_count_lines_like_the_parser_not_like_splitlines():
    text = "def f(self):\n    s = 'a\u2028b'\n    return s\n" + "\n".join(f"    y{i} = {i}" for i in range(30))
    assert "return s" in window(text, 1, 3, 1)
    assert signature_and_doc("def f(self):\n    '''a\u2028b'''\n    return 1") == "def f(self):\n    '''a\u2028b'''\n    ..."


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


def test_a_function_named_in_the_question_comes_first_with_its_whole_body(world):  # noqa: F811
    graph, store = world
    ev = gather(store, graph, 'How does "long_one" work?', "", searcher=CodeSearcher(FakeEmbedder(), store))
    first = ev.pieces[0]
    assert (first.role, first.module, first.name) == ("definition", "demo_core", "long_one")
    assert "def long_one(self):" in first.text and "value_149 = compute(self, 149)" in first.text
    assert "return value_0" in first.text  # the end of the function: nothing was cut
    assert first.label == "big.model.long_one"
    assert sum(p.role == "search" for p in ev.pieces) <= 2  # the named function already points at the code


def test_named_definitions_after_the_first_ones_are_signatures_and_a_long_one_is_cut(world, monkeypatch):  # noqa: F811
    from rag import code_answer

    graph, store = world
    monkeypatch.setattr(code_answer, "DEFINITIONS_FULL", 1)
    ev = gather(store, graph, "explain action_sync", "", searcher=None)
    core, shop = [p for p in ev.pieces if p.role == "definition"]
    assert "return True" in core.text  # the first definition in full
    assert shop.text.endswith("    ...") and "super().action_sync()" not in shop.text  # the next one as its signature
    monkeypatch.setattr(code_answer, "DEFINITION_MAX_TOKENS", 50)
    (long_one,) = [p for p in gather(store, graph, "long_one", "", searcher=None).pieces if p.role == "definition"]
    assert "value_0 = compute(self, 0)" in long_one.text and "value_149" not in long_one.text
    assert long_one.text.endswith("(the rest of the function is not shown)")


def test_a_named_function_that_is_a_traceback_frame_is_not_quoted_twice(world):  # noqa: F811
    graph, store = world
    ev = gather(store, graph, "what does action_sync do?", TRACEBACK, searcher=None)
    spans = [(p.module, p.path, p.line_start) for p in ev.pieces]
    assert len(spans) == len(set(spans))
    assert not any(p.role == "definition" for p in ev.pieces)  # both definitions are the frames already


def test_without_a_traceback_the_prompt_says_to_explain_the_code_step_by_step(world):  # noqa: F811
    graph, store = world
    asked = gather(store, graph, "how does long_one work?", "", searcher=None)
    _, context = build_code_context(asked.pieces)
    user = build_code_messages("how does long_one work?", asked, context)[1]["content"]
    assert "No traceback was given" in user and "step by step" in user and user.endswith("Question: how does long_one work?")
    failed = gather(store, graph, "", TRACEBACK, searcher=None)
    _, context = build_code_context(failed.pieces)
    assert "No traceback" not in build_code_messages("", failed, context)[1]["content"]


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


def test_a_fence_is_longer_than_any_backtick_run_in_the_text():
    assert code_fence("x = 1") == "```" and code_fence("a ``` b") == "````" and code_fence("`` `````") == "``````"
    _, context = build_code_context([piece(1, "Use:\n```\nrun\n```\nafter the block")], max_tokens=1000)
    assert context.count("````") == 2 and "```\nrun\n```" in context
    # the language follows the file, and a README is plain text
    py = Piece("frame", "u", "m", "1", "models/a.py", 1, 2, "f", "x = 1")
    xml = Piece("search", "u", "m", "1", "views/a.xml", 1, 2, "f", "<a/>")
    rst = Piece("search", "u", "m", "1", "README.rst", 1, 2, "f", "text")
    _, context = build_code_context([py, xml, rst], max_tokens=1000)
    assert "```python\nx = 1\n```" in context and "```xml\n<a/>\n```" in context and "```\ntext\n```" in context


def test_a_module_whose_index_differs_from_the_graph_gets_a_note(world):  # noqa: F811
    graph, store = world
    graph.modules["demo_shop"].ref = "moved"
    ev = gather(store, graph, "", TRACEBACK, searcher=None)
    assert any(n.startswith("demo_shop: the index and the graph differ") for n in ev.notes)
    assert not any(n.startswith("demo_core: the index") for n in ev.notes)


def test_error_codes_come_from_the_modules_of_the_traceback_first(world, tmp_path):  # noqa: F811
    from test_code_graph import build
    from test_odoo_module import REGISTRY_PY

    from rag.code_index import index_module

    graph, store = world
    info, units = build(tmp_path, "demo_other", ["base"], {"exceptions.py": REGISTRY_PY.replace("unpaid invoice", "other meaning")})
    graph.add_module(info, units)
    index_module(store, FakeEmbedder(), "demo_other", units, max_tokens=150, log=lambda m: None, ref=info.ref)
    ev = gather(store, graph, "", TRACEBACK, searcher=None)
    assert {p.module for p in ev.pieces if p.role == "error code"} == {"demo_core"}


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


def test_a_module_filter_still_finds_a_named_function_of_the_modules_it_depends_on(world):  # noqa: F811
    graph, store = world
    # long_one is defined in demo_core only; demo_shop depends on it
    ev = gather(store, graph, "how does long_one work?", "", searcher=None, modules=["demo_shop"])
    assert [(p.role, p.module) for p in ev.pieces] == [("definition", "demo_core")]
    both = gather(store, graph, "how does action_sync work?", "", searcher=None, modules=["demo_shop"])
    assert [(p.role, p.module) for p in both.pieces] == [("definition", "demo_core"), ("definition", "demo_shop")]
    # a module that is not in the graph filters nothing in, and does not break the lookup
    assert gather(store, graph, "how does long_one work?", "", searcher=None, modules=["nope"]).pieces == []


# --- repository documentation (kind "guide") --------------------------------------------------------------------


@pytest.fixture
def with_guides(world, tmp_path):  # noqa: F811
    from rag.code_index import index_module
    from rag.sources.guide_docs import parse_guides
    from rag.sources.odoo_module import DirSource

    graph, store = world
    docs = tmp_path / "project_docs"
    docs.mkdir()
    (docs / "SUPPORT.md").write_text(
        "# Support\n\n## Error codes\n\nE514 means the client has an unpaid invoice: collect the payment and retry the order sync.\n\n"
        "## Carriers\n\nThe carrier is set on the order before sync.\n", encoding="utf-8",
    )
    units = parse_guides(DirSource(docs), "project")
    index_module(store, FakeEmbedder(), "docs:project", units, max_tokens=150, log=lambda m: None, ref="19.0")
    return graph, store


def guide_pieces(ev):
    return [p for p in ev.pieces if p.role == "guide"]


def test_a_guide_that_mentions_the_error_code_comes_right_after_the_code_registry(with_guides):
    graph, store = with_guides
    ev = gather(store, graph, "", TRACEBACK, searcher=None)
    roles = [p.role for p in ev.pieces]
    (guide,) = guide_pieces(ev)
    assert (guide.module, guide.path, guide.label) == ("docs:project", "docs/SUPPORT.md", "Support > Error codes")
    assert "unpaid invoice" in guide.text
    assert roles.index("guide") > roles.index("error code")
    assert not any(p.role == "error code" and p.module == "docs:project" for p in ev.pieces)


def test_search_adds_the_best_guides_even_when_a_module_is_chosen(with_guides):
    graph, store = with_guides
    searcher = CodeSearcher(FakeEmbedder(), store)
    for modules in (None, ["demo_shop"]):
        ev = gather(store, graph, "which carrier is used for the order sync", "", searcher=searcher, modules=modules)
        assert any(p.module == "docs:project" for p in guide_pieces(ev))
    ev = gather(store, graph, "which carrier is used for the order sync", "", searcher=searcher)
    assert not any(p.role == "search" and p.module.startswith("docs:") for p in ev.pieces)


def test_guides_are_numbered_and_described_for_the_llm(with_guides):
    graph, store = with_guides
    ev = gather(store, graph, "", TRACEBACK, searcher=None)
    sources, context = build_code_context(ev.pieces)
    assert any(s.role == "guide" and s.module == "docs:project" for s in sources)
    assert "docs:project/docs/SUPPORT.md:" in context and "documentation of the repository" in context


def test_the_index_of_a_docs_source_is_not_reported_as_missing_from_the_graph(with_guides):
    from rag.code_lookup import sync_problems

    graph, store = with_guides
    assert sync_problems(graph, store) == {}
