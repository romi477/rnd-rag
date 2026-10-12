import pytest
from test_code_graph import ABSTRACT, CORE_ORDER, SHOP_CLIENT, SHOP_ORDER, build
from test_code_index import FakeEmbedder
from test_odoo_module import REGISTRY_PY

from rag.code_graph import CodeGraph
from rag.code_index import index_module
from rag.code_lookup import find_error_code, related_methods, resolve_frames, scope_modules, sync_problems
from rag.code_store import CodeStore
from rag.traceback import parse_traceback

DECORATOR_PY = '''def deco(func):
    def wrapper(*args):
        return func(*args)
    return wrapper
'''

LONG_PY = "from odoo import models\n\n\nclass Big(models.Model):\n    _inherit = 'big.model'\n\n    def long_one(self):\n" + "".join(
    f"        value_{i} = compute(self, {i})\n" for i in range(150)
) + "        err = ErrorStore.raise_error(err_code='E514')\n        return value_0\n"


@pytest.fixture
def world(tmp_path):
    core = build(tmp_path, "demo_core", ["base"], {
        "models/order.py": CORE_ORDER, "api/abstract.py": ABSTRACT, "exceptions.py": REGISTRY_PY,
        "tools.py": DECORATOR_PY, "models/big.py": LONG_PY,
    })
    shop = build(tmp_path, "demo_shop", ["demo_core"], {"models/order.py": SHOP_ORDER, "api/client.py": SHOP_CLIENT})
    graph, store = CodeGraph(), CodeStore(tmp_path / "db")
    store.reset("fake", FakeEmbedder.dim)
    for info, units in (core, shop):
        graph.add_module(info, units)
        index_module(store, FakeEmbedder(), info.name, units, max_tokens=150, log=lambda m: None, ref=info.ref)
    return graph, store


def tb(graph, *frames: tuple[str, int, str], message="ValueError: boom"):
    lines = ["Traceback (most recent call last):"]
    lines += [f'  File "/opt/project/{path}", line {line}, in {func}\n    code' for path, line, func in frames]
    return parse_traceback("\n".join(lines + [message]), set(graph.modules))


def test_frames_resolve_to_units_innermost_first_with_the_whole_text(world):
    graph, store = world
    parsed = tb(graph, ("demo_shop/models/order.py", 8, "action_sync"), ("demo_core/models/order.py", 8, "action_sync"))
    found = resolve_frames(store, parsed)
    assert [(f.module, f.unit_id.split(":")[0]) for f in found] == [("demo_core", "demo_core"), ("demo_shop", "demo_shop")]
    top = found[0]
    assert (top.name, top.model, top.line_start, top.line_end, top.name_matches) == ("action_sync", "sale.order", 7, 8, True)
    assert top.text.startswith("def action_sync(self):") and "return True" in top.text


def test_a_line_outside_every_unit_with_an_unknown_name_is_reported_not_found(world):
    graph, store = world
    (hit,) = resolve_frames(store, tb(graph, ("demo_core/models/order.py", 999, "removed_since")))
    assert not hit.found and hit.text == ""


def test_nested_function_resolves_to_the_enclosing_unit_with_a_name_mismatch(world):
    graph, store = world
    (hit,) = resolve_frames(store, tb(graph, ("demo_core/tools.py", 3, "wrapper")))
    assert hit.found and hit.name == "deco" and hit.name_matches is False


def test_a_split_method_is_returned_whole_whichever_part_holds_the_line(world):
    graph, store = world
    (hit,) = resolve_frames(store, tb(graph, ("demo_core/models/big.py", 120, "long_one")))
    assert hit.text.count("\n") + 1 == hit.line_end - hit.line_start + 1
    assert "value_0 = compute(self, 0)" in hit.text and "return value_0" in hit.text
    assert (hit.line_start, hit.line_end) == (7, 7 + 1 + 150 + 1)


def test_related_methods_are_the_overrides_in_visible_modules_without_the_frame_itself(world):
    graph, store = world
    parsed = tb(graph, ("demo_shop/models/order.py", 8, "action_sync"), ("demo_core/models/order.py", 8, "action_sync"))
    shop_hit = next(f for f in resolve_frames(store, parsed) if f.module == "demo_shop")
    related = related_methods(graph, shop_hit, visible={"demo_core", "demo_shop"})
    assert [m.module for m in related.chain] == ["demo_core"]
    # what the module extends is always related, the traceback shows it or not
    assert [m.module for m in related_methods(graph, shop_hit, visible={"demo_shop"}).chain] == ["demo_core"]


def test_a_sibling_module_that_overrides_the_method_is_not_related_unless_visible(world, tmp_path):  # noqa: F811
    from test_code_graph import EXTRA_ORDER, build

    graph, store = world
    graph.add_module(*build(tmp_path, "demo_woo", ["demo_core"], {"models/order.py": EXTRA_ORDER}))
    parsed = tb(graph, ("demo_shop/models/order.py", 8, "action_sync"))
    (hit,) = resolve_frames(store, parsed)
    assert [m.module for m in related_methods(graph, hit, visible={"demo_shop"}).chain] == ["demo_core"]
    assert "demo_woo" in [m.module for m in related_methods(graph, hit, visible={"demo_shop", "demo_woo"}).chain]


def test_a_frame_whose_line_landed_in_another_function_is_resolved_by_name(world):  # noqa: F811
    graph, store = world
    # line 11 is inside `other` in this version; the client's version had action_sync there
    (hit,) = resolve_frames(store, tb(graph, ("demo_core/models/order.py", 11, "action_sync")))
    assert hit.found and hit.name == "action_sync" and hit.drifted is True
    assert (hit.line_start, hit.line_end) == (7, 8)


def test_a_frame_beyond_the_end_of_the_file_is_resolved_by_name_too(world):  # noqa: F811
    graph, store = world
    (hit,) = resolve_frames(store, tb(graph, ("demo_core/models/order.py", 999, "action_sync")))
    assert hit.found and hit.name == "action_sync" and hit.drifted is True


def test_a_matching_name_at_the_line_is_not_drift(world):  # noqa: F811
    graph, store = world
    (hit,) = resolve_frames(store, tb(graph, ("demo_core/models/order.py", 8, "action_sync")))
    assert hit.drifted is False


def test_same_named_units_resolve_to_the_one_nearest_the_line(world, tmp_path):  # noqa: F811
    from test_code_graph import build

    graph, store = world
    twice = "class A:\n    def run(self):\n        return 1\n\n\nclass B:\n    def run(self):\n        return 2\n"
    info, units = build(tmp_path, "demo_twice", ["demo_core"], {"twice.py": twice})
    graph.add_module(info, units)
    index_module(store, FakeEmbedder(), "demo_twice", units, max_tokens=150, log=lambda m: None, ref=info.ref)
    (hit,) = resolve_frames(store, tb(graph, ("demo_twice/twice.py", 90, "run")))
    assert hit.line_start == 7 and hit.drifted is True


def test_adapter_frame_brings_the_abstract_declaration_but_only_its_own_implementation(world):
    graph, store = world
    (hit,) = resolve_frames(store, tb(graph, ("demo_shop/api/client.py", 3, "get_orders")))
    related = related_methods(graph, hit, visible={"demo_shop"})
    assert [(m.module, m.class_name) for m in related.abstract] == [("demo_core", "AbsApiClient")]
    assert related.chain == []


def test_error_code_finds_the_registry_header_before_the_raise_sites(world):
    _, store = world
    found = find_error_code(store, "E514")
    assert [(u.kind, u.name) for u in found][:1] == [("class", "ErrorStore")]
    assert any(u.kind == "method" and u.name == "long_one" for u in found)
    assert find_error_code(store, "E999") == []


def test_cli_trace_prints_frames_units_related_codes_and_version_note(world, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from rag import config
    from rag.cli import app
    from rag.code_graph import graph_path
    from rag.store import store_path

    graph, store = world
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    graph.save(graph_path())
    CodeStore(store_path("fake")).reset("fake", FakeEmbedder.dim)  # the CLI opens the store of the model it is given
    # copy the indexed rows into the CLI's store
    target = CodeStore(store_path("fake"))
    target.table.add(store.table.to_arrow())
    target.write_meta(modules=store.indexed_modules())
    text = tmp_path / "tb.txt"
    text.write_text(
        'Traceback (most recent call last):\n  File "/opt/project/demo_shop/models/order.py", line 8, in action_sync\n'
        "    return super().action_sync()\n  File \"/opt/project/demo_core/models/order.py\", line 8, in action_sync\n    x\n"
        "odoo.exceptions.UserError: [E514] unpaid\n"
    )
    res = CliRunner().invoke(app, ["trace", str(text), "--model", "fake", "--client-version", "19.0.0.9.0"])
    assert res.exit_code == 0, res.output
    assert "demo_core/models/order.py:7-8" in res.output and "demo_shop/models/order.py:7-8" in res.output
    assert res.output.index("demo_core/models/order.py:7-8") < res.output.index("demo_shop/models/order.py:7-8")
    assert "E514" in res.output and "ErrorStore" in res.output
    assert "line numbers may not match" in res.output


NESTED_PY = "def run(self):\n    return 1\n"


def test_a_frame_path_with_two_readings_settles_on_the_one_that_is_indexed(world, tmp_path):  # noqa: F811
    graph, store = world
    for name, files in (("demo_flat", {"a.py": NESTED_PY}), ("demo_nest", {"demo_nest/a.py": NESTED_PY})):
        info, units = build(tmp_path, name, ["demo_core"], files)
        graph.add_module(info, units)
        index_module(store, FakeEmbedder(), name, units, max_tokens=150, log=lambda m: None, ref=info.ref)

    def resolve(path):
        parsed = parse_traceback(f'File "{path}", line 2, in run\n    x\nValueError: boom\n', set(graph.modules))
        (hit,) = resolve_frames(store, parsed)
        return hit, parsed.own_frames[0]

    # a repository folder called like the module, the module's files directly in it: the closest reading
    hit, frame = resolve("/srv/demo_flat/demo_flat/a.py")
    assert hit.found and (hit.module, hit.path) == ("demo_flat", "a.py") and frame.rel == "a.py"
    # the module has a folder of its own name: the closest reading is not in the index, the other one is
    hit, frame = resolve("/srv/demo_nest/demo_nest/a.py")
    assert hit.found and (hit.module, hit.path) == ("demo_nest", "demo_nest/a.py")
    assert (frame.module, frame.rel, frame.alternatives) == ("demo_nest", "demo_nest/a.py", [])  # changed in place
    # a folder named like another module above the addon does not matter
    hit, _ = resolve("/srv/demo_core/ext/demo_flat/a.py")
    assert hit.found and hit.module == "demo_flat"
    hit, _ = resolve("/srv/demo_flat/nothing.py")
    assert not hit.found


def test_error_code_prefers_the_modules_of_the_traceback_and_their_dependencies(world, tmp_path):  # noqa: F811
    graph, store = world
    other_registry = REGISTRY_PY.replace("unpaid invoice", "stock mismatch")
    info, units = build(tmp_path, "demo_other", ["base"], {"exceptions.py": other_registry})
    graph.add_module(info, units)
    index_module(store, FakeEmbedder(), "demo_other", units, max_tokens=150, log=lambda m: None, ref=info.ref)
    assert {u.module for u in find_error_code(store, "E514")} == {"demo_core", "demo_other"}
    scope = scope_modules(graph, {"demo_shop"})
    assert scope == {"demo_shop", "demo_core"}
    assert {u.module for u in find_error_code(store, "E514", scope)} == {"demo_core"}
    # nothing in the scope mentions the code: any module's units are better than none
    assert {u.module for u in find_error_code(store, "E514", {"demo_shop"})} == {"demo_core", "demo_other"}


def test_sync_problems_name_modules_whose_index_and_graph_differ(world, tmp_path):  # noqa: F811
    graph, store = world
    assert sync_problems(graph, store) == {}
    info, units = build(tmp_path, "demo_new", ["demo_core"], {"a.py": "x = 1\n"})
    graph.add_module(info, units)  # added with --no-index
    assert "not in the search index" in sync_problems(graph, store)["demo_new"]
    graph.modules["demo_shop"].ref = "other"  # the graph was updated to another ref than the rows hold
    problems = sync_problems(graph, store, only=["demo_shop", "demo_core"])
    assert set(problems) == {"demo_shop"} and "add the module again" in problems["demo_shop"]
    store.delete_module("demo_core")
    assert "demo_core" in sync_problems(graph, store)
    graph.remove_module("demo_shop")
    assert "not in the graph" in sync_problems(graph, store)["demo_shop"]
