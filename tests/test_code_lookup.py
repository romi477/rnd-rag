import pytest
from test_code_graph import ABSTRACT, CORE_ORDER, SHOP_CLIENT, SHOP_ORDER, build
from test_code_index import FakeEmbedder
from test_odoo_module import REGISTRY_PY

from rag.code_graph import CodeGraph
from rag.code_index import index_module
from rag.code_lookup import find_error_code, related_methods, resolve_frames
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
        index_module(store, FakeEmbedder(), info.name, units, max_tokens=150, log=lambda m: None)
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


def test_a_line_outside_every_unit_is_reported_not_found(world):
    graph, store = world
    (hit,) = resolve_frames(store, tb(graph, ("demo_core/models/order.py", 999, "action_sync")))
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
    assert related_methods(graph, shop_hit, visible={"demo_shop"}).chain == []


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
