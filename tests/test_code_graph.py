from pathlib import Path

import pytest

from rag.code_graph import CodeGraph, ModuleInfo
from rag.sources.odoo_module import DirSource, parse_module

CORE_ORDER = '''from odoo import models


class SaleOrder(models.Model):
    _inherit = 'sale.order'

    def action_sync(self):
        return True

    def other(self):
        return 1
'''

SHOP_ORDER = '''from odoo import models


class SaleOrder(models.Model):
    _inherit = 'sale.order'

    def action_sync(self):
        return super().action_sync()
'''

EXTRA_ORDER = '''from odoo import models


class SaleOrder(models.Model):
    _inherit = 'sale.order'

    def action_sync(self):
        return super().action_sync()
'''

ABSTRACT = '''import abc


class AbsApiClient(abc.ABC):
    @abc.abstractmethod
    def get_orders(self, since):
        """Fetch orders."""
'''

SHOP_CLIENT = '''class ShopClient(AbsApiClient):
    def get_orders(self, since):
        return []
'''

WOO_CLIENT = '''class WooClient(AbsApiClient):
    def get_orders(self, since):
        return []

    def get_products(self):
        return []
'''


def build(tmp_path: Path, name: str, depends: list[str], files: dict[str, str]) -> tuple[ModuleInfo, list]:
    root = tmp_path / name
    manifest = f"{{'name': '{name}', 'version': '19.0.1.0.0', 'depends': {depends!r}}}"
    for rel, text in {"__manifest__.py": manifest, **files}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    units = parse_module(DirSource(root), name)
    return ModuleInfo.from_units(name, units, origin="connector", ref="test"), units


@pytest.fixture
def modules(tmp_path):
    return {
        "demo_core": build(tmp_path, "demo_core", ["base", "sale"], {"models/order.py": CORE_ORDER, "api/abstract.py": ABSTRACT}),
        "demo_shop": build(tmp_path, "demo_shop", ["demo_core"], {"models/order.py": SHOP_ORDER, "api/client.py": SHOP_CLIENT}),
        "demo_woo": build(tmp_path, "demo_woo", ["demo_core"], {"api/client.py": WOO_CLIENT}),
        "demo_extra": build(tmp_path, "demo_extra", ["demo_shop"], {"models/order.py": EXTRA_ORDER}),
    }


def graph_of(modules, order: list[str]) -> CodeGraph:
    graph = CodeGraph()
    for name in order:
        graph.add_module(*modules[name])
    return graph


def test_module_info_reads_version_and_depends(modules):
    info = modules["demo_shop"][0]
    assert (info.name, info.version, info.depends, info.ref) == ("demo_shop", "19.0.1.0.0", ["demo_core"], "test")


def test_order_follows_depends_whatever_the_add_order(modules):
    first = graph_of(modules, ["demo_core", "demo_shop", "demo_woo", "demo_extra"]).order()
    shuffled = graph_of(modules, ["demo_extra", "demo_woo", "demo_shop", "demo_core"]).order()
    assert first == shuffled == ["demo_core", "demo_shop", "demo_woo", "demo_extra"]


def test_chain_lists_overrides_in_dependency_order(modules):
    graph = graph_of(modules, ["demo_extra", "demo_shop", "demo_core"])
    chain = graph.chain("sale.order", "action_sync")
    assert [(m.module, m.calls_super) for m in chain] == [
        ("demo_core", False), ("demo_shop", True), ("demo_extra", True),
    ]
    assert (chain[1].path, chain[1].line_start, chain[1].line_end) == ("models/order.py", 7, 8)


def test_chain_can_be_limited_to_visible_modules(modules):
    graph = graph_of(modules, ["demo_core", "demo_shop", "demo_extra"])
    chain = graph.chain("sale.order", "action_sync", only={"demo_core", "demo_extra"})
    assert [m.module for m in chain] == ["demo_core", "demo_extra"]


def test_chain_of_unknown_method_is_empty(modules):
    assert graph_of(modules, ["demo_core"]).chain("sale.order", "nope") == []


def test_models_record_defining_and_extending_modules(modules):
    graph = graph_of(modules, ["demo_core", "demo_shop"])
    assert [(d.module, d.extends) for d in graph.model_defs("sale.order")] == [
        ("demo_core", True), ("demo_shop", True),
    ]


def test_adapter_group_has_abstract_and_each_connectors_implementation(modules):
    group = graph_of(modules, ["demo_core", "demo_shop", "demo_woo"]).adapter_group("get_orders")
    assert [m.module for m in group.abstract] == ["demo_core"]
    assert sorted(group.impls) == ["demo_shop", "demo_woo"]
    assert group.impls["demo_woo"][0].class_name == "WooClient"
    only_woo = graph_of(modules, ["demo_core", "demo_woo"]).adapter_group("get_products")
    assert only_woo.abstract == [] and list(only_woo.impls) == ["demo_woo"]


def test_missing_depends_are_the_ones_not_added(modules):
    graph = graph_of(modules, ["demo_shop"])
    assert graph.missing_depends("demo_shop") == ["demo_core"]
    graph.add_module(*modules["demo_core"])
    assert graph.missing_depends("demo_shop") == []
    assert graph.missing_depends("demo_core") == ["base", "sale"]


def test_adding_a_module_again_replaces_it(modules):
    graph = graph_of(modules, ["demo_core", "demo_core"])
    assert len(graph.chain("sale.order", "action_sync")) == 1


def test_remove_module_drops_its_methods_and_models(modules):
    graph = graph_of(modules, ["demo_core", "demo_shop"])
    graph.remove_module("demo_shop")
    assert [m.module for m in graph.chain("sale.order", "action_sync")] == ["demo_core"]
    assert "demo_shop" not in graph.modules
    with pytest.raises(KeyError):
        graph.remove_module("demo_shop")


def test_save_and_load_roundtrip(modules, tmp_path):
    graph = graph_of(modules, ["demo_core", "demo_shop", "demo_woo"])
    path = tmp_path / "graph.json"
    graph.save(path)
    loaded = CodeGraph.load(path)
    assert loaded.order() == graph.order()
    assert loaded.chain("sale.order", "action_sync") == graph.chain("sale.order", "action_sync")
    assert sorted(loaded.adapter_group("get_orders").impls) == ["demo_shop", "demo_woo"]


def test_load_of_a_missing_file_is_an_empty_graph(tmp_path):
    assert CodeGraph.load(tmp_path / "none.json").modules == {}


def test_cli_add_list_chain_adapters_rm(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from rag import config
    from rag.cli import app

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    build(tmp_path, "demo_core", ["base"], {"models/order.py": CORE_ORDER, "api/abstract.py": ABSTRACT})
    build(tmp_path, "demo_shop", ["demo_core"], {"models/order.py": SHOP_ORDER, "api/client.py": SHOP_CLIENT})
    run = lambda *args: CliRunner().invoke(app, ["module", *args])  # noqa: E731

    refused = run("add", str(tmp_path / "demo_shop"), "--no-index")  # a version to read is required
    assert refused.exit_code == 1 and "--ref" in refused.output and "--working-tree" in refused.output
    res = run("add", str(tmp_path / "demo_shop"), "--no-index", "--working-tree")
    assert res.exit_code == 0 and "not added yet: demo_core" in res.output
    assert run("add", str(tmp_path / "demo_core"), "--origin", "core", "--no-index", "--working-tree").exit_code == 0

    listing = run("list").output
    assert "demo_core" in listing and "demo_shop" in listing and "19.0.1.0.0" in listing
    assert "demo_shop: not in the search index" in listing  # added with --no-index
    assert "run again without --no-index" in res.output  # `module add --no-index` says the index is not updated

    chain = run("chain", "sale.order", "action_sync").output
    assert chain.index("demo_core") < chain.index("demo_shop") and "super" in chain
    assert "statically" in chain

    adapters = run("adapters", "get_orders").output
    assert "abstract" in adapters and "ShopClient" in adapters

    blocked = run("rm", "demo_core")  # demo_shop depends on it
    assert blocked.exit_code == 1 and "demo_shop depend on demo_core" in blocked.output
    assert run("rm", "demo_shop").exit_code == 0
    assert "demo_shop" not in run("list").output
    assert run("rm", "demo_shop").exit_code == 1
    assert run("rm", "demo_core", "--force").exit_code == 0


def test_cli_add_indexes_and_search_corpus_code_finds_it(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from rag import cli, config
    from test_code_index import FakeEmbedder

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cli, "make_embedder", lambda model=None: FakeEmbedder())
    build(tmp_path, "demo_core", ["base"], {"models/order.py": CORE_ORDER})
    run = lambda *args: CliRunner().invoke(cli.app, list(args))  # noqa: E731

    res = run("module", "add", str(tmp_path / "demo_core"), "--working-tree")
    assert res.exit_code == 0, res.output
    assert "indexed" in res.output

    found = run("search", "action_sync", "--corpus", "code", "--mode", "fts")
    assert found.exit_code == 0, found.output
    assert "models/order.py:7" in found.output and "demo_core" in found.output
    listing = run("module", "list", "--model", "fake")
    assert listing.exit_code == 0 and "demo_core" in listing.output and "search index" not in listing.output

    assert run("module", "rm", "demo_core", "--model", "fake").exit_code == 0
    assert "No results" in run("search", "action_sync", "--corpus", "code", "--mode", "fts").output
