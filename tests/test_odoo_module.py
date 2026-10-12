import subprocess
from pathlib import Path

import pytest

from rag.sources.odoo_module import (
    DirSource,
    GitSource,
    latest_tag,
    parse_module,
)

MANIFEST = """{
    'name': 'Demo Core',
    'summary': 'Core of the demo connector',
    'version': '19.0.1.2.3',
    'depends': ['base', 'sale'],
    'data': ['views/order_views.xml'],
}
"""

ORDER_PY = '''from odoo import fields, models


class SaleOrder(models.Model):
    """Order extension."""

    _inherit = 'sale.order'

    external_ref = fields.Char(
        string='External Ref',
        help='Reference of the order in the e-commerce platform.',
    )
    sync_state = fields.Selection(
        [('new', 'New'), ('done', 'Done')],
        string='Sync State',
        help='Where the order is in the sync.',
    )

    def action_sync(self, force=False):
        """Push the order to the platform."""
        self.ensure_one()
        return super().action_sync(force=force)

    def _helper(self):
        return 1
'''

ABSTRACT_PY = '''import abc


class AbsApiClient(abc.ABC):
    @abc.abstractmethod
    def get_orders(self, since):
        """Fetch orders created after ``since``."""
'''

SHOP_CLIENT_PY = '''from .abstract import AbsApiClient


class ShopClient(AbsApiClient):
    def get_orders(self, since):
        """Fetch Shop orders."""
        return self._get('/orders', since)

    def _get(self, url, since):
        return []
'''

REGISTRY_PY = '''class ErrorStore:
    """Registry of coded errors."""

    CODES = {
        'E514': 'unpaid invoice',  # soft failure
        'E516': 'no done picking',
    }

    def raise_error(self, code):
        raise ValueError(code)


class JustMethods:
    def run(self):
        return 1
'''

VIEWS_XML = """<?xml version="1.0" encoding="utf-8"?>
<odoo>
    <record id="view_order_form" model="ir.ui.view">
        <field name="name">sale.order.form.demo</field>
        <field name="model">sale.order</field>
        <field name="inherit_id" ref="sale.view_order_form"/>
        <field name="arch" type="xml">
            <xpath expr="//sheet" position="inside">
                <button name="action_sync" string="Sync Now" type="object"/>
                <field name="external_ref" help="Platform reference"/>
            </xpath>
        </field>
    </record>
    <record id="entity_order" model="demo.import.entity">
        <field name="name">Orders</field>
    </record>
</odoo>
"""

README_RST = "Demo Core\n=========\n\nSyncs orders with the platform.\n"
INDEX_HTML = (
    "<html><head><style>p{}</style></head><body>"
    "<h1>Demo</h1><p>Sync orders automatically.</p><script>var x = 1;</script></body></html>"
)


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def module(tmp_path) -> Path:
    root = tmp_path / "demo_core"
    write(root, "__manifest__.py", MANIFEST)
    write(root, "__init__.py", "from . import models\n")
    write(root, "models/order.py", ORDER_PY)
    write(root, "api/abstract.py", ABSTRACT_PY)
    write(root, "api/shop_client.py", SHOP_CLIENT_PY)
    write(root, "views/order_views.xml", VIEWS_XML)
    write(root, "exceptions.py", REGISTRY_PY)
    write(root, "README.rst", README_RST)
    write(root, "static/description/index.html", INDEX_HTML)
    write(root, "tests/test_order.py", "def test_x():\n    assert True\n")
    write(root, "migrations/19.0.1.0/post.py", "def migrate(cr, v):\n    pass\n")
    return root


def units(module: Path, **kw):
    return parse_module(DirSource(module), module.name, **kw)


def by_kind(items, kind):
    return [u for u in items if u.kind == kind]


def test_manifest_unit_has_name_summary_depends(module):
    (m,) = by_kind(units(module), "manifest")
    assert m.module == "demo_core"
    assert m.module_version == "19.0.1.2.3"
    assert "Demo Core" in m.text and "Core of the demo connector" in m.text
    assert m.extra["depends"] == ["base", "sale"]


def test_model_card_lists_fields_with_help_and_selection(module):
    (card,) = by_kind(units(module), "model")
    assert card.model == "sale.order"
    assert card.extra["extends"] is True
    assert card.line_start == 4
    assert "external_ref" in card.text
    assert "Reference of the order in the e-commerce platform." in card.text
    assert "done=Done" in card.text
    assert card.embed_text.startswith("demo_core > sale.order")


def test_method_unit_has_range_signature_docstring_and_super(module):
    items = units(module)
    (m,) = [u for u in by_kind(items, "method") if u.name == "action_sync"]
    assert (m.path, m.model) == ("models/order.py", "sale.order")
    assert m.line_start == 19 and m.line_end == 22
    assert m.text.startswith("def action_sync(self, force=False):")
    assert m.extra["calls_super"] is True
    assert m.extra["docstring"] == "Push the order to the platform."
    (helper,) = [u for u in by_kind(items, "method") if u.name == "_helper"]
    assert helper.extra["calls_super"] is False


def test_adapter_subclass_is_flagged_with_its_bases(module):
    items = units(module)
    (impl,) = [u for u in by_kind(items, "method") if u.name == "get_orders" and u.path.endswith("shop_client.py")]
    assert impl.extra["adapter"] is True
    assert impl.extra["bases"] == ["AbsApiClient"]
    assert impl.extra["class_name"] == "ShopClient"
    (abstract,) = [u for u in by_kind(items, "method") if u.name == "get_orders" and u.path.endswith("abstract.py")]
    assert abstract.extra["adapter"] is False
    assert abstract.extra["abstract"] is True


def test_view_unit_carries_visible_strings_and_lines(module):
    (view,) = by_kind(units(module), "view")
    assert (view.name, view.model, view.path) == ("view_order_form", "sale.order", "views/order_views.xml")
    assert (view.line_start, view.line_end) == (3, 13)
    assert "Sync Now" in view.embed_text and "action_sync" in view.embed_text
    assert "Platform reference" in view.embed_text
    assert "arch" not in view.embed_text.splitlines()  # the wrapper field is not a visible string
    assert '<button name="action_sync"' in view.text


def test_plain_record_is_a_data_unit(module):
    (rec,) = by_kind(units(module), "data")
    assert (rec.name, rec.model) == ("entity_order", "demo.import.entity")
    assert "Orders" in rec.text


def test_docs_are_plain_text(module):
    docs = {u.path: u for u in by_kind(units(module), "doc")}
    assert set(docs) == {"README.rst", "static/description/index.html"}
    html = docs["static/description/index.html"].text
    assert "Sync orders automatically." in html
    assert "var x" not in html and "<p>" not in html


def test_tests_and_migrations_are_skipped(module):
    paths = {u.path for u in units(module)}
    assert not any(p.startswith(("tests/", "migrations/")) for p in paths)


def test_broken_python_file_is_skipped_not_fatal(module):
    write(module, "models/broken.py", "def (:\n")
    assert by_kind(units(module), "manifest")


def test_origin_and_ids(module):
    items = units(module, origin="customer")
    assert {u.origin for u in items} == {"customer"}
    assert len({u.id for u in items}) == len(items)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_git_source_reads_the_ref_not_the_working_tree(tmp_path):
    repo = tmp_path / "repo"
    mod = repo / "demo_core"
    write(mod, "__manifest__.py", MANIFEST)
    write(mod, "models/order.py", ORDER_PY)
    git(repo.parent, "init", "-q", str(repo))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "init")
    git(repo, "tag", "core-19.0.1.2.3")
    write(mod, "models/order.py", ORDER_PY.replace("_helper", "_changed"))
    write(mod, "models/untracked.py", "class X:\n    def y(self):\n        pass\n")

    items = parse_module(GitSource(repo, "core-19.0.1.2.3", "demo_core"), "demo_core")
    names = {u.name for u in items if u.kind == "method"}
    assert "_helper" in names and "_changed" not in names
    assert not any(u.path == "models/untracked.py" for u in items)


def test_latest_tag_picks_highest_version_of_the_series(tmp_path):
    repo = tmp_path / "repo"
    write(repo, "a.txt", "x")
    git(repo.parent, "init", "-q", str(repo))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "init")
    for tag in ["core-19.0.2.9.0", "core-19.0.2.10.0", "core-20.0.9.0.0", "core-18.0.3.0.0", "shopify-19.0.9.9.9", "core-19.0.2.10.0-rc1"]:
        git(repo, "tag", tag)
    assert latest_tag(repo, "core", "19.0") == "core-19.0.2.10.0"
    assert latest_tag(repo, "missing", "19.0") is None


def test_cli_inspect_counts_units_and_dumps_json(module, tmp_path):
    import json

    from typer.testing import CliRunner

    from rag.cli import app

    out = tmp_path / "units.json"
    res = CliRunner().invoke(app, ["module", "inspect", str(module), "--json", str(out)])
    assert res.exit_code == 0, res.output
    assert "method" in res.output and "manifest" in res.output
    dumped = json.loads(out.read_text())
    assert {u["kind"] for u in dumped} >= {"manifest", "model", "method", "view", "data", "doc"}
    assert all("id" in u for u in dumped)


def test_plain_class_with_attributes_gets_a_class_unit_up_to_its_first_method(module):
    classes = by_kind(units(module), "class")
    assert [c.name for c in classes] == ["ErrorStore"]  # JustMethods and the AbsApiClient have no header worth a unit
    (reg,) = classes
    assert (reg.path, reg.line_start, reg.line_end) == ("exceptions.py", 1, 7)
    assert "'E514': 'unpaid invoice',  # soft failure" in reg.text and "def raise_error" not in reg.text
    assert reg.embed_text.startswith("demo_core > ErrorStore")


def test_model_classes_get_no_class_unit(module):
    assert all(c.name != "SaleOrder" for c in by_kind(units(module), "class"))
