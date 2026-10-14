import json

import httpx
import pytest

from rag.sources import confluence
from rag.sources.confluence import collapse_blank_lines, storage_to_markdown, sync

# --- storage format -> markdown -----------------------------------------------------


def test_self_closing_parameter_does_not_swallow_text():
    html = ('<ac:structured-macro ac:name="toc"><ac:parameter ac:name="class" /></ac:structured-macro>'
            "<h2>First</h2><p>Kept one.</p><ac:parameter ac:name=\"x\">drop</ac:parameter><h2>Second</h2><p>Kept two.</p>")
    md = storage_to_markdown(html)
    assert "## First" in md and "Kept one." in md and "## Second" in md and "drop" not in md


def test_self_closing_image_and_code_macro():
    html = ('<p>A</p><ac:image ri:filename="x.png" /><p>B</p>'
            '<ac:structured-macro ac:name="code" /><p>C</p>')
    md = storage_to_markdown(html)
    assert "[image: x.png]" in md and "B" in md and "C" in md


def test_image_caption_is_kept():
    html = '<ac:image ac:alt="a"><ri:attachment ri:filename="s.png" /><ac:caption><p>Order settings</p></ac:caption></ac:image>'
    assert storage_to_markdown(html) == "[image: s.png]\n\n*Order settings*\n"


def test_adf_panel_without_attributes_or_duplicate():
    html = ('<ac:adf-extension><ac:adf-node type="panel"><ac:adf-attribute key="panel-type">note</ac:adf-attribute>'
            '<ac:adf-attribute key="local-id">0d87-uuid</ac:adf-attribute><ac:adf-content><p>Body text.</p></ac:adf-content>'
            '</ac:adf-node><ac:adf-fallback><div class="panel"><p>Body text.</p></div></ac:adf-fallback></ac:adf-extension>')
    md = storage_to_markdown(html)
    assert md.count("Body text.") == 1 and "uuid" not in md and md.startswith("**NOTE:**")


def test_underscores_titles_tasks_and_links():
    html = ('<p>Set my_field.</p>'
            '<ac:structured-macro ac:name="expand"><ac:parameter ac:name="title">Show details</ac:parameter>'
            "<ac:rich-text-body><p>Hidden.</p></ac:rich-text-body></ac:structured-macro>"
            "<ac:task-list><ac:task><ac:task-id>1</ac:task-id><ac:task-status>incomplete</ac:task-status>"
            "<ac:task-body>Buy milk</ac:task-body></ac:task></ac:task-list>"
            '<p>See <ac:link><ri:page ri:content-title="Setup Guide" /></ac:link> for details.</p>'
            '<p><ac:link><ri:page ri:content-title="X" /><ac:link-body>this page</ac:link-body></ac:link></p>')
    md = storage_to_markdown(html)
    assert "my_field" in md and "\\_" not in md
    assert "**Show details**" in md and "Hidden." in md
    assert "- [ ] Buy milk" in md and "1abc" not in md
    assert "See Setup Guide for details." in md and "this page" in md


def test_blank_lines_inside_code_are_kept():
    assert collapse_blank_lines("a\n\n\n\nb\n```\nx\n\n\n\ny\n```") == "a\n\nb\n```\nx\n\n\n\ny\n```"


# --- sync ---------------------------------------------------------------------------


def listing(n: int) -> dict:
    return {"results": [
        {"id": str(i), "title": f"P{i}", "parentId": None, "version": {"number": 1, "createdAt": "2026"},
         "_links": {"webui": f"/p/{i}"}, "body": {"storage": {"value": f"<p>Text {i}</p>"}}}
        for i in range(1, n + 1)
    ]}


@pytest.fixture
def server(monkeypatch):
    """A fake Confluence: `state["pages"]` pages; `state["fail"]` statuses to return first."""
    state = {"pages": 4, "fail": [], "calls": 0}

    def handler(request):
        state["calls"] += 1
        if state["fail"]:
            return httpx.Response(state["fail"].pop(0), headers={"Retry-After": "0"})
        if request.url.path.endswith("/spaces"):
            return httpx.Response(200, json={"results": [{"id": "100"}]})
        if request.url.path.endswith("/labels"):
            return httpx.Response(200, json={"results": []})
        return httpx.Response(200, json=listing(state["pages"]))

    real = httpx.Client

    def client(**kwargs):
        kwargs.pop("auth", None)
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(confluence.httpx, "Client", client)
    monkeypatch.setattr(confluence.time, "sleep", lambda s: None)
    monkeypatch.setenv("CONFLUENCE_EMAIL", "e@x")
    monkeypatch.setenv("CONFLUENCE_API_TOKEN", "t")
    return state


def test_sync_writes_pages_and_retries(server, tmp_path):
    server["fail"] = [429, 503]
    sync("sp", tmp_path)
    pages = tmp_path / "sp" / "pages"
    assert sorted(p.name for p in pages.glob("*.json")) == ["1.json", "2.json", "3.json", "4.json"]
    assert not list(pages.glob("*.tmp"))
    assert json.loads((tmp_path / "sp" / "_manifest.json").read_text())["1"]["version"] == 1


def test_sync_refuses_to_remove_most_pages(server, tmp_path):
    sync("sp", tmp_path)
    server["pages"] = 0
    with pytest.raises(SystemExit, match="Refusing"):
        sync("sp", tmp_path)
    assert len(list((tmp_path / "sp" / "pages").glob("*.json"))) == 4
    server["pages"] = 3
    sync("sp", tmp_path)  # one of four: a normal removal
    assert len(list((tmp_path / "sp" / "pages").glob("*.json"))) == 3


def test_pages_without_manifest_are_still_removed(server, tmp_path):
    sync("sp", tmp_path)
    (tmp_path / "sp" / "_manifest.json").unlink()  # e.g. a sync interrupted before the manifest
    server["pages"] = 3
    sync("sp", tmp_path)
    assert not (tmp_path / "sp" / "pages" / "4.md").exists()


# --- images of the pages ---------------------------------------------------------------


def test_referenced_images_are_the_attachments_shown_on_the_page():
    html = ('<ac:image><ri:attachment ri:filename="a.png" /></ac:image><p>x</p>'
            '<ac:image ac:alt="a"><ri:attachment ri:filename="a.png" /><ac:caption><p>c</p></ac:caption></ac:image>'
            '<ac:image ac:src="https://x/y.jpg"><ri:url ri:value="https://x/y.jpg" /></ac:image>'
            '<ac:link><ri:attachment ri:filename="doc.zip" /></ac:link>')
    assert confluence.referenced_images(html) == {"a.png"}


class FakeAttachments:
    """The attachment calls of the Confluence client: a listing and files named by the attachment title."""

    def __init__(self, titles):
        self.titles, self.downloads = titles, []

    def attachments(self, page_id):
        return [{"id": str(100 + i), "title": t, "downloadLink": f"/download/{t}", "version": {"number": 1}}
                for i, t in enumerate(self.titles)]

    def download(self, url, dest):
        self.downloads.append(url)
        dest.write_bytes(b"data")


def test_only_the_wanted_attachments_are_downloaded_once(tmp_path):
    api = FakeAttachments(["a.png", "b.zip", "c.png"])
    att = tmp_path / "att"
    confluence.sync_attachments(api, "1", att, only={"a.png"})
    assert sorted(p.name for p in att.iterdir()) == ["100-a.png", "_index.json"]
    assert json.loads((att / "_index.json").read_text())["100"]["title"] == "a.png"
    confluence.sync_attachments(api, "1", att, only={"a.png"})
    assert api.downloads == ["/wiki/download/a.png"]  # the second run found it


def test_an_attachment_the_page_no_longer_shows_is_removed(tmp_path):
    api = FakeAttachments(["a.png", "c.png"])
    att = tmp_path / "att"
    confluence.sync_attachments(api, "1", att, only={"a.png", "c.png"})
    confluence.sync_attachments(api, "1", att, only={"c.png"})
    assert sorted(p.name for p in att.iterdir()) == ["101-c.png", "_index.json"]


def test_without_a_filter_every_attachment_is_downloaded(tmp_path):
    api = FakeAttachments(["a.png", "b.zip"])
    confluence.sync_attachments(api, "1", tmp_path / "att")
    assert len(api.downloads) == 2
