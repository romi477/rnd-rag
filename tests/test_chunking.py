from rag.chunking import (
    Page,
    approx_tokens,
    chunk_page,
    split_blocks,
    split_sections,
    strip_frontmatter,
)


def make_page(body: str, breadcrumbs=("Home", "Direct Print", "FAQ")) -> Page:
    return Page(
        id="42",
        title="How to Print",
        version=3,
        url="https://example/wiki/42",
        updated="2026-01-01T00:00:00Z",
        breadcrumbs=list(breadcrumbs),
        labels=[],
        body=body,
    )


def para(n_tokens: int, word: str = "lorem") -> str:
    return " ".join([word] * (n_tokens * 4 // (len(word) + 1)))


def test_strip_frontmatter_removes_meta_and_title():
    md = '---\nid: "42"\ntitle: "How to Print"\n---\n# How to Print\n\nBody text.\n'
    assert strip_frontmatter(md, "How to Print") == "Body text."


def test_split_sections_ignores_headings_inside_code():
    body = "Intro\n\n## Setup\n\n```bash\n# not a heading\n```\n\n### Details\n\nMore"
    sections = split_sections(body)
    assert [s.path for s in sections] == [[], ["Setup"], ["Setup", "Details"]]
    assert "# not a heading" in sections[1].content


def test_split_blocks_keeps_code_with_blank_lines_whole():
    text = "Para one\n\n```py\na = 1\n\nb = 2\n```\n\nPara two"
    assert split_blocks(text) == ["Para one", "```py\na = 1\n\nb = 2\n```", "Para two"]


def test_small_page_is_single_chunk_with_context_header():
    chunks = chunk_page(make_page("## A\n\n" + para(50) + "\n\n## B\n\n" + para(50)))
    assert len(chunks) == 1
    assert chunks[0].product == "Direct Print"
    assert chunks[0].embed_text.startswith("Direct Print > FAQ > How to Print\n\n")


def test_empty_parent_page_is_skipped():
    assert chunk_page(make_page("")) == []
    assert chunk_page(make_page("## Children\n\n[image: x.png]")) == []


def test_large_page_splits_by_section_and_merges_small_ones():
    body = "\n\n".join(
        [
            "## Install\n\n" + para(300),
            "## Configure\n\n" + para(300),
            "## Note\n\n" + para(20),  # small: merged into previous
        ]
    )
    chunks = chunk_page(make_page(body), max_tokens=400, min_tokens=80)
    assert [c.section_path for c in chunks] == [["Install"], []]
    assert "## Configure" in chunks[1].text and "## Note" in chunks[1].text
    assert all(c.n_tokens <= 400 for c in chunks)


def test_oversized_section_split_on_paragraphs_code_kept_whole():
    code = "```bash\n" + "\n".join(f"echo {i}" for i in range(200)) + "\n```"
    body = "## Big\n\n" + "\n\n".join([para(150), para(150), code, para(150)])
    chunks = chunk_page(make_page(body), max_tokens=400, min_tokens=80)
    assert len(chunks) > 1
    assert all(c.section_path == ["Big"] for c in chunks)
    assert sum(c.text.count("```") for c in chunks) == 2
    assert any(c.text.startswith("```bash") and c.text.endswith("```") for c in chunks)
    assert all("Big" in c.embed_text.split("\n\n")[0] for c in chunks)


def test_image_markers_not_embedded():
    chunks = chunk_page(make_page("Text about labels " + para(30) + "\n\n[image: screen.png]"))
    assert "[image:" in chunks[0].text
    assert "[image:" not in chunks[0].embed_text


def test_approx_tokens():
    assert approx_tokens("abcd" * 10) == 10


def test_heading_only_section_attaches_to_following_content():
    body = "\n\n".join(
        [
            "## Causes\n\n" + para(300),
            "## How to Fix It",
            "### Option 1\n\n" + para(300),
        ]
    )
    chunks = chunk_page(make_page(body), max_tokens=400, min_tokens=80)
    assert chunks[0].text.rstrip().endswith("lorem")
    assert chunks[1].text.startswith("## How to Fix It\n\n### Option 1")
    assert chunks[1].section_path == ["How to Fix It"]


def test_heading_emphasis_is_removed_from_path():
    sections = split_sections("# **Common Case:** `ref_field`\n\ntext")
    assert sections[0].path == ["Common Case: `ref_field`"]


def test_no_tiny_trailing_piece():
    body = "## Big\n\n" + "\n\n".join([para(300), para(300), para(20)])
    chunks = chunk_page(make_page(body), max_tokens=400, min_tokens=80)
    assert len(chunks) == 2
    assert min(c.n_tokens for c in chunks) >= 80


def test_short_parent_intro_attaches_to_children():
    body = "\n\n".join(
        [
            "## Causes\n\n" + para(300),
            "## How to Fix It\n\nShort intro.",
            "### Option 1\n\n" + para(300),
        ]
    )
    chunks = chunk_page(make_page(body), max_tokens=400, min_tokens=80)
    assert chunks[1].text.startswith("## How to Fix It\n\nShort intro.\n\n### Option 1")
