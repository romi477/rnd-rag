import subprocess
from pathlib import Path

import pytest
from test_code_index import FakeEmbedder

from rag.code_docs import DocsRegistry, DocsSourceError, add_source, remove_source, sync_sources
from rag.code_store import CodeStore
from rag.sources.guide_docs import GUIDE_KIND, DOCS_PREFIX, docs_module, parse_guides, resolve_docs_dir
from rag.sources.odoo_module import DirSource, GitSource

SUPPORT = """\
# Support

How to triage a report.

## Error codes

Every failure carries a code such as E514 (unpaid invoice).

```text
# not a heading
E514 raised by ErrorStore
```

### Look one up

grep the code.

## Empty section

---

## Logs

Read the job log first.
"""


def write(root: Path, rel: str, text: str) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text, encoding="utf-8")


def parse(files: dict[str, str], tmp_path: Path, name="project", directory="docs"):
    for rel, text in files.items():
        write(tmp_path, rel, text)
    return parse_guides(DirSource(tmp_path), name, directory)


def test_docs_module_name_marks_the_source_as_documentation():
    assert docs_module("project") == DOCS_PREFIX + "project" == "docs:project"


def test_sections_become_units_with_their_heading_path_and_file_lines(tmp_path):
    units = parse({"SUPPORT.md": SUPPORT}, tmp_path)
    by_name = {u.name: u for u in units}
    assert set(by_name) == {"Support", "Support > Error codes", "Support > Error codes > Look one up", "Support > Logs"}
    codes = by_name["Support > Error codes"]
    assert (codes.kind, codes.module, codes.path, codes.model) == (GUIDE_KIND, "docs:project", "docs/SUPPORT.md", "")
    assert (codes.line_start, codes.line_end) == (5, 12)
    assert codes.text.splitlines()[0] == "## Error codes" and "E514" in codes.text
    assert codes.embed_text.startswith("docs:project > docs/SUPPORT.md > Support > Error codes\n")
    assert len({u.id for u in units}) == len(units)


def test_a_hash_line_inside_a_code_fence_is_not_a_heading(tmp_path):
    units = parse({"SUPPORT.md": SUPPORT}, tmp_path)
    assert not any("not a heading" in u.name for u in units)
    assert "# not a heading" in next(u for u in units if u.name == "Support > Error codes").text


def test_a_heading_with_nothing_under_it_is_dropped(tmp_path):
    units = parse({"SUPPORT.md": SUPPORT}, tmp_path)
    assert not any(u.name.endswith("Empty section") for u in units)


def test_text_before_the_first_heading_is_kept_under_the_file_name(tmp_path):
    (unit,) = parse({"NOTES.md": "Plain notes without a heading.\n"}, tmp_path)
    assert unit.name == "NOTES" and unit.text == "Plain notes without a heading."


def test_nested_folders_other_files_and_the_directory_label(tmp_path):
    files = {"features/returns.md": "# Returns\n\nRefunds sync.\n", "image.png": "x", "notes.txt": "# no\n\ntext\n"}
    units = parse(files, tmp_path, directory="documentation")
    assert [(u.path, u.name) for u in units] == [("documentation/features/returns.md", "Returns")]


def test_windows_line_endings_and_a_form_feed_do_not_shift_line_numbers(tmp_path):
    text = "# Title\r\n\r\nfirst\x0csecond\r\n\r\n## Next\r\n\r\nbody\r\n"
    units = parse({"A.md": text}, tmp_path)
    assert {u.name: (u.line_start, u.line_end) for u in units} == {"Title": (1, 3), "Title > Next": (5, 7)}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    write(root, "docs/SUPPORT.md", SUPPORT)
    write(root, "docs/features/returns.md", "# Returns\n\nA refund creates a credit note and a carrier order.\n")
    write(root, "src/code.py", "x = 1\n")
    git(tmp_path, "init", "-q", "-b", "19.0", str(root))
    git(root, "add", ".")
    git(root, "commit", "-qm", "docs")
    return root


def test_git_source_reads_the_docs_of_the_ref_not_the_working_tree(repo):
    write(repo, "docs/SUPPORT.md", "# Changed\n\nuncommitted\n")
    units = parse_guides(GitSource(repo, "19.0", "docs"), "project", "docs")
    assert "Support > Error codes" in {u.name for u in units}
    assert not any("uncommitted" in u.text for u in units)


def test_a_docs_link_is_followed_to_the_directory_it_points_at(tmp_path):
    root = tmp_path / "ext"
    write(root, "shared/guides/FEATURE.md", "# Feature\n\nHow it works.\n")
    (root / "docs").symlink_to("shared/guides")
    git(tmp_path, "init", "-q", "-b", "19.0", str(root))
    git(root, "add", ".")
    git(root, "commit", "-qm", "link")
    assert resolve_docs_dir(root, "19.0", "docs") == "shared/guides"
    assert resolve_docs_dir(root, "19.0", "shared/guides") == "shared/guides"
    units = parse_guides(GitSource(root, "19.0", resolve_docs_dir(root, "19.0", "docs")), "ext", "docs")
    assert [u.name for u in units] == ["Feature"]


def test_a_missing_docs_directory_is_an_error(repo):
    with pytest.raises(DocsSourceError, match="no docs"):
        resolve_docs_dir(repo, "19.0", "documentation")
    with pytest.raises(DocsSourceError, match="no such ref"):
        resolve_docs_dir(repo, "20.0", "docs")


# --- registry and sync ------------------------------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path, repo):
    store = CodeStore(tmp_path / "db")
    return DocsRegistry(tmp_path / "docs_sources.json"), store, FakeEmbedder()


def run_sync(env, **kwargs):
    registry, store, embedder = env
    return sync_sources(registry, store, embedder, log=lambda m: None, **kwargs)


def test_add_records_the_source_and_sync_indexes_its_sections(env, repo):
    registry, store, _ = env
    source = add_source(registry, repo, ref="19.0")
    assert (source.name, source.dir, source.ref) == ("repo", "docs", "19.0")
    assert Path(registry.load()["repo"].repo) == repo.resolve()
    (result,) = run_sync(env)
    assert result.name == "repo" and result.rows > 0 and not result.skipped
    assert store.modules() == {"docs:repo": result.rows}
    assert store.indexed_modules()["docs:repo"]["ref"] == "19.0"
    rows = store.rows_where("kind = 'guide'", ["path", "name"])
    assert {"docs/SUPPORT.md", "docs/features/returns.md"} == {r["path"] for r in rows}


def test_sync_skips_a_source_whose_docs_did_not_change_and_picks_up_new_files(env, repo):
    add_source(env[0], repo, ref="19.0")
    run_sync(env)
    (again,) = run_sync(env)
    assert again.skipped and again.rows == 0
    write(repo, "docs/features/inventory.md", "# Inventory\n\nStock levels are exported hourly.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "more docs")
    (after,) = run_sync(env)
    assert not after.skipped
    assert any(r["path"] == "docs/features/inventory.md" for r in env[1].rows_where("kind = 'guide'", ["path"]))
    (forced,) = run_sync(env, force=True)
    assert not forced.skipped


def test_removed_file_leaves_the_index_on_the_next_sync(env, repo):
    add_source(env[0], repo, ref="19.0")
    run_sync(env)
    git(repo, "rm", "-q", "docs/features/returns.md")
    git(repo, "commit", "-qm", "drop")
    run_sync(env)
    assert not any(r["path"].endswith("returns.md") for r in env[1].rows_where("kind = 'guide'", ["path"]))


def test_a_source_with_a_broken_ref_does_not_stop_the_others(env, repo, tmp_path):
    registry = env[0]
    add_source(registry, repo, ref="19.0")
    other = tmp_path / "other"
    write(other, "docs/A.md", "# A\n\ntext about sync\n")
    git(tmp_path, "init", "-q", "-b", "main", str(other))
    git(other, "add", ".")
    git(other, "commit", "-qm", "x")
    add_source(registry, other, ref="main", name="other")
    git(other, "branch", "-m", "renamed")  # the registered ref is gone
    results = {r.name: r for r in run_sync(env)}
    assert results["repo"].rows > 0 and results["other"].error and "no such ref" in results["other"].error


def test_add_refuses_a_second_source_with_the_same_name_unless_it_is_the_same_repo(env, repo, tmp_path):
    registry = env[0]
    add_source(registry, repo, ref="19.0")
    add_source(registry, repo, ref="19.0", dir="docs/features")  # same repo: replaces the entry
    assert registry.load()["repo"].dir == "docs/features"
    other = tmp_path / "x" / "repo"
    write(other, "docs/A.md", "# A\n\ntext\n")
    git(tmp_path, "init", "-q", "-b", "19.0", str(other))
    git(other, "add", ".")
    git(other, "commit", "-qm", "x")
    with pytest.raises(DocsSourceError, match="already"):
        add_source(registry, other, ref="19.0")
    add_source(registry, other, ref="19.0", name="repo2")


def test_remove_drops_the_entry_and_its_rows(env, repo):
    add_source(env[0], repo, ref="19.0")
    run_sync(env)
    remove_source(env[0], env[1], "repo")
    assert env[0].load() == {} and env[1].modules() == {}
    with pytest.raises(DocsSourceError, match="no docs source"):
        remove_source(env[0], env[1], "repo")
