"""Markdown documentation of a repository (a `docs/` directory) -> units for the code index.

Each section (a heading and the text up to the next heading) is one unit of kind `guide`, stored under the pseudo module
`docs:<name>` so that it never collides with an Odoo module of the same name.
"""

from __future__ import annotations

import posixpath
import re
import subprocess
from pathlib import Path, PurePath

from rag.sources.odoo_module import CodeUnit, FileSource, split_lines

GUIDE_KIND = "guide"
DOCS_PREFIX = "docs:"
HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
BARE_RULE = re.compile(r"[\s\-_*=|]*")  # a section with nothing but rules and spaces under its heading is empty


class DocsSourceError(RuntimeError):
    pass


def docs_module(name: str) -> str:
    return DOCS_PREFIX + name


def git_out(repo: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise DocsSourceError(exc.stderr.decode("utf-8", errors="replace").strip() or "git failed") from exc
    return out.stdout.decode("utf-8", errors="replace")


def resolve_docs_dir(repo: Path, ref: str, directory: str) -> str:
    """The directory of `ref` that holds the docs. A `docs` entry that is a symbolic link (an extension repository
    pointing at shared docs) is followed to the directory it names, inside the same repository."""
    try:
        git_out(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    except DocsSourceError as exc:
        raise DocsSourceError(f"no such ref {ref!r} in {repo}") from exc
    current = directory.strip("/")
    for _ in range(5):
        entry = git_out(repo, "ls-tree", ref, "--", current).strip()
        if not entry:
            raise DocsSourceError(f"no docs directory {current!r} at {ref} in {repo}")
        mode = entry.split(" ", 1)[0]
        if mode == "040000":
            return current
        if mode != "120000":
            raise DocsSourceError(f"{current!r} at {ref} in {repo} is a file, not a directory")
        target = posixpath.normpath(posixpath.join(posixpath.dirname(current), git_out(repo, "show", f"{ref}:{current}").strip()))
        if target.startswith(".."):
            raise DocsSourceError(f"{current!r} points outside the repository ({target})")
        current = target
    raise DocsSourceError(f"{directory!r}: too many links in a row")


def sections(lines: list[str]) -> list[tuple[list[str], int, int]]:
    """(heading path, first line, last line) of each section, 0-based and inclusive. A `#` line inside a code fence is text."""
    starts: list[tuple[int, int, str]] = []  # (line, level, title)
    fence = ""
    for i, line in enumerate(lines):
        if m := FENCE.match(line):
            mark = m.group(1)
            if not fence:
                fence = mark
            elif mark[0] == fence[0] and len(mark) >= len(fence):
                fence = ""
        elif not fence and (h := HEADING.match(line)):
            starts.append((i, len(h.group(1)), h.group(2).strip()))
    found: list[tuple[list[str], int, int]] = []
    if not starts or starts[0][0] > 0:
        found.append(([], 0, (starts[0][0] if starts else len(lines)) - 1))
    stack: list[tuple[int, str]] = []
    for n, (line, level, title) in enumerate(starts):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        found.append(([t for _, t in stack], line, (starts[n + 1][0] if n + 1 < len(starts) else len(lines)) - 1))
    return found


def guide_units(module: str, rel: str, shown: str, source: str) -> list[CodeUnit]:
    lines = split_lines(source)
    units = []
    for path, first, last in sections(lines):
        while last >= first and not lines[last].strip():
            last -= 1
        body = lines[first + 1 : last + 1] if path else lines[first : last + 1]
        if last < first or BARE_RULE.fullmatch("\n".join(body)):
            continue
        name = " > ".join(path) or PurePath(rel).stem
        text = "\n".join(lines[first : last + 1])
        units.append(CodeUnit(
            GUIDE_KIND, module, "", "docs", "", name, shown, first + 1, last + 1, text, f"{module} > {shown} > {name}\n{text}",
        ))
    return units


def parse_guides(source: FileSource, name: str, directory: str = "docs") -> list[CodeUnit]:
    """The sections of every Markdown file under the docs directory. `directory` is how the directory is called in
    citations: a unit's path is `<directory>/<file inside it>`."""
    module = docs_module(name)
    units: list[CodeUnit] = []
    for rel in source.files():
        if rel.lower().endswith(".md"):
            units += guide_units(module, rel, f"{directory.strip('/')}/{rel}", source.read(rel))
    return units
