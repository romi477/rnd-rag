"""Registered documentation sources (the `docs/` directory of a repository at a git ref) -> the code index."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from rag import config
from rag.code_index import CodeIndexError, index_module
from rag.code_store import CodeStore
from rag.sources.guide_docs import DocsSourceError, docs_module, git_out, parse_guides, resolve_docs_dir
from rag.sources.odoo_module import GitSource

__all__ = ["DocsSource", "DocsRegistry", "DocsSourceError", "SyncResult", "add_source", "remove_source", "sync_sources"]

REGISTRY_FILE = "docs_sources.json"


@dataclass
class DocsSource:
    name: str
    repo: str
    dir: str
    ref: str
    tree: str = ""  # the git tree of the docs directory when it was last indexed


@dataclass
class SyncResult:
    name: str
    rows: int = 0
    skipped: bool = False
    error: str = ""


class DocsRegistry:
    def __init__(self, path: Path | None = None):
        self.path = path or config.DATA_DIR / "code" / REGISTRY_FILE

    def load(self) -> dict[str, DocsSource]:
        if not self.path.exists():
            return {}
        return {name: DocsSource(**fields) for name, fields in json.loads(self.path.read_text(encoding="utf-8")).items()}

    def save(self, sources: dict[str, DocsSource]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({n: asdict(s) for n, s in sources.items()}, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def add_source(registry: DocsRegistry, path: Path, ref: str, dir: str | None = None, name: str | None = None) -> DocsSource:
    """Register the docs of the repository that holds `path`. `path` is the repository or a directory inside it; a
    directory inside it is taken as the docs directory itself unless `dir` says otherwise. The docs are looked up at
    `ref` now, so a wrong ref or directory fails here and not at the first sync."""
    path = Path(path).resolve()
    top = Path(git_out(path if path.is_dir() else path.parent, "rev-parse", "--show-toplevel").strip()).resolve()
    inside = path.relative_to(top).as_posix()
    directory = dir or ("docs" if inside == "." else inside)
    resolve_docs_dir(top, ref, directory)
    name = name or top.name
    sources = registry.load()
    if name in sources and Path(sources[name].repo) != top:
        raise DocsSourceError(f"a docs source named {name!r} already exists for {sources[name].repo}: pass --name")
    sources[name] = DocsSource(name, str(top), directory.strip("/"), ref)
    registry.save(sources)
    return sources[name]


def remove_source(registry: DocsRegistry, store: CodeStore, name: str) -> None:
    sources = registry.load()
    if name not in sources:
        raise DocsSourceError(f"no docs source {name!r}: {', '.join(sources) or 'none registered'}")
    store.delete_module(docs_module(name))
    del sources[name]
    registry.save(sources)


def sync_sources(
    registry: DocsRegistry,
    store: CodeStore,
    embedder,
    log: Callable[[str], None] = print,
    force: bool = False,
    only: list[str] | None = None,
) -> list[SyncResult]:
    """Read every registered source at its ref again and replace its rows. A source whose docs directory has the same
    git tree as at the last sync is skipped (`force` reads it anyway). One failing source does not stop the others."""
    sources = registry.load()
    if only:
        if missing := [n for n in only if n not in sources]:
            raise DocsSourceError(f"no docs source {', '.join(missing)}: {', '.join(sources) or 'none registered'}")
        sources = {n: sources[n] for n in only}
    results = []
    for name, source in sources.items():
        module, repo = docs_module(name), Path(source.repo)
        try:
            real = resolve_docs_dir(repo, source.ref, source.dir)
            tree = git_out(repo, "rev-parse", f"{source.ref}:{real}").strip()
            held = store.indexed_modules().get(module)
            if not force and tree == source.tree and held and held["ref"] == source.ref and module in store.modules():
                results.append(SyncResult(name, skipped=True))
                continue
            units = parse_guides(GitSource(repo, source.ref, real), name, real)
            rows = index_module(store, embedder, module, units, log=log, ref=source.ref)
        except (DocsSourceError, CodeIndexError, subprocess.CalledProcessError) as exc:
            results.append(SyncResult(name, error=str(exc)))
            continue
        source.tree = tree
        registry.save(registry.load() | {name: source})
        results.append(SyncResult(name, rows=rows))
    return results
