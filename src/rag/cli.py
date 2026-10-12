"""`rag` command line: sync, index, search, ask, bench, eval, serve."""

from __future__ import annotations

import textwrap
from pathlib import Path

import typer

from rag import config

app = typer.Typer(no_args_is_help=True, add_completion=False)
module_app = typer.Typer(no_args_is_help=True, help="Odoo addons as a code corpus")
app.add_typer(module_app, name="module")

MODES = ("hybrid", "vector", "fts")  # = rag.search.MODES, without importing torch for --help


def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise typer.BadParameter(f"expected one of {', '.join(MODES)}")
    return mode


CORPORA = ("docs", "code")


def check_corpus(corpus: str) -> str:
    if corpus not in CORPORA:
        raise typer.BadParameter(f"expected one of {', '.join(CORPORA)}")
    return corpus


def check_lang(lang: str) -> str:
    from rag.answer import check_language

    try:
        return check_language(lang)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def fail(message: str) -> None:
    typer.secho(message, fg="red", err=True)
    raise typer.Exit(1)


def git_ignored(path: Path) -> bool:
    """True if git ignores the path (or there is no git repo to commit it to)."""
    import subprocess

    try:
        res = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", str(path.resolve())],
            cwd=config.ROOT, capture_output=True,
        )
    except OSError:
        return True
    return res.returncode != 1  # 0: ignored; 128: not a repo / outside it


def make_embedder(model: str = config.EMBED_MODEL):
    from rag.embedding import Embedder

    return Embedder(model)


def open_searcher(model: str = config.EMBED_MODEL):
    from rag.embedding import Embedder
    from rag.search import Searcher

    try:
        return Searcher(Embedder(model))
    except RuntimeError as exc:  # no index yet
        fail(str(exc))


def open_module(path: Path, ref: str, tag_prefix: str, series: str, working_tree: bool = False):
    """(module directory, git ref or '', file source) for the --ref / --tag-prefix / --working-tree options of
    `rag module`. The files are read from a git ref: the checked-out files are whatever branch and edits the
    repository has at the moment, so reading them needs the explicit --working-tree."""
    from rag.sources.odoo_module import DirSource, GitSource, git_location, latest_tag

    path = path.resolve()
    if not ref and not tag_prefix:
        if not working_tree:
            fail("Say which version to read: --ref REF (a tag, branch or sha), --tag-prefix PREFIX, or --working-tree "
                 "for the files as checked out (they may not match any commit)")
        if not (path / "__manifest__.py").exists():
            fail(f"{path} has no __manifest__.py")
        return path, "", DirSource(path)
    repo, prefix = git_location(path)
    if tag_prefix:
        ref = latest_tag(repo, tag_prefix, series) or ""
        if not ref:
            fail(f"no tag {tag_prefix}-{series}.* in {repo}")
    return path, ref, GitSource(repo, ref, prefix)


@module_app.command("inspect")
def module_inspect(
    path: Path = typer.Argument(..., help="Addon directory (contains __manifest__.py)"),
    ref: str = typer.Option("", help="Git ref to read instead of the working tree (tag, branch, sha)"),
    tag_prefix: str = typer.Option("", help="Resolve the latest <prefix>-<series>.* tag instead of --ref"),
    series: str = typer.Option("19.0", help="Odoo series for --tag-prefix"),
    origin: str = typer.Option("connector", help="core | connector | customer"),
    json_out: Path = typer.Option(None, "--json", help="Write all units to this JSON file"),
    working_tree: bool = typer.Option(False, "--working-tree", help="Read the files as checked out instead of a git ref"),
) -> None:
    """Parse an addon offline (no embeddings, graph untouched) and print unit counts by kind."""
    import collections
    import dataclasses
    import json

    from rag.sources.odoo_module import parse_module

    path, ref, source = open_module(path, ref, tag_prefix, series, working_tree)
    units = parse_module(source, path.name, origin=origin)
    version = next((u.module_version for u in units if u.kind == "manifest"), "?")
    typer.echo(f"{path.name} {version} ({ref or 'working tree'}): {len(units)} units")
    for kind, n in sorted(collections.Counter(u.kind for u in units).items()):
        typer.echo(f"  {kind:<10}{n:>5}")
    if json_out:
        json_out.write_text(json.dumps([{**dataclasses.asdict(u), "id": u.id} for u in units], indent=1))
        typer.echo(f"wrote {json_out}")


@module_app.command("add")
def module_add(
    path: Path = typer.Argument(..., help="Addon directory (contains __manifest__.py)"),
    ref: str = typer.Option("", help="Git ref to read instead of the working tree (tag, branch, sha)"),
    tag_prefix: str = typer.Option("", help="Resolve the latest <prefix>-<series>.* tag instead of --ref"),
    series: str = typer.Option("19.0", help="Odoo series for --tag-prefix"),
    origin: str = typer.Option("connector", help="core | connector | customer"),
    index: bool = typer.Option(True, "--index/--no-index", help="Embed the units for search (needs the embedding model)"),
    model: str = typer.Option(config.EMBED_MODEL, help="Embedding model (selects the index)"),
    reset_index: bool = typer.Option(False, help="Drop all modules from the search index first (after an index version change)"),
    working_tree: bool = typer.Option(False, "--working-tree", help="Read the files as checked out instead of a git ref"),
) -> None:
    """Add (or replace) an addon in the code graph and, unless --no-index, in the code search index."""
    from rag.code_graph import CodeGraph, ModuleInfo, graph_path
    from rag.code_index import CodeIndexError, index_module
    from rag.code_store import CodeStore
    from rag.sources.odoo_module import parse_module
    from rag.store import store_path

    path, ref, source = open_module(path, ref, tag_prefix, series, working_tree)
    units = parse_module(source, path.name, origin=origin)
    if not any(u.kind == "manifest" for u in units):
        fail(f"{path.name}: no readable __manifest__.py at {ref or 'the working tree'}")
    graph = CodeGraph.load(graph_path())
    info = ModuleInfo.from_units(path.name, units, origin, ref)
    if index:
        embedder = make_embedder(model)
        try:
            n_rows = index_module(
                CodeStore(store_path(embedder.model_name)), embedder, info.name, units, log=typer.echo, reset=reset_index, ref=ref
            )
        except CodeIndexError as exc:
            fail(str(exc))
    graph.add_module(info, units)
    graph.save(graph_path())
    typer.echo(f"{info.name} {info.version} ({ref or 'working tree'}): {len(units)} units")
    if index:
        typer.echo(f"indexed {n_rows} rows for search")
    else:
        typer.secho(f"{info.name} is not in the search index: run again without --no-index before using it for answers", fg="yellow")
    if missing := graph.missing_depends(info.name):
        typer.echo(f"depends not added yet: {', '.join(missing)} (Odoo core and other repositories are expected here)")


@module_app.command("list")
def module_list(
    model: str = typer.Option(config.EMBED_MODEL, help="Embedding model (selects the index)"),
) -> None:
    """Modules in the code graph, dependencies first; a warning where the search index and the graph differ."""
    from rag.code_graph import CodeGraph, graph_path
    from rag.code_lookup import sync_problems
    from rag.code_store import CodeStore
    from rag.store import store_path

    graph = CodeGraph.load(graph_path())
    if not graph.modules:
        typer.echo("no modules; add one with `rag module add`")
    for name in graph.order():
        m = graph.modules[name]
        n_methods = sum(1 for x in graph.methods if x.module == name)
        typer.echo(f"{name:<28}{m.version:<14}{m.origin:<10}{m.ref or 'working tree':<28}{n_methods:>5} methods")
    for name, problem in sync_problems(graph, CodeStore(store_path(model))).items():
        typer.secho(f"{name}: {problem}", fg="yellow")


@module_app.command("rm")
def module_rm(
    name: str,
    model: str = typer.Option(config.EMBED_MODEL, help="Embedding model (selects the index)"),
    force: bool = typer.Option(False, "--force", help="Remove it although other modules depend on it"),
) -> None:
    """Remove an addon from the code graph and the code search index."""
    from rag.code_graph import CodeGraph, graph_path
    from rag.code_store import CodeStore
    from rag.store import store_path

    graph = CodeGraph.load(graph_path())
    if name not in graph.modules:
        fail(f"{name} is not in the graph")
    if dependants := [m.name for m in graph.modules.values() if name in m.depends]:
        if not force:
            fail(f"{', '.join(dependants)} depend on {name}: their overrides and chains would lose it (--force removes it anyway)")
        typer.secho(f"{', '.join(dependants)} depend on {name}", fg="yellow")
    CodeStore(store_path(model)).delete_module(name)  # the index first: a failure leaves the graph, which `module list` shows
    graph.remove_module(name)
    graph.save(graph_path())
    typer.echo(f"removed {name}")


def echo_methods(methods) -> None:
    for m in methods:
        flags = [f for f, on in (("super", m.calls_super), ("abstract", m.abstract)) if on]
        typer.echo(
            f"  {m.module}: {m.path}:{m.line_start}-{m.line_end}  {m.class_name}.{m.name}"
            + (f"  [{', '.join(flags)}]" if flags else "")
        )


@module_app.command("chain")
def module_chain(
    model: str,
    method: str,
    only: list[str] = typer.Option(None, "--only", help="Limit to these modules (repeatable)"),
) -> None:
    """Definitions of a method on a model across modules, base first."""
    from rag.code_graph import CodeGraph, graph_path

    chain = CodeGraph.load(graph_path()).chain(model, method, set(only) if only else None)
    if not chain:
        fail(f"{model}.{method} is not defined in the indexed modules")
    typer.echo(f"{model}.{method}: {len(chain)} definition(s), base first (found statically)")
    echo_methods(chain)


@module_app.command("adapters")
def module_adapters(method: str) -> None:
    """The abstract declaration of an adapter method and each connector's implementation."""
    from rag.code_graph import CodeGraph, graph_path

    group = CodeGraph.load(graph_path()).adapter_group(method)
    if not group.abstract and not group.impls:
        fail(f"no adapter method {method}")
    typer.echo(f"{method}: abstract declaration")
    echo_methods(group.abstract)
    typer.echo("implementations")
    for methods in group.impls.values():
        echo_methods(methods)


@app.command()
def trace(
    file: Path = typer.Argument(None, help="File with the traceback; omit to read stdin"),
    model: str = typer.Option(config.EMBED_MODEL, help="Embedding model (selects the index)"),
    client_version: str = typer.Option("", help="Module version on the client's server, to warn when it differs"),
    show_text: bool = typer.Option(False, "--text", help="Print the code of each frame"),
) -> None:
    """Map a traceback to the indexed code: functions, overrides, error codes. Exact lookups, no model needed."""
    import sys

    from rag.code_graph import CodeGraph, graph_path
    from rag.code_lookup import find_error_code, related_methods, resolve_frames, scope_modules, sync_problems
    from rag.code_store import CodeStore
    from rag.store import store_path
    from rag.traceback import parse_traceback, semantic_query, version_note

    graph = CodeGraph.load(graph_path())
    store = CodeStore(store_path(model))
    if not graph.modules or not store.exists():
        fail("No code index; add modules with `rag module add` first")
    text = file.read_text() if file else sys.stdin.read()
    tb = parse_traceback(text, set(graph.modules))
    if not tb.frames and not tb.exception:
        fail("No traceback found in the input")
    if tb.exception:
        typer.echo(f"{tb.exception.name}: {textwrap.shorten(tb.exception.message, 200)}")
    resolved = resolve_frames(store, tb)  # may settle a frame on another module than its path first suggests
    visible = {f.module for f in tb.own_frames}
    typer.echo(f"{len(tb.frames)} frames, {len(tb.own_frames)} in indexed modules (innermost first)")
    for hit in resolved:
        f = hit.frame
        if not hit.found:
            typer.echo(f"  {f.module}/{f.rel}:{f.line} in {f.func}: no indexed unit contains this line")
            continue
        note = "" if hit.name_matches else f"  (inside {hit.name})"
        typer.secho(f"  {f.module}/{f.rel}:{hit.line_start}-{hit.line_end}  {hit.model or '-'}.{hit.name}  at line {f.line}{note}", bold=True)
        related = related_methods(graph, hit, visible)
        for m in related.chain:
            typer.echo(f"      also: {m.module}/{m.path}:{m.line_start}-{m.line_end}" + ("  [super]" if m.calls_super else ""))
        for m in related.abstract:
            typer.echo(f"      abstract: {m.module}/{m.path}:{m.line_start}-{m.line_end}")
        if show_text:
            typer.echo(textwrap.indent(hit.text, "      | "))
    scope = scope_modules(graph, visible)
    for code in tb.error_codes:
        typer.echo(f"error code {code}:")
        for u in find_error_code(store, code, scope):
            typer.echo(f"  {u.kind} {u.module}/{u.path}:{u.line_start}-{u.line_end}  {u.name}")
    versions = {m.name: m.version for m in graph.modules.values()}
    problems = sync_problems(graph, store, visible)
    for module in sorted(visible):
        if note := version_note(client_version, versions[module]):
            typer.echo(f"{module}: {note}")
        if module in problems:
            typer.secho(f"{module}: the index and the graph differ ({problems[module]})", fg="yellow")
    typer.echo(f"search query: {semantic_query(tb)}")


@app.command()
def sync(
    space: str = typer.Option(config.SPACE_KEY, help="Confluence space key"),
    attachments: bool = typer.Option(False, help="Also download page attachments"),
    force: bool = typer.Option(False, help="Rewrite pages even if version unchanged"),
    reconvert: bool = typer.Option(False, help="Offline: rebuild the markdown from the stored HTML"),
) -> None:
    """Download/refresh Confluence pages into raw-data/ (--force: rewrite all, accept mass removals)."""
    from rag.sources.confluence import reconvert as run_reconvert
    from rag.sources.confluence import sync as run_sync

    if reconvert:
        typer.echo(f"Re-converted {run_reconvert(space, config.RAW_DIR)} pages")
        return
    run_sync(space, config.RAW_DIR, attachments=attachments, force=force)


@app.command()
def index(
    space: str = typer.Option(config.SPACE_KEY, help="Confluence space key"),
    model: str = typer.Option(config.EMBED_MODEL, help="Embedding model"),
    force: bool = typer.Option(False, help="Rebuild the whole index; also accept dropping many pages"),
) -> None:
    """Chunk + embed synced pages into LanceDB (incremental by page version)."""
    from rag.embedding import Embedder
    from rag.indexing import IndexingError, index_space

    try:
        stats = index_space(space, embedder=Embedder(model), force=force, log=typer.echo)
    except (IndexingError, ValueError) as exc:
        fail(str(exc))
    typer.echo(
        f"pages={stats.pages} updated={stats.updated} removed={stats.removed} "
        f"chunks_written={stats.chunks_written} total_chunks={stats.total_chunks}"
    )


def search_code(query: str, k: int, mode: str, model: str, modules: list[str] | None, show_text: bool) -> None:
    from rag.code_search import CodeSearcher

    try:
        searcher = CodeSearcher(make_embedder(model))
    except RuntimeError as exc:  # no code index yet
        fail(str(exc))
    hits = searcher.search(query, k=k, mode=mode, modules=modules or None)
    for i, hit in enumerate(hits, 1):
        owner = f"{hit.model}.{hit.name}" if hit.model and hit.kind == "method" else hit.model or hit.name
        typer.secho(f"{i}. {hit.kind} {owner}  [{hit.score:.3f}]", bold=True)
        typer.echo(f"   {hit.module} {hit.module_version} | {hit.path}:{hit.line_start}-{hit.line_end}")
        if show_text:
            typer.echo(textwrap.indent(textwrap.shorten(" ".join(hit.text.split()), 400), "     "))
    if not hits:
        typer.echo("No results")


@app.command()
def search(
    query: str,
    k: int = typer.Option(5, "-k", help="Number of pages"),
    mode: str = typer.Option("hybrid", help="hybrid | vector | fts", callback=check_mode),
    product: str | None = typer.Option(None, help="Filter by product, e.g. 'Direct Print'"),
    rerank: bool = typer.Option(False, help="Rerank candidates with a cross-encoder"),
    model: str = typer.Option(config.EMBED_MODEL, help="Embedding model (selects the index)"),
    show_text: bool = typer.Option(False, "--text", help="Print matched chunk text"),
    corpus: str = typer.Option("docs", help="docs (Confluence) | code (Odoo addons)", callback=check_corpus),
    module: list[str] = typer.Option(None, help="Code corpus: only these modules (repeatable)"),
) -> None:
    """Search the index from the terminal."""
    if corpus == "code":
        return search_code(query, k, mode, model, module, show_text)
    searcher = open_searcher(model)
    hits = searcher.search(query, k=k, mode=mode, product=product, rerank=rerank)
    for i, page in enumerate(hits, 1):
        low = "  (low relevance)" if page.low_relevance else ""
        typer.secho(f"{i}. {page.title}  [{page.score:.3f}]{low}", bold=not page.low_relevance, dim=page.low_relevance)
        typer.echo(f"   {' > '.join(page.breadcrumbs) or page.product} | {page.url}")
        for chunk in page.chunks:
            section = " > ".join(chunk.section_path) or "(page)"
            typer.echo(f"   - {section} [{chunk.score:.3f}]")
            if show_text:
                typer.echo(textwrap.indent(textwrap.shorten(chunk.text, 400), "     "))
    if not hits:
        typer.echo("No results")


def code_events(query, traceback_file, modules, client_version, with_docs, model, llm, show_prompt, lang="en"):
    """Events of a code answer; with `show_prompt` a single delta holding the messages for the LLM."""
    import sys

    from rag.code_answer import build_code_context, build_code_messages, gather, stream_code_answer
    from rag.code_graph import CodeGraph, graph_path
    from rag.code_search import CodeSearcher
    from rag.code_store import CodeStore
    from rag.store import store_path
    from rag.traceback import looks_like_traceback, semantic_query

    graph = CodeGraph.load(graph_path())
    store = CodeStore(store_path(model))
    if not graph.modules or not store.exists():
        fail("No code index; add modules with `rag module add` first")
    text = traceback_file.read_text() if traceback_file else ""
    if not text and looks_like_traceback(query):
        text, query = query, ""
    if not text and not query.strip():
        fail("Give a question or --traceback FILE")
    ev = gather(store, graph, query, text, CodeSearcher(make_embedder(model), store), modules or None, client_version)
    pages = None
    if with_docs:
        search_text = semantic_query(ev.tb) if ev.tb else query
        pages = open_searcher(model).search(search_text or query, k=config.LLM_CONTEXT_PAGES, rerank=True)
    if show_prompt:
        sources, context = build_code_context(ev.pieces)
        yield {"type": "sources", "sources": [dict(vars(s)) for s in sources]}
        for m in build_code_messages(query, ev, context, lang):
            yield {"type": "delta", "text": f"--- {m['role']} ---\n{m['content']}\n"}
        return
    yield from stream_code_answer(query, ev, llm, pages=pages, lang=lang)


@app.command()
def ask(
    query: str = typer.Argument("", help="The question (with --corpus code it may be empty when a traceback is given)"),
    mode: str = typer.Option("hybrid", help="hybrid | vector | fts", callback=check_mode),
    product: str | None = typer.Option(None, help="Filter by product, e.g. 'Direct Print'"),
    rerank: bool = typer.Option(True, help="Rerank and drop low-relevance sources (--no-rerank: off)"),
    llm_model: str = typer.Option(config.LLM_MODEL, help="LLM id (default: the one loaded in LM Studio, or claude-sonnet-5-5)"),
    corpus: str = typer.Option("docs", help="docs (Confluence) | code (Odoo addons)", callback=check_corpus),
    traceback_file: Path = typer.Option(None, "--traceback", help="Code corpus: file with a failed job's traceback"),
    module: list[str] = typer.Option(None, help="Code corpus: search only these modules (repeatable)"),
    client_version: str = typer.Option("", help="Code corpus: module version on the client's server"),
    with_docs: bool = typer.Option(False, "--docs", help="Code corpus: also quote Confluence pages"),
    model: str = typer.Option(config.EMBED_MODEL, help="Embedding model (selects the index)"),
    show_prompt: bool = typer.Option(False, help="Code corpus: print the sources and the prompt, do not call the LLM"),
    lang: str = typer.Option("", "--lang", help="Answer language: en, es, de, pl, ru (default: RAG_ANSWER_LANGUAGE)", callback=check_lang),
) -> None:
    """Answer a question with the LLM (RAG_LLM_PROVIDER), grounded in the search results."""
    from rag.answer import stream_answer
    from rag.llm import make_llm

    if corpus == "code":
        from rag.code_answer import external_llm_blocked

        if not show_prompt and (reason := external_llm_blocked()):
            fail(reason)
        events = code_events(
            query, traceback_file, module, client_version, with_docs, model, make_llm(model=llm_model), show_prompt, lang
        )
    else:
        if not query.strip():
            fail("Give a question")
        hits = open_searcher().search(query, k=config.LLM_CONTEXT_PAGES, mode=mode, product=product, rerank=rerank)
        events = stream_answer(query, hits, make_llm(model=llm_model), lang=lang)
    for event in events:
        if event["type"] == "delta":
            typer.echo(event["text"], nl=False)
        elif event["type"] == "error":
            typer.secho(f"\nLLM error: {event['message']}", fg="red", err=True)
            raise typer.Exit(1)
        elif event["type"] == "done":
            typer.echo()
            if event.get("truncated"):
                typer.secho("[cut off: the answer reached max_tokens]", fg="yellow", err=True)
            if event["model"]:
                typer.secho(f"\n[{event['model']}, {event['took_ms']} ms]", dim=True)
        elif event["type"] == "sources" and event["sources"]:
            for s in event["sources"]:
                if s.get("kind") == "code":
                    typer.secho(f"[{s['n']}] {s['module']}/{s['path']}:{s['line_start']}-{s['line_end']}  {s['label']}", dim=True)
                else:
                    typer.secho(f"[{s['n']}] {s['title']}  {s['url']}", dim=True)
            typer.echo()


@app.command("eval-traces")
def eval_traces(
    file: Path = typer.Option(config.TRACES_FILE, help="YAML list of traceback cases"),
    model: str = typer.Option(config.EMBED_MODEL, help="Embedding model (selects the index)"),
    search: bool = typer.Option(True, help="Include the semantic search hits (--no-search: frames and error codes only)"),
    misses: bool = typer.Option(False, help="List the cases whose cause is not first or not sent to the LLM"),
) -> None:
    """Does the evidence for each traceback contain the function the failure comes from? (no LLM)"""
    from rag.code_graph import CodeGraph, graph_path
    from rag.code_search import CodeSearcher
    from rag.code_store import CodeStore
    from rag.store import store_path
    from rag.trace_eval import evaluate_case, load_cases, summarize

    if not file.exists():
        fail(f"{file} not found: copy eval/traces.example.yaml and describe your cases")
    graph, store = CodeGraph.load(graph_path()), CodeStore(store_path(model))
    if not graph.modules or not store.exists():
        fail("No code index; add modules with `rag module add` first")
    searcher = CodeSearcher(make_embedder(model), store) if search else None
    results = []
    for case in load_cases(file):
        results.append(evaluate_case(case, store, graph, searcher))
    typer.echo(f"model={model} cases={len(results)} search={'on' if search else 'off'}")
    typer.echo(f"{'group':<12}{'n':>4}{'frames':>9}{'cause@1':>9}{'cause@3':>9}{'sent':>7}{'codes':>8}{'chain':>8}")
    show = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    for group, row in sorted(summarize(results).items(), key=lambda kv: (kv[0] == "all", kv[0])):
        typer.echo(
            f"{group:<12}{row['n']:>4}{show(row['frames']):>9}{show(row['cause@1']):>9}{show(row['cause@3']):>9}"
            f"{show(row['cause_sent']):>7}{show(row['codes']):>8}{show(row['chain']):>8}"
        )
    if misses:
        typer.secho("\ncases without the cause first or sent to the LLM", bold=True)
        for r in results:
            if r.cause_rank != 1 or not r.cause_sent or r.codes_found is False or r.chain_ok is False:
                typer.echo(
                    f"  {r.id} [{r.group}] frames {r.frames_found}/{r.frames_total} cause rank {r.cause_rank or '-'}"
                    f"{'' if r.cause_sent else ' (not sent)'} codes {r.codes_found} chain {r.chain_ok}"
                )


@app.command("bench-traces")
def bench_traces(
    models: list[str] = typer.Argument(None, help="LLM ids to compare (default: the loaded one)"),
    file: Path = typer.Option(config.TRACES_FILE, help="YAML list of traceback cases"),
    group: str | None = typer.Option(None, help="Only the cases of this group (real, synthetic, message, drift...)"),
    limit: int = typer.Option(0, help="Only the first N cases (0: all)"),
    model: str = typer.Option(config.EMBED_MODEL, help="Embedding model (selects the index)"),
    out: Path = typer.Option(None, help="Report path (default: data/bench/traces-<time>.md)"),
    lang: str = typer.Option("", "--lang", help="Answer language: en, es, de, pl, ru (default: RAG_ANSWER_LANGUAGE)", callback=check_lang),
) -> None:
    """Compare LLMs on the failure cases: the same evidence and prompt, answers side by side in a report."""
    from rag.code_answer import external_llm_blocked
    from rag.code_graph import CodeGraph, graph_path
    from rag.code_search import CodeSearcher
    from rag.code_store import CodeStore
    from rag.llm import LLMError, make_llm
    from rag.store import store_path
    from rag.trace_bench import default_report_path, render_report, run_trace_bench
    from rag.trace_eval import load_cases

    if reason := external_llm_blocked():
        fail(reason)
    if not file.exists():
        fail(f"{file} not found: copy eval/traces.example.yaml and describe your cases")
    graph, store = CodeGraph.load(graph_path()), CodeStore(store_path(model))
    if not graph.modules or not store.exists():
        fail("No code index; add modules with `rag module add` first")
    cases = [c for c in load_cases(file) if group is None or c.group == group]
    cases = cases[:limit] if limit else cases
    if not cases:
        fail("No cases selected")
    llm = make_llm()
    try:
        models = list(models or [llm.resolve_model()])
    except LLMError as exc:
        fail(f"LLM: {exc}")
    path = out or default_report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = f"{len(cases)} cases from {file.name}, embedding {model}, context {config.CODE_CONTEXT_TOKENS} tokens"
    typer.echo(f"{settings}; models: {', '.join(models)}")
    searcher = CodeSearcher(make_embedder(model), store)
    try:
        items = run_trace_bench(
            cases, models, store, graph, searcher, llm, log=typer.echo, lang=lang,
            on_item=lambda items: path.write_text(render_report(items, models, settings, partial=True)),
        )
    except (LLMError, KeyboardInterrupt) as exc:
        fail(f"stopped: {exc or 'interrupted'}; the report so far is in {path}")
    path.write_text(render_report(items, models, settings))
    typer.echo(f"report: {path}")


@app.command()
def bench(
    models: list[str] = typer.Argument(None, help="LLM ids to compare (default: the loaded one)"),
    file: Path = typer.Option(config.BENCH_FILE, help="YAML list of questions"),
    group: str | None = typer.Option(None, help="Only questions of this group, e.g. 'novice'"),
    out: Path | None = typer.Option(None, help="Markdown report (default: data/bench/<time>.md)"),
    mode: str = typer.Option("hybrid", help="hybrid | vector | fts", callback=check_mode),
    rerank: bool = typer.Option(True, help="Rerank and drop low-relevance sources"),
    lang: str = typer.Option("", "--lang", help="Answer language: en, es, de, pl, ru (default: RAG_ANSWER_LANGUAGE)", callback=check_lang),
) -> None:
    """Compare LLMs on the same questions and sources: latency, tokens/s, answers side by side.

    The report is rewritten after every question, so an interrupted run keeps what it measured.
    """
    import yaml

    from rag.bench import (
        default_report_path, load_questions, render_report, retrieval_line, run_bench, summary_rows,
    )
    from rag.llm import LLMError, make_llm

    if not file.exists():
        fail(f"No questions file {file}: cp eval/bench.example.yaml eval/bench.yaml")
    try:
        questions = load_questions(file, group)
    except (ValueError, yaml.YAMLError) as exc:
        fail(f"Bad questions file: {exc}")
    if not questions:
        fail(f"No questions in {file}" + (f" for group '{group}'" if group else ""))
    llm = make_llm()
    try:
        models = models or [llm.resolve_model()]
    except LLMError as exc:
        fail(str(exc))
    searcher = open_searcher()

    settings = (
        f"Questions: {len(questions)} ({file.name}{f', group {group}' if group else ''}) · search: {mode}{' + rerank' if rerank else ''} · "
        f"LLM: {llm.describe()} · max_tokens {config.LLM_MAX_TOKENS}"
    )
    out = out or default_report_path()
    if not git_ignored(out):
        fail(f"{out} is not git-ignored, and the report quotes the internal docs: put it under data/")
    out.parent.mkdir(parents=True, exist_ok=True)
    done: list = []

    def write(items: list, partial: bool = True) -> None:
        done[:] = items
        out.write_text(render_report(items, models, settings, partial=partial))

    try:
        items = run_bench(searcher, llm, questions, models, mode=mode, rerank=rerank, log=typer.echo, on_item=write, lang=lang)
    except (KeyboardInterrupt, LLMError) as exc:
        if done:
            write(done)
            typer.secho(f"\nStopped after {len(done)} of {len(questions)} questions; report: {out}", fg="yellow", err=True)
        fail(f"Bench stopped: {exc}" if isinstance(exc, LLMError) else "Interrupted")
    write(items, partial=False)

    rows = summary_rows(items, models)
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    typer.echo()
    for n, row in enumerate(rows):
        typer.secho("  ".join(c.ljust(w) for c, w in zip(row, widths)), bold=n == 0)
    typer.echo(retrieval_line(items))
    typer.echo(f"\nReport with all answers: {out}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: int = typer.Option(8000, help="Port"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes"),
) -> None:
    """Run the web search UI and JSON API."""
    import uvicorn

    from rag.store import Store, store_path

    if not Store(store_path(config.EMBED_MODEL)).exists():
        fail(f"No search index for {config.EMBED_MODEL}: run scripts/update.sh (or `rag index`) first")

    if host not in ("127.0.0.1", "localhost", "::1") and not config.API_TOKEN:
        typer.secho(f"Listening on {host} without RAG_API_TOKEN: anyone who can reach it reads the docs.", fg="yellow", err=True)
    uvicorn.run("rag.api:app", host=host, port=port, reload=reload)


@app.command("eval")
def eval_cmd(
    model: str = typer.Option(config.EMBED_MODEL, help="Embedding model (selects the index)"),
    modes: str = typer.Option("fts,vector,hybrid", help="Comma-separated modes: fts, vector, hybrid"),
    rerank: bool = typer.Option(True, help="Also evaluate hybrid + cross-encoder rerank"),
    misses: bool = typer.Option(False, help="List queries whose answer is not ranked first"),
    fts_weights: str = typer.Option(
        str(config.HYBRID_FTS_WEIGHT), help="Comma-separated BM25 weights to try for hybrid"
    ),
) -> None:
    """Compare retrieval modes on eval/queries.yaml (page-level hit@k and MRR)."""
    from rag.evaluation import evaluate, load_queries

    if not config.EVAL_FILE.exists():
        fail(f"No {config.EVAL_FILE}: cp eval/queries.example.yaml eval/queries.yaml and fill it")
    mode_list = [check_mode(m.strip()) for m in modes.split(",") if m.strip()]
    queries = load_queries(config.EVAL_FILE)
    searcher = open_searcher(model)
    weights = [float(w) for w in fts_weights.split(",")]
    runs = []  # (name, mode, rerank, fts_weight)
    for mode in mode_list:
        if mode == "hybrid":
            runs += [(f"hybrid w={w:g}", "hybrid", False, w) for w in weights]
        else:
            runs.append((mode, mode, False, config.HYBRID_FTS_WEIGHT))
    if rerank:
        runs += [(f"hybrid w={w:g}+rr", "hybrid", True, w) for w in weights]

    typer.echo(f"model={model} queries={len(queries)}")
    typer.echo(f"{'config':<20}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'hit@10':>8}{'MRR':>8}")
    results = []
    for name, mode, rr, w in runs:
        res = evaluate(searcher, queries, mode=mode, rerank=rr, name=name, fts_weight=w)
        results.append(res)
        typer.echo(
            f"{res.name:<20}{res.hit(1):>8.2f}{res.hit(3):>8.2f}{res.hit(5):>8.2f}"
            f"{res.hit(10):>8.2f}{res.mrr:>8.3f}"
        )
    if misses:
        for res in results:
            typer.secho(f"\n{res.name}: not ranked first", bold=True)
            for q, rank in res.misses:
                typer.echo(f"  [{rank or '-'}] {q}")
