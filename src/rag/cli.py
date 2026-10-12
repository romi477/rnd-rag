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


def open_searcher(model: str = config.EMBED_MODEL):
    from rag.embedding import Embedder
    from rag.search import Searcher

    try:
        return Searcher(Embedder(model))
    except RuntimeError as exc:  # no index yet
        fail(str(exc))


@module_app.command("inspect")
def module_inspect(
    path: Path = typer.Argument(..., help="Addon directory (contains __manifest__.py)"),
    ref: str = typer.Option("", help="Git ref to read instead of the working tree (tag, branch, sha)"),
    tag_prefix: str = typer.Option("", help="Resolve the latest <prefix>-<series>.* tag instead of --ref"),
    series: str = typer.Option("19.0", help="Odoo series for --tag-prefix"),
    origin: str = typer.Option("connector", help="core | connector | customer"),
    json_out: Path = typer.Option(None, "--json", help="Write all units to this JSON file"),
) -> None:
    """Parse an addon offline (no embeddings) and print unit counts by kind."""
    import collections
    import dataclasses
    import json

    from rag.sources.odoo_module import DirSource, GitSource, git_location, latest_tag, parse_module

    path = path.resolve()
    if not (path / "__manifest__.py").exists() and not ref and not tag_prefix:
        fail(f"{path} has no __manifest__.py")
    if tag_prefix or ref:
        repo, prefix = git_location(path)
        if tag_prefix:
            ref = latest_tag(repo, tag_prefix, series) or ""
            if not ref:
                fail(f"no tag {tag_prefix}-{series}.* in {repo}")
        source = GitSource(repo, ref, prefix)
    else:
        source = DirSource(path)
    units = parse_module(source, path.name, origin=origin)
    version = next((u.module_version for u in units if u.kind == "manifest"), "?")
    typer.echo(f"{path.name} {version} ({ref or 'working tree'}): {len(units)} units")
    for kind, n in sorted(collections.Counter(u.kind for u in units).items()):
        typer.echo(f"  {kind:<10}{n:>5}")
    if json_out:
        json_out.write_text(json.dumps([{**dataclasses.asdict(u), "id": u.id} for u in units], indent=1))
        typer.echo(f"wrote {json_out}")


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


@app.command()
def search(
    query: str,
    k: int = typer.Option(5, "-k", help="Number of pages"),
    mode: str = typer.Option("hybrid", help="hybrid | vector | fts", callback=check_mode),
    product: str | None = typer.Option(None, help="Filter by product, e.g. 'Direct Print'"),
    rerank: bool = typer.Option(False, help="Rerank candidates with a cross-encoder"),
    model: str = typer.Option(config.EMBED_MODEL, help="Embedding model (selects the index)"),
    show_text: bool = typer.Option(False, "--text", help="Print matched chunk text"),
) -> None:
    """Search the index from the terminal."""
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


@app.command()
def ask(
    query: str,
    mode: str = typer.Option("hybrid", help="hybrid | vector | fts", callback=check_mode),
    product: str | None = typer.Option(None, help="Filter by product, e.g. 'Direct Print'"),
    rerank: bool = typer.Option(True, help="Rerank and drop low-relevance sources (--no-rerank: off)"),
    llm_model: str = typer.Option(config.LLM_MODEL, help="LLM id (default: the one loaded in LM Studio, or claude-sonnet-5-5)"),
) -> None:
    """Answer a question with the LLM (RAG_LLM_PROVIDER), grounded in the search results."""
    from rag.answer import stream_answer
    from rag.llm import make_llm

    hits = open_searcher().search(query, k=config.LLM_CONTEXT_PAGES, mode=mode, product=product, rerank=rerank)
    for event in stream_answer(query, hits, make_llm(model=llm_model)):
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
                typer.secho(f"[{s['n']}] {s['title']}  {s['url']}", dim=True)
            typer.echo()


@app.command()
def bench(
    models: list[str] = typer.Argument(None, help="LLM ids to compare (default: the loaded one)"),
    file: Path = typer.Option(config.BENCH_FILE, help="YAML list of questions"),
    group: str | None = typer.Option(None, help="Only questions of this group, e.g. 'novice'"),
    out: Path | None = typer.Option(None, help="Markdown report (default: data/bench/<time>.md)"),
    mode: str = typer.Option("hybrid", help="hybrid | vector | fts", callback=check_mode),
    rerank: bool = typer.Option(True, help="Rerank and drop low-relevance sources"),
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
        items = run_bench(searcher, llm, questions, models, mode=mode, rerank=rerank, log=typer.echo, on_item=write)
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
