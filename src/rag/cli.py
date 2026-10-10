"""`rag` command line: sync, index, search, ask, bench, eval, serve."""

from __future__ import annotations

import textwrap
from pathlib import Path

import typer

from rag import config

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def sync(
    space: str = typer.Option(config.SPACE_KEY, help="Confluence space key"),
    attachments: bool = typer.Option(False, help="Also download page attachments"),
    force: bool = typer.Option(False, help="Rewrite pages even if version unchanged"),
) -> None:
    """Download/refresh Confluence pages into raw-data/."""
    from rag.sources.confluence import sync as run_sync

    run_sync(space, config.RAW_DIR, attachments=attachments, force=force)


@app.command()
def index(
    space: str = typer.Option(config.SPACE_KEY, help="Confluence space key"),
    model: str = typer.Option(config.EMBED_MODEL, help="Embedding model"),
    force: bool = typer.Option(False, help="Rebuild the whole index"),
) -> None:
    """Chunk + embed synced pages into LanceDB (incremental by page version)."""
    from rag.embedding import Embedder
    from rag.indexing import index_space

    stats = index_space(space, embedder=Embedder(model), force=force, log=typer.echo)
    typer.echo(
        f"pages={stats.pages} updated={stats.updated} removed={stats.removed} "
        f"chunks_written={stats.chunks_written} total_chunks={stats.total_chunks}"
    )


@app.command()
def search(
    query: str,
    k: int = typer.Option(5, "-k", help="Number of pages"),
    mode: str = typer.Option("hybrid", help="hybrid | vector | fts"),
    product: str | None = typer.Option(None, help="Filter by product, e.g. 'Direct Print'"),
    rerank: bool = typer.Option(False, help="Rerank candidates with a cross-encoder"),
    model: str = typer.Option(config.EMBED_MODEL, help="Embedding model (selects the index)"),
    show_text: bool = typer.Option(False, "--text", help="Print matched chunk text"),
) -> None:
    """Search the index from the terminal."""
    from rag.embedding import Embedder
    from rag.search import Searcher

    searcher = Searcher(Embedder(model))
    hits = searcher.search(query, k=k, mode=mode, product=product, rerank=rerank)
    for i, page in enumerate(hits, 1):
        low = "  (low relevance)" if page.low_relevance else ""
        typer.secho(f"{i}. {page.title}  [{page.score:.3f}]{low}", bold=not page.low_relevance, dim=page.low_relevance)
        typer.echo(f"   {' > '.join(page.breadcrumbs[1:]) or page.product} | {page.url}")
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
    mode: str = typer.Option("hybrid", help="hybrid | vector | fts"),
    product: str | None = typer.Option(None, help="Filter by product, e.g. 'Direct Print'"),
    rerank: bool = typer.Option(True, help="Rerank and drop low-relevance sources"),
    llm_model: str = typer.Option(config.LLM_MODEL, help="LLM id (default: the one loaded in LM Studio)"),
) -> None:
    """Answer a question with the local LLM, grounded in the search results."""
    from rag.answer import stream_answer
    from rag.llm import LLMClient
    from rag.search import Searcher

    hits = Searcher().search(query, k=config.LLM_CONTEXT_PAGES, mode=mode, product=product, rerank=rerank)
    for event in stream_answer(query, hits, LLMClient(model=llm_model)):
        if event["type"] == "delta":
            typer.echo(event["text"], nl=False)
        elif event["type"] == "error":
            typer.secho(f"\nLLM error: {event['message']}", fg="red", err=True)
            raise typer.Exit(1)
        elif event["type"] == "done":
            typer.echo()
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
    mode: str = typer.Option("hybrid", help="hybrid | vector | fts"),
    rerank: bool = typer.Option(True, help="Rerank and drop low-relevance sources"),
) -> None:
    """Compare LLMs on the same questions and sources: latency, tokens/s, answers side by side."""
    from rag.bench import (
        default_report_path, load_questions, render_report, retrieval_line, run_bench, summary_rows,
    )
    from rag.llm import LLMClient, LLMError
    from rag.search import Searcher

    if not file.exists():
        typer.secho(f"No questions file {file}: cp eval/bench.example.yaml eval/bench.yaml", fg="red", err=True)
        raise typer.Exit(1)
    questions = load_questions(file, group)
    if not questions:
        typer.secho(f"No questions in {file}" + (f" for group '{group}'" if group else ""), fg="red", err=True)
        raise typer.Exit(1)
    llm = LLMClient()
    try:
        models = models or [llm.resolve_model()]
    except LLMError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1)

    items = run_bench(Searcher(), llm, questions, models, mode=mode, rerank=rerank, log=typer.echo)

    settings = (
        f"Questions: {len(questions)} ({file.name}{f', group {group}' if group else ''}) · search: {mode}{' + rerank' if rerank else ''} · "
        f"temperature {config.LLM_TEMPERATURE} · max_tokens {config.LLM_MAX_TOKENS} · "
        f"reasoning {config.LLM_REASONING or 'default'}"
    )
    out = out or default_report_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(items, models, settings))

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

    uvicorn.run("rag.api:app", host=host, port=port, reload=reload)


@app.command("eval")
def eval_cmd(
    model: str = typer.Option(config.EMBED_MODEL, help="Embedding model (selects the index)"),
    modes: str = typer.Option("fts,vector,hybrid", help="Comma-separated modes"),
    rerank: bool = typer.Option(True, help="Also evaluate hybrid + cross-encoder rerank"),
    misses: bool = typer.Option(False, help="List queries whose answer is not ranked first"),
    fts_weights: str = typer.Option(
        str(config.HYBRID_FTS_WEIGHT), help="Comma-separated BM25 weights to try for hybrid"
    ),
) -> None:
    """Compare retrieval modes on eval/queries.yaml (page-level hit@k and MRR)."""
    from rag.embedding import Embedder
    from rag.evaluation import evaluate, load_queries
    from rag.search import Searcher

    queries = load_queries(config.EVAL_FILE)
    searcher = Searcher(Embedder(model))
    weights = [float(w) for w in fts_weights.split(",")]
    runs = []  # (name, mode, rerank, fts_weight)
    for mode in modes.split(","):
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
