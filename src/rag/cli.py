"""`rag` command line: sync, index, search, eval, serve."""

from __future__ import annotations

import textwrap

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
        typer.secho(f"{i}. {page.title}  [{page.score:.3f}]", bold=True)
        typer.echo(f"   {' > '.join(page.breadcrumbs[1:]) or page.product} | {page.url}")
        for chunk in page.chunks:
            section = " > ".join(chunk.section_path) or "(page)"
            typer.echo(f"   - {section} [{chunk.score:.3f}]")
            if show_text:
                typer.echo(textwrap.indent(textwrap.shorten(chunk.text, 400), "     "))
    if not hits:
        typer.echo("No results")


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
