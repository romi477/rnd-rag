# rnd-rag

Local search over a Confluence space: hybrid retrieval (BM25 + embeddings) with a small web UI.
Everything runs on your machine: documents, index and models never leave it.

**Status:** stage 1, retrieval + search UI. Stage 2 (LLM answers with citations) is planned.

## How it works

```
Confluence ──sync──▶ raw-data/          markdown + metadata per page (incremental by page version)
                        │
                        ▼ chunking       split by headings, merge small / split large sections,
                        │                prepend "Product > Breadcrumbs > Title > Section" header
                        ▼
              ┌─────────┴──────────┐
              ▼                    ▼
     embeddings (local)       BM25 full-text index
     Qwen3-Embedding-0.6B          │
              └────────┬───────────┘
                       ▼
                LanceDB (data/lancedb/, embedded, no server)
                       │
   query ──▶ vector top-40 + BM25 top-40 ──▶ weighted RRF fusion ──▶ [optional cross-encoder rerank]
                       ──▶ group chunks by page ──▶ web UI / JSON API / CLI
```

Search modes:

| Mode | Finds | Good at | Weak at |
|---|---|---|---|
| Keyword (`fts`) | chunks containing the query words (BM25) | error codes, module names, settings | synonyms, paraphrases |
| Semantic (`vector`) | chunks with similar meaning (embeddings) | paraphrased questions | rare exact tokens |
| Hybrid (default) | rank fusion of both lists | robust default | — |
| + Rerank | re-scores top candidates with a cross-encoder that reads query and chunk together | precise ordering, meaningful scores | +1–2 s per query |

## Requirements

- macOS (Apple Silicon uses the GPU via MPS) or Linux; Python 3.13 is installed by `uv`
- [uv](https://docs.astral.sh/uv/)
- ~4 GB disk for models (downloaded on first use to the Hugging Face cache)
- Confluence Cloud account with an [API token](https://id.atlassian.com/manage-profile/security/api-tokens)

## Quick start

```bash
scripts/setup.sh      # install deps, create .env from .env.example (set CONFLUENCE_EMAIL)
scripts/update.sh     # download Confluence pages and build the index
scripts/serve.sh      # web UI at http://127.0.0.1:8000
```

## Scripts

| Script | What it does |
|---|---|
| `setup.sh` | `uv sync`; creates `.env` from `.env.example` if missing |
| `update.sh [--force]` | `sync.sh` + `index.sh`; run it when docs change in Confluence |
| `sync.sh [--force] [--attachments]` | download new/changed pages into `raw-data/` |
| `index.sh [--force] [--model M]` | chunk + embed changed pages into `data/lancedb/` |
| `serve.sh [port]` | web UI + JSON API (`HOST` / `PORT` env vars also work) |
| `search.sh "query" [--text]` | search from the terminal |
| `eval.sh [--misses]` | retrieval quality on `eval/queries.yaml` |
| `test.sh` | unit tests |

The scripts are thin wrappers over the `rag` CLI: `uv run rag --help`.
A running server picks up index updates without a restart.

## Configuration

`.env` (see `.env.example`):

| Variable | Meaning |
|---|---|
| `CONFLUENCE_BASE_URL` | e.g. `https://<site>.atlassian.net` |
| `CONFLUENCE_EMAIL` | Atlassian account email |
| `CONFLUENCE_API_TOKEN` | API token; if empty, `$JIRA_TOKEN_VENT` from the shell profile is used |
| `CONFLUENCE_SPACE_KEY` | space to sync, default `pd` |

Tuning (env, see `src/rag/config.py`):

| Variable | Default |
|---|---|
| `RAG_EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` |
| `RAG_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` |
| `RAG_HYBRID_FTS_WEIGHT` | `0.3`, weight of BM25 relative to vectors in RRF fusion |
| `RAG_CHUNK_MAX_TOKENS` / `RAG_CHUNK_MIN_TOKENS` | `500` / `80` |

Each embedding model gets its own index directory, so models can be compared side by side:
`scripts/index.sh --model BAAI/bge-base-en-v1.5`, then `scripts/eval.sh --model BAAI/bge-base-en-v1.5`.

Models are cached in `~/.cache/huggingface` (override with `HF_HOME`). After the first download,
`HF_HUB_OFFLINE=1` skips the update check on startup.

## API

`GET /api/search?q=...&k=10&mode=hybrid|vector|fts&product=...&rerank=false`

```json
{
  "query": "...", "took_ms": 54,
  "results": [{
    "page_id": "...", "title": "...", "url": "...", "product": "...", "breadcrumbs": ["..."],
    "updated": "...", "labels": [], "score": 0.021,
    "chunks": [{"chunk_id": "...", "section_path": ["..."], "text": "...", "score": 0.021}]
  }]
}
```

`GET /api/products` lists the values for the `product` filter.

## Evaluation

`eval/queries.yaml` is a golden set of questions with the page ids that answer them.
The real file is git-ignored because it is derived from internal docs. Start from the example:

```bash
cp eval/queries.example.yaml eval/queries.yaml
scripts/eval.sh --misses                        # hit@1/3/5/10 and MRR per mode
scripts/eval.sh --fts-weights 0,0.3,0.5,1       # tune hybrid fusion
```

## Layout

```
src/rag/                 backend (Python package)
  sources/confluence.py  Confluence REST v2 sync: storage HTML -> markdown + metadata
  chunking.py            heading-aware chunking with context headers
  embedding.py           sentence-transformers embedder and cross-encoder reranker
  store.py               LanceDB table, per-page upsert, BM25 index
  indexing.py            incremental indexing by page version
  search.py              vector / BM25 / weighted-RRF hybrid, rerank, group by page
  evaluation.py          hit@k / MRR over the golden set
  api.py                 FastAPI app: JSON API + serves src/web/
  cli.py                 `rag` command (sync, index, search, eval, serve)
src/web/                 UI: index.html, styles.css, app.js (vanilla JS, no build step)
scripts/                 bash entry points (see above)
tests/                   unit tests (a fake embedder is used, no model downloads)
```

Not in git: `raw-data/` (synced pages), `data/` (index), `_docs/` (internal notes),
`eval/queries.yaml`, `.env`.

## Roadmap

- [x] Confluence sync, chunking, local embeddings, hybrid search, web UI, evaluation
- [ ] Relevance cut-off (rerank score threshold)
- [ ] Answer generation with citations (stage 2)
- [ ] MCP server exposing search to coding agents
