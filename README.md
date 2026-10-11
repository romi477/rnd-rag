# rnd-rag

Local RAG over a Confluence space: hybrid retrieval (BM25 + embeddings), optional answers with
citations from a local LLM, and a small web UI.
Everything runs on your machine: documents, index and models never leave it.

**Status:** stage 1 (retrieval + search UI) and stage 2 (LLM answers with citations) are done.

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
                                    │
                                    ▼ (optional, "Answer with LLM")
            top-5 pages as numbered sources ──▶ local LLM (LM Studio) ──▶ streamed answer with [n] citations
```

The UI works like a browser: every tab keeps its own search (query, filters, chosen LLM, results
and answer) and tabs survive a page reload. To compare models, run the same question in two tabs
with different models. Models marked "(not loaded)" are loaded by LM Studio on first use.
Result cards collapse to their title: the answer and the first card start expanded, the rest
collapsed, and a citation [n] expands the card it points to.
Only the button runs a query: it reads "Find", or "Ask" when "Answer with LLM" is on. Changing
settings afterwards marks the results as outdated until the next run. The eraser icon clears the tab.
Cards are numbered by rank; with an LLM answer, the cards used as sources keep a filled badge with
their source number. The bug icon turns on debug mode: under each card and fragment it shows how the
fragment was scored (`vector #rank cosine · bm25 #rank score · rrf · retrieval #rank · rerank`).

Search modes:

| Mode | Finds | Good at | Weak at |
|---|---|---|---|
| Keyword (`fts`) | chunks containing the query words (BM25) | error codes, module names, settings | synonyms, paraphrases |
| Semantic (`vector`) | chunks with similar meaning (embeddings) | paraphrased questions | rare exact tokens |
| Hybrid (default) | rank fusion of both lists | robust default | — |
| + Rerank (UI default) | re-scores the top 10 chunks with a cross-encoder that reads query and chunk together; pages below the cut-off are collapsed and not sent to the LLM | precise ordering, meaningful 0..1 scores, drops junk | +~0.5 s per query |

## Requirements

- macOS (Apple Silicon uses the GPU via MPS) or Linux; Python 3.13 is installed by `uv`
- [uv](https://docs.astral.sh/uv/)
- ~4 GB disk for models (downloaded on first use to the Hugging Face cache)
- Confluence Cloud account with an [API token](https://id.atlassian.com/manage-profile/security/api-tokens)
- For answers (optional): [LM Studio](https://lmstudio.ai) with a chat model loaded and the local server
  started (Developer → Start Server, port 1234), or any other OpenAI-compatible server

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
| `sync.sh [--force] [--attachments]` | download new/changed pages into `raw-data/`; `--reconvert` rebuilds the markdown offline |
| `index.sh [--force] [--model M]` | chunk + embed changed pages into `data/lancedb/` |
| `serve.sh [port]` | web UI + JSON API (`RAG_HOST` / `RAG_PORT` env vars also work) |
| `search.sh "query" [--text]` | search from the terminal |
| `ask.sh "question" [--no-rerank]` | search + streamed LLM answer with sources in the terminal |
| `eval.sh [--misses]` | retrieval quality on `eval/queries.yaml` |
| `bench.sh [MODEL ...] [--group G]` | compare LLMs on `eval/bench.yaml`: latency, tokens/s, answers side by side |
| `test.sh` | unit tests |

The scripts are thin wrappers over the `rag` CLI: `uv run rag --help`.
A running server picks up index updates without a restart: the indexer replaces a page's chunks
in one commit, so searches never see a page missing, even during `index.sh --force`.

Safety nets (each one lifted by `--force`): a sync that would delete more than a quarter of the local
pages stops (an empty or partial listing from Confluence), and so does an index run that would
drop more than a quarter of the indexed pages or index another space into the same directory.
The index also records the chunking settings and rebuilds itself when they change.

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
| `RAG_RERANK_CANDIDATES` | `10`, chunks re-scored by the cross-encoder |
| `RAG_RERANK_MIN_SCORE` | `0.1`, relevance cut-off on rerank scores (pages below are flagged `low_relevance`) |
| `RAG_CHUNK_MAX_TOKENS` / `RAG_CHUNK_MIN_TOKENS` | `500` / `80` |
| `RAG_LLM_BASE_URL` | `http://127.0.0.1:1234/v1` (LM Studio; any OpenAI-compatible server works) |
| `RAG_LLM_MODEL` | empty: use the model loaded in LM Studio |
| `RAG_LLM_REASONING` | `none`: sent as `reasoning_effort`, turns off "thinking" for faster answers; empty: not sent |
| `RAG_LLM_TEMPERATURE` / `RAG_LLM_MAX_TOKENS` | `0.2` / `1000` |
| `RAG_LLM_CONTEXT_PAGES` / `RAG_LLM_CONTEXT_TOKENS` | `5` / `4000`: how much of the search result goes into the prompt |
| `RAG_ALLOWED_HOSTS` | `127.0.0.1,localhost`: host names the server answers to (see below) |

The server binds to `127.0.0.1` and answers only to requests addressed to `127.0.0.1` or `localhost`,
so a web page elsewhere can't reach it by DNS rebinding. It has no login: binding to another
interface (`RAG_HOST=0.0.0.0`) shares the internal docs with the network, and then the machine's
name has to go into `RAG_ALLOWED_HOSTS`. There are no `/docs` pages (they load scripts from a CDN).

Each embedding model gets its own index directory, so models can be compared side by side:
`scripts/index.sh --model BAAI/bge-base-en-v1.5`, then `scripts/eval.sh --model BAAI/bge-base-en-v1.5`.

Models are cached in `~/.cache/huggingface` (override with `HF_HOME`). After the first download,
`HF_HUB_OFFLINE=1` skips the update check on startup.

## API

`GET /api/search?q=...&k=10&mode=hybrid|vector|fts&product=...&rerank=false`

Quotes and backslashes in `q` are ignored by the BM25 side (no phrase queries); if the BM25 index is
missing (an interrupted rebuild), hybrid search falls back to vectors only.

```json
{
  "query": "...", "took_ms": 54,
  "results": [{
    "page_id": "...", "title": "...", "url": "...", "product": "...", "breadcrumbs": ["..."],
    "updated": "...", "labels": [], "score": 0.93, "low_relevance": false,
    "chunks": [{"chunk_id": "...", "section_path": ["..."], "text": "...", "score": 0.021}]
  }]
}
```

`GET /api/answer` takes the same parameters (but `rerank` is on by default, so that unrelated pages
stay out of the prompt) plus optional `model` (an LLM id known to the LLM server; default: the model
loaded in LM Studio) and streams NDJSON (one JSON event per line).
The search runs once: its results come first, then the answer.

```
{"type": "results", "query": "...", "took_ms": 54, "results": [...]}   same as /api/search
{"type": "sources", "sources": [{"n": 1, "page_id": "...", "title": "...", "url": "..."}]}
{"type": "delta", "text": "The error means ... [1]"}                  repeated while streaming
{"type": "done", "model": "...", "took_ms": 1771, "truncated": false}  or {"type": "error", "message": "..."}
```

`truncated` is true when the answer hit `RAG_LLM_MAX_TOKENS`. A stream that the LLM server closes
early, or an answer with no text (a reasoning model that spent the budget thinking), is an `error`.

`GET /api/llm` reports whether answer generation is available, the default model and all chat
models with their load state (`{"available", "model", "models": [{"id", "loaded"}], "error"}`).
`GET /api/products` lists the values for the `product` filter.

## Evaluation

`eval/queries.yaml` is a golden set of questions with the page ids that answer them.
The real file is git-ignored because it is derived from internal docs. Start from the example:

```bash
cp eval/queries.example.yaml eval/queries.yaml
scripts/eval.sh --misses                        # hit@1/3/5/10 and MRR per mode
scripts/eval.sh --fts-weights 0,0.3,0.5,1       # tune hybrid fusion
```

`eval/bench.yaml` holds harder questions for comparing LLMs (also git-ignored): multi-step questions,
or the way a first-time user asks, with pasted errors and vague wording. Each item may have a `group`
and `expect` (page ids that answer it; `[]` means the docs don't cover it, so the model should say so).

```bash
cp eval/bench.example.yaml eval/bench.yaml
scripts/bench.sh google/gemma-4-e4b qwen/qwen3.8-27b       # all questions, both models
scripts/bench.sh qwen/qwen3.8-27b --group novice           # one group
```

The models answer the same questions from the same sources, with the prompt the UI uses. The
terminal shows time to the first token, answer time and tokens/s per model, plus how often the
expected page made it into the sources. The Markdown report in `data/bench/` puts the answers side
by side, to be checked against the docs (quality is not scored automatically). Models that are not
loaded in LM Studio get loaded during the warm-up, which is not measured; a model that fails it
stops the bench. The report is rewritten after every question, so Ctrl-C keeps what was measured.
Tokens/s counts all generated tokens (reasoning included) over the time from the first generated
token, summed over all answers. A custom `--out` must be git-ignored: the report quotes the docs.

## Layout

```
src/rag/                 backend (Python package)
  sources/confluence.py  Confluence REST v2 sync: storage HTML -> markdown + metadata
  chunking.py            heading-aware chunking with context headers
  embedding.py           sentence-transformers embedder and cross-encoder reranker
  store.py               LanceDB table, per-page upsert, BM25 index
  indexing.py            incremental indexing by page version
  search.py              vector / BM25 / weighted-RRF hybrid, rerank, group by page
  llm.py                 minimal OpenAI-compatible streaming client, model auto-detection
  answer.py              numbered sources + prompt -> streamed answer events
  evaluation.py          hit@k / MRR over the golden set
  bench.py               LLM comparison: latency, tokens/s, answers side by side
  api.py                 FastAPI app: JSON API + serves src/web/
  cli.py                 `rag` command (sync, index, search, ask, bench, eval, serve)
src/web/                 UI: index.html, styles.css, app.js (vanilla JS, no build step)
scripts/                 bash entry points (see above)
tests/                   unit tests (fake embedder and mocked LLM server: no downloads, no network)
```

Not in git: `raw-data/` (synced pages), `data/` (index), `_docs/` (internal notes),
`eval/*.yaml` except the examples, `.env` and its variants.

## Roadmap

- [x] Confluence sync, chunking, local embeddings, hybrid search, web UI, evaluation
- [x] Answer generation with citations from a local LLM (stage 2)
- [x] Relevance cut-off (rerank score threshold)
- [ ] Answer quality evaluation
- [ ] MCP server exposing search to coding agents
