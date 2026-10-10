# AGENTS.md

Guidance for AI coding agents working in this repository. Read `README.md` for the user-facing overview.

## Project

Local RAG over the internal Confluence space `pd` ("R&D: Products Documentation", ~250 pages, English).
- Stage 1 (done): sync, chunking, local embeddings, hybrid search, web UI, evaluation.
- Stage 2 (next): answer generation with citations on top of `Searcher.search()`.

## Hard rules

- **The GitHub repo is public.** Never commit or `git add -f` these paths: `raw-data/`, `data/`,
  `_docs/`, `eval/queries.yaml`, `.env`. They hold internal documentation, text derived from it,
  or personal data. Keep the `.gitignore` entries.
- Do not put internal page text, page titles or customer details in code, tests, commits or the README.
  Tests use synthetic data.
- Never print, log or copy the Confluence token. It comes from `CONFLUENCE_API_TOKEN` or falls back
  to `$JIRA_TOKEN_VENT`, which is defined only in the user's interactive zsh profile.
  Run syncs through `scripts/sync.sh` (it goes through `zsh -ic`), and don't dump the environment.
- Everything stays local: no external embedding or LLM APIs unless the user decides otherwise.

## Commands

```bash
scripts/setup.sh              # uv sync, create .env
scripts/update.sh             # sync Confluence + incremental index
scripts/index.sh --force      # full rebuild (needed after chunking changes)
scripts/search.sh "query" --text
scripts/serve.sh [port]       # http://127.0.0.1:8000
scripts/eval.sh --misses      # retrieval metrics
scripts/test.sh               # pytest
```

Use `uv` only (`uv add`, `uv run`), no `pip`. Python 3.13 (`.python-version`); dependencies live in `pyproject.toml`.

## Architecture

| Module | Responsibility |
|---|---|
| `src/rag/config.py` | paths and tunables from env / `.env` |
| `src/rag/sources/confluence.py` | Confluence REST v2 → `raw-data/confluence/<space>/pages/<id>.{md,json,storage.html}` + `_manifest.json` |
| `src/rag/chunking.py` | `Page` → `Chunk`s: split on headings, merge small, split large on paragraphs, keep code/tables whole, context header in `embed_text` |
| `src/rag/embedding.py` | `Embedder` (sentence-transformers, MPS) and `Reranker` (cross-encoder), loaded lazily |
| `src/rag/store.py` | LanceDB table `chunks`, one directory per embedding model (`data/lancedb/<model>/`), upsert per page, FTS index |
| `src/rag/indexing.py` | incremental indexing by page `version`; a model change triggers a full rebuild |
| `src/rag/search.py` | `Searcher`: `fts` / `vector` / `hybrid` (own weighted RRF), optional rerank, group by page |
| `src/rag/evaluation.py` | hit@k and MRR over `eval/queries.yaml` |
| `src/rag/api.py` | FastAPI: `/api/search`, `/api/products`; mounts `src/web/` (`config.WEB_DIR`) at `/` |
| `src/web/` | UI, separate from the backend: `index.html`, `styles.css`, `app.js` (vanilla JS, no build step); talks to the backend only via `/api/*` |
| `src/rag/cli.py` | `rag` typer CLI; `scripts/*.sh` are thin wrappers over it |

## Conventions

- Code, comments, docstrings, commit messages and README are in English. Planning notes and
  results go to `_docs/` (git-ignored), in Russian; talk to the user in Russian.
- Keep modules small and explicit, with no LangChain/LlamaIndex-style frameworks. Match the existing style.
- Tests must not download models: use the fake embedder pattern from `tests/test_store_search.py`.
  Run `scripts/test.sh` before finishing a change.
- Retrieval changes (chunking, fusion, models, rerank) must be backed by `scripts/eval.sh` numbers,
  before and after. With ~40 queries, one query equals 0.025 hit@1, so don't over-read small deltas.
- Chunking changes are not detected by the incremental indexer: run `scripts/index.sh --force` afterwards.
- New user-facing actions get a script in `scripts/` (source `_common.sh`; bash 3.2 compatible)
  plus a row in the README scripts table.

## Library notes

- LanceDB 0.40 (local):
  - `db.table_exists()` raises `NotImplementedError`, use `TABLE in db.list_tables().tables`;
  - build FTS with `table.create_index("embed_text", config=FTS(...), replace=True)`;
    `create_fts_index` is deprecated;
  - include `_score` / `_distance` explicitly in `.select()` to avoid scoring-autoprojection warnings.
- Qwen3-Embedding: queries use `prompt_name="query"`, documents use no prompt; left padding is set
  via `processor_kwargs`. bge models use an explicit query prefix (`QUERY_PROMPTS` in `embedding.py`).
- Check library APIs against current docs (Context7) rather than memory. These libraries change fast.
