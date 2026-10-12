# AGENTS.md

Guidance for AI coding agents working in this repository. Read `README.md` for the user-facing overview.

## Project

Local RAG over the internal Confluence space `pd` ("R&D: Products Documentation", ~250 pages, English).
- Stage 1 (done): sync, chunking, local embeddings, hybrid search, web UI, evaluation.
- Stage 2 (done): answers with [n] citations from a local LLM (LM Studio, OpenAI-compatible API)
  on top of `Searcher.search()`; UI toggle "Answer with LLM".
- Stage 4: the Claude API (`claude-sonnet-5-5`) as an alternative LLM, `RAG_LLM_PROVIDER=anthropic`.

## Hard rules

- **The GitHub repo is public.** Never commit or `git add -f` these paths: `raw-data/`, `data/`,
  `_docs/`, `eval/*.yaml` (except `*.example.yaml`), `.env` and `.env.*` (except `.env.example`), bench reports.
  They hold internal documentation, text derived from it, or personal data. Keep the `.gitignore` entries.
- Do not put internal page text, page titles or customer details in code, tests, commits or the README.
  Tests use synthetic data.
- Never print, log or copy the Confluence token. It comes from `CONFLUENCE_API_TOKEN` or falls back
  to `$JIRA_TOKEN_VENT`, which is defined only in the user's interactive zsh profile.
  Run syncs through `scripts/sync.sh` (it goes through `zsh -ic`), and don't dump the environment.
- Everything stays local except what the user decided otherwise: on 2026-10-12 the user allowed sending the
  question and doc excerpts to the Claude API for answers (`RAG_LLM_PROVIDER=anthropic`; the docs describe products
  that are sold). Embeddings, rerank, the index and the raw pages stay local; no other external APIs.
  `ANTHROPIC_API_KEY` lives in `.env` only: never print, log or commit it.
- Source code of the Odoo addons (`data/code/`, the `code_units` table) is not documentation: that permission does
  not cover it. Code answers refuse the Claude provider unless the user sets `RAG_CODE_EXTERNAL_LLM=1`; do not run
  a code answer through it on your own.

## Commands

```bash
scripts/setup.sh              # uv sync, create .env
scripts/update.sh             # sync Confluence + incremental index
scripts/index.sh --force      # full rebuild (chunking changes rebuild by themselves)
scripts/sync.sh --reconvert   # offline: rebuild raw-data/*.md from the stored HTML (after converter changes)
scripts/search.sh "query" --text
scripts/ask.sh "question"     # search + LLM answer (LM Studio on :1234 with a model loaded, or the Claude API)
scripts/serve.sh [port] [--offline]   # http://127.0.0.1:8000; --offline: no Hugging Face update check
scripts/eval.sh --misses      # retrieval metrics
scripts/bench.sh M1 M2 [--group novice]   # compare LLMs on eval/bench.yaml, report in data/bench/
scripts/test.sh               # pytest
```

Use `uv` only (`uv add`, `uv run`), no `pip`. Python 3.13 (`.python-version`); dependencies live in `pyproject.toml`.

## Architecture

| Module | Responsibility |
|---|---|
| `src/rag/config.py` | paths and tunables from env / `.env` |
| `src/rag/sources/confluence.py` | Confluence REST v2 → `raw-data/confluence/<space>/pages/<id>.{md,json,storage.html}` + `_manifest.json`; retries 429/5xx, writes atomically, known pages come from `pages/*.json`; refuses to delete >25% of pages without `--force` |
| `src/rag/chunking.py` | `Page` → `Chunk`s: split on headings, merge small, split large on paragraphs, keep code/tables whole, context header in `embed_text` |
| `src/rag/embedding.py` | `Embedder` (sentence-transformers, MPS) and `Reranker` (cross-encoder), loaded lazily |
| `src/rag/store.py` | LanceDB table `chunks`, one directory per embedding model (`data/lancedb/<model>/`); `upsert_pages` replaces and deletes pages in one `merge_insert` commit after validating the vectors; FTS index (created with the table) |
| `src/rag/indexing.py` | incremental indexing by page version + title / breadcrumbs / labels; `index_meta.json` records model, space and `chunk_settings()`: a change rebuilds in place; refuses an empty raw dir, another space, or dropping >25% of pages without `force`; restores a missing FTS index |
| `src/rag/search.py` | `Searcher`: `fts` / `vector` / `hybrid` (own weighted RRF; quotes stripped for BM25, falls back to vector if FTS fails; unknown mode = `ValueError`), optional rerank of the top `RERANK_CANDIDATES` chunks, group by page; with rerank, pages below `RERANK_MIN_SCORE` get `low_relevance=True` (UI collapses them, `answer.py` skips them); every `ChunkHit.signals` records how it was scored (vector / BM25 rank and score, RRF, retrieval rank, rerank) for the UI debug mode |
| `src/rag/llm.py` | `make_llm()` picks the client by `RAG_LLM_PROVIDER`; both have `list_models` / `resolve_model` / `status` / `stream_chat(info=)`. `ClaudeClient`: official `anthropic` SDK (a missing `ANTHROPIC_API_KEY` is an `LLMError`, not the SDK's `TypeError`), `client.beta.messages.stream`, system prompt as `system=`, no `temperature`, `RAG_LLM_REASONING=none` = `thinking: between_tools` on Sonnet 5.5 (effort `low` elsewhere), server-side refusal fallback (`fallbacks: "default"`) on the models that take it, `max_tokens` stop = `finish_reason: length`, refusal = `LLMError`, usage mapped to `prompt_tokens` / `completion_tokens`, adaptive thinking gets `THINKING_HEADROOM` added to `max_tokens`; model ids match their dated variants (`same_model`); models from `/v1/models` (reused for 5 minutes, `status()` refreshes) limited to `RAG_CLAUDE_MODELS` (default Sonnet 5.5 + Haiku 5.5; Haiku runs with `thinking: disabled`) (`loaded: None`). `LLMClient`: OpenAI-compatible streaming chat over httpx (no SDK, `trust_env=False`); SSE split on `\n` only; `stream_chat(info=)` reports `finish_reason`, `usage`, `first_delta_at`; a stream without `[DONE]`/finish reason is an `LLMError`; `list_models()` (LM Studio `/api/v0/models` with load state, fallback `/v1/models`); default model = `RAG_LLM_MODEL` or the loaded one; `/api/answer?model=` overrides per request |
| `src/rag/answer.py` | top pages → numbered sources within a token budget → system prompt → `stream_answer()` events (`sources`, `delta`, `done` with `truncated`, `first_token_ms` and `usage` (`token_usage()`; shown in the UI debug mode) / `error`, also for an empty answer) |
| `src/rag/evaluation.py` | hit@k and MRR over `eval/queries.yaml` |
| `src/rag/bench.py` | `rag bench`: the same questions and sources to several LLMs; time to first token, total time, tokens/s (from `usage`, filled in by `stream_chat(info=...)`), whether the `expect`ed page is among the sources; Markdown report with the answers side by side |
| `src/rag/api.py` | FastAPI: `/api/search`, `/api/answer` (NDJSON stream, rerank on by default, `model` checked against the LLM server, `RAG_ANSWERS_PER_HOUR` limit = 429, an unexpected failure ends the stream with an `error` event), `/api/llm` (with `provider`; the UI marks "via Claude API"), `/api/products`, `/api/pages` (every indexed page with its Confluence link, `Store.pages()`); mounts `src/web/` (`config.WEB_DIR`) at `/`. `TrustedHostMiddleware` (`RAG_ALLOWED_HOSTS`), no `/docs`. UI files are sent with `Cache-Control: no-cache` (no build step to version them). Models warm up in a background thread; the search runs under a lock (MPS), the LLM stream outside it |
| `src/web/` | UI, separate from the backend: `index.html`, `styles.css`, `app.js` (vanilla JS, no build step); talks to the backend only via `/api/*`. The UI always searches hybrid + rerank (`SEARCH_PARAMS`; no Mode / Rerank controls, older tabs and links with other values are ignored), other modes stay in the API and CLI. Browser-like tabs: each tab is an independent search state (form, results, answer, LLM model), persisted in `localStorage` (`rag.tabs.v1`; saved at most twice a second while streaming, a tab saved mid-request is stored as interrupted, windows merge each other's tabs via the `storage` event); a background tab keeps streaming. The Search / Browse switch (`tab.view`, title filter `tab.filter`; the article list from `/api/pages` and the product list load once, a failed load is retried) mirrors the Odoo add-on `integration_rag`: keep the two UIs in step (Stop, error banner, Browse). Light / dark theme button (`rag.theme`; follows the system until used; `index.html` applies the saved theme before the first paint). `renderMarkdown` handles fenced code, nested lists and ordered-list starts; inline code is protected from other markup |
| `src/rag/sources/odoo_module.py`, `code_graph.py`, `code_store.py`, `code_index.py`, `code_search.py` | Code corpus (experimental, branch `feat/code-search`): an Odoo addon read from a git ref (never the working tree) -> units (manifest, model card, method, class header, view, data, doc) -> graph (`data/code/graph.json`: modules, model defs, methods, override chains, adapter groups) and LanceDB table `code_units` beside `chunks` (own `code_index_meta.json`). `rag module add/list/rm/chain/adapters/inspect`, `rag search --corpus code`. BM25 column `fts_text` adds the words of camelCase names |
| `src/rag/traceback.py`, `code_lookup.py`, `code_answer.py` | Failed-job analysis: tolerant traceback parser (frames by module directory name, error codes), exact frame -> unit lookup and error-code lookup, evidence packed into a token budget (innermost frames in full, callers as signature + call site), `/api/code/{modules,search,answer}`, `rag trace`, `rag ask --corpus code`. Code answers are refused for the Claude provider unless `RAG_CODE_EXTERNAL_LLM=1` |
| `src/rag/cli.py` | `rag` typer CLI; `scripts/*.sh` are thin wrappers over it |

## Conventions

- Code, comments, docstrings, commit messages and README are in English. Planning notes and
  results go to `_docs/` (git-ignored), in Russian; talk to the user in Russian.
- `_docs/` files are prefixed by stage: `s<N>-<topic>.<ext>`; `_docs/README.md` indexes the stages and
  holds the rules. A new stage starts with `_docs/s<N>-plan.md` (goal, decisions, steps, open questions);
  write it before the code and keep its "Журнал" section current so another agent can pick the work up.
  Read the current stage's plan first.
- Keep modules small and explicit, with no LangChain/LlamaIndex-style frameworks. Match the existing style.
- Tests must not download models or call servers: use the fake embedder in `tests/test_store_search.py`
  and the `FakeLLM` / `httpx.MockTransport` patterns in `tests/test_answer.py`.
  Run `scripts/test.sh` before finishing a change.
- Retrieval changes (chunking, fusion, models, rerank) must be backed by `scripts/eval.sh` numbers,
  before and after. With ~40 queries, one query equals 0.025 hit@1, so don't over-read small deltas.
- Prompt or LLM changes: compare answers with `scripts/bench.sh` before and after, and read them
  against the sources; the speed numbers alone say nothing about quality.
- Chunking logic changes: bump `chunking.CHUNKER_VERSION`, the next `scripts/index.sh` rebuilds
  (limit changes via `RAG_CHUNK_*` are detected by themselves). Converter changes in
  `sources/confluence.py`: `scripts/sync.sh --reconvert`, then `scripts/index.sh --force`.
- The root page title (`chunking.ROOT_PAGE`, "Home") is dropped from breadcrumbs in `Page`; image
  markers are dropped from chunk text in `chunk_page`. Don't filter them again downstream.
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
- Relevance cut-off works only on rerank scores (calibrated 0..1). Hybrid RRF scores are rank-based
  and vector cosine overlaps too much between relevant and irrelevant pages to threshold. The 0.1 cut-off
  and 10 candidates were chosen from score distributions on the eval set plus out-of-domain queries.
- LM Studio: reasoning models (e.g. Gemma 4) stream "thinking" in `delta.reasoning_content` before
  the answer; `reasoning_effort: "none"` on `/v1/chat/completions` turns it off (`RAG_LLM_REASONING`).
  `parse_sse()` ignores reasoning deltas. Small models (~4B) sometimes pick an off-topic source:
  the prompt tells the model to discard irrelevant sources; rerank or a bigger model helps more.
- Check library APIs against current docs (Context7) rather than memory. These libraries change fast.
