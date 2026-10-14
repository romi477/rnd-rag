# rnd-rag

Local RAG over a Confluence space: hybrid retrieval (BM25 + embeddings), optional answers with
citations from an LLM (local in LM Studio, or the Claude API), and a small web UI.
Search runs on your machine: documents, index and embedding models never leave it. With the Claude API
as the LLM, each answer sends the question and the top doc excerpts to Anthropic.

**Status:** stage 1 (retrieval + search UI) and stage 2 (LLM answers with citations) are done; stage 4 adds
the Claude API as an alternative LLM (`RAG_LLM_PROVIDER=anthropic`).
On the branch `feat/code-search` (experimental, not merged) the same tool also searches the code of Odoo addons
and explains failed background jobs from their traceback: see [Code corpus](#code-corpus-experimental-branch-featcode-search).

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
                                    ▼ (optional, "AI answer")
            top-5 pages as numbered sources ──▶ LLM (LM Studio or Claude API) ──▶ streamed answer with [n] citations
```

The UI works like a browser: every tab keeps its own search (query, filters, chosen LLM, results
and answer) and tabs survive a page reload. To compare models, run the same question in two tabs
with different models. Models marked "(not loaded)" are loaded by LM Studio on first use.
Result cards collapse to their title: the answer and the first card start expanded, the rest
collapsed, and a citation [n] expands the card it points to.
The answer language (English, Spanish, German, Polish, Russian) is chosen next to the model: the prompts stay English, the
model is told to answer in that language and to keep menu paths, settings, error messages and code as in the sources.
The API takes `lang` (`/api/answer`) or `language` (`/api/code/answer`); `/api/languages` lists them. The documentation is
English: questions in the other languages are searched by the multilingual embedder and reranker, not translated.

Only the button runs a query: it reads "Find", or "Ask" when "AI answer" is on, and "Stop"
next to it cancels a running request (what has streamed so far stays). Changing settings afterwards
marks the results as outdated until the next run. The eraser icon clears the tab.
Keyboard: `/` focuses the question box (not while typing in a field), `F1` / `F2` / `F3` jump to Search / Browse / Code and
`Ctrl+Alt+Left` / `Ctrl+Alt+Right` step through them. The model and the answer language are read-only while
"AI answer" is off.
The Search / Browse switch in the header belongs to the tab: Browse lists every indexed article
(`/api/pages`) grouped by product and section, newest first inside a group (by the date the page
was created in Confluence, which the list shows; the tooltip adds the last update), narrowed by the product filter and by the words typed
in the title filter; each view keeps its own text, and `?view=browse&filter=...` opens it directly.
The groups open folded: click a group's heading (it shows how many articles it holds) to open it, or use the small button beside
the article count to open or fold them all; the open groups are remembered per tab, and while a title filter is typed the groups it
matched stay open.
The eye button at the start of a row opens a preview: the article is fetched from the server when pressed (`/api/pages/{id}`)
and shown in a window over the blurred page, with its headings, lists, tables, quotes, links and pictures (a picture opens full size on a
click). Pictures come from `scripts/sync.sh --attachments`, which downloads the images each page shows (not its other attachments;
`/api/pages/{id}/images/{name}` serves them, PNG / JPEG / GIF / WebP only, judged by the file's content); one that was not downloaded, or is a URL, stays a label. Esc, the cross or a click outside closes it; "Open in Confluence" goes to the original.
The "AI translate" checkbox (Browse only, with a language select: Spanish, German, Polish, Russian; the articles are English) turns
the eye button into a translate button: the article is translated by the Claude API (Haiku, `RAG_TRANSLATE_MODEL`, whatever
`RAG_LLM_PROVIDER` is) and the translation streams into the same window, marked "AI translation". Menu paths, settings, code and
image markers stay as they are. The model selects (the answer model in Search and Code, the translation model in Browse) show only
while the bug (Debug) button is on, and so does what an LLM call cost: under an answer (Search, Code) and in the footer of the translation
window, the tokens in and out as the server reported them, the time and the time to the first token; the default of the answers is the server's model (Sonnet 5.5 with the Claude API), of the translation Haiku 5.5. The article text goes to Anthropic (needs `ANTHROPIC_API_KEY`; without it the checkbox is disabled),
each translation counts toward `RAG_ANSWERS_PER_HOUR`, and a finished translation is kept by the page until it is reloaded.
Cards are numbered by rank; with an LLM answer, the cards used as sources keep a filled badge with
their source number. The bug icon turns on debug mode: under each card and fragment it shows how the
fragment was scored (`vector #rank cosine · bm25 #rank score · rrf · retrieval #rank · rerank`); under an
LLM answer, the tokens it took (prompt, answer, reasoning if counted) and the time to the first token.
The half-filled circle switches between the light and dark theme; until it is used, the page follows
the system setting. The three icons sit in the title row, above the Find button.

Search modes. The UI always searches Hybrid + Rerank; the others are there for the API, the CLI
(`--mode`, `--no-rerank`) and `scripts/eval.sh`:

| Mode | Finds | Good at | Weak at |
|---|---|---|---|
| Keyword (`fts`) | chunks containing the query words (BM25) | error codes, module names, settings | synonyms, paraphrases |
| Semantic (`vector`) | chunks with similar meaning (embeddings) | paraphrased questions | rare exact tokens |
| Hybrid (default, the UI) | rank fusion of both lists | robust default | — |
| + Rerank (always on in the UI) | re-scores the top 10 chunks with a cross-encoder that reads query and chunk together; pages below the cut-off are collapsed and not sent to the LLM | precise ordering, meaningful 0..1 scores, drops junk | +~0.5 s per query |

## Requirements

- macOS (Apple Silicon uses the GPU via MPS) or Linux; Python 3.13 is installed by `uv`
- [uv](https://docs.astral.sh/uv/)
- ~4 GB disk for models (downloaded on first use to the Hugging Face cache)
- Confluence Cloud account with an [API token](https://id.atlassian.com/manage-profile/security/api-tokens)
- For answers (optional), one of:
  - [LM Studio](https://lmstudio.ai) with a chat model loaded and the local server started
    (Developer → Start Server, port 1234), or any other OpenAI-compatible server;
  - a Claude API key with credits (`RAG_LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY` in `.env`)

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
| `sync.sh [--force] [--attachments]` | download new/changed pages into `raw-data/` (`--attachments`: also the images they show, for the Browse preview); `--reconvert` rebuilds the markdown offline |
| `index.sh [--force] [--model M]` | chunk + embed changed pages into `data/lancedb/` |
| `serve.sh [port] [--offline]` | web UI + JSON API (`RAG_HOST` / `RAG_PORT` env vars also work); `--offline` skips the Hugging Face update check |
| `search.sh "query" [--text]` | search from the terminal |
| `ask.sh "question" [--no-rerank] [--lang ru]` | search + streamed LLM answer with sources in the terminal; `--lang` en, es, de, pl, ru (default `RAG_ANSWER_LANGUAGE`) |
| `eval.sh [--misses]` | retrieval quality on `eval/queries.yaml` |
| `bench.sh [MODEL ...] [--group G]` | compare LLMs on `eval/bench.yaml`: latency, tokens/s, answers side by side |
| `module.sh add\|list\|rm\|chain\|adapters\|inspect ...` | code corpus (experimental): Odoo addons read from a git ref into the code graph and the code search index; see "Code corpus" |
| `docs.sh add REPO --ref REF \| sync \| list \| rm NAME` | code corpus (experimental): the Markdown `docs/` of a repository, read from a git ref and indexed next to the code; `sync` re-reads every registered source and skips the unchanged ones |
| `trace.sh FILE [--client-version V] [--text]` | map a failed job's traceback to the indexed code: functions, overrides, error codes |
| `eval-traces.sh [--misses] [--no-search]` | does the evidence for each traceback of `eval/traces.yaml` contain the cause? (no LLM) |
| `bench-traces.sh MODEL [MODEL ...] [--group G]` | compare LLMs on the failure cases, report in `data/bench/` |
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
| `RAG_LLM_PROVIDER` | `openai`: LM Studio or another OpenAI-compatible server; `anthropic`: the Claude API |
| `RAG_LLM_BASE_URL` | `http://127.0.0.1:1234/v1` (LM Studio; any OpenAI-compatible server works) |
| `ANTHROPIC_API_KEY` | the Claude API key, with `RAG_LLM_PROVIDER=anthropic` |
| `RAG_CLAUDE_MODELS` | `claude-sonnet-5-5,claude-haiku-5-5`: the Claude models offered in the UI and accepted by `/api/answer` (empty: every model the key can use) |
| `RAG_LLM_MODEL` | empty: the model loaded in LM Studio; with the Claude API, `claude-sonnet-5-5` |
| `RAG_LLM_REASONING` | `none`: as little "thinking" as the model allows, for faster answers (LM Studio: sent as `reasoning_effort`; Claude Sonnet 5.5: `thinking: between_tools`, other Claude models: effort `low`); Claude also takes `low` / `medium` / `high` as the effort level; empty: the server default |
| `RAG_LLM_TEMPERATURE` / `RAG_LLM_MAX_TOKENS` | `0.2` / `1000` (temperature is not sent to Claude: current models take only the default) |
| `RAG_TRANSLATE_MODEL` / `RAG_TRANSLATE_MAX_TOKENS` | `claude-haiku-5-5` / `6000` (x1.3 es, x1.4 de, x1.7 pl, x2 ru): the Claude model and the output limit of the Browse "AI translate"; the key is `ANTHROPIC_API_KEY` whatever the answer provider is |
| `RAG_ANSWERS_PER_HOUR` | `200`: at most this many `/api/answer` and translation requests per hour, then 429 (0: no limit). Claude answers are billed: set `RAG_API_TOKEN` when others can reach the server, and also set a monthly spend limit in the Anthropic Console |
| `RAG_LLM_CONTEXT_PAGES` / `RAG_LLM_CONTEXT_TOKENS` | `5` / `4000`: how much of the search result goes into the prompt |
| `RAG_API_TOKEN` | empty: no login. A token that the UI and every endpoint then require, as the password of HTTP Basic auth (the user name is ignored): the browser asks once, `curl -u :TOKEN`, `Authorization: Basic base64(":TOKEN")` |
| `RAG_ALLOWED_HOSTS` | `127.0.0.1,localhost`: host names the server answers to (see below) |

The server binds to `127.0.0.1` and answers only to requests addressed to `127.0.0.1` or `localhost`,
so a web page elsewhere can't reach it by DNS rebinding. It has no login unless `RAG_API_TOKEN` is set: binding to another
interface (`RAG_HOST=0.0.0.0`) shares the internal docs with the network, so set the token and put the machine's
name into `RAG_ALLOWED_HOSTS`. Basic auth sends the token in every request: use it over HTTPS (a reverse proxy) beyond a trusted network. There are no `/docs` pages (they load scripts from a CDN).

Each embedding model gets its own index directory, so models can be compared side by side:
`scripts/index.sh --model BAAI/bge-base-en-v1.5`, then `scripts/eval.sh --model BAAI/bge-base-en-v1.5`.

Models are cached in `~/.cache/huggingface` (override with `HF_HOME`). After the first download,
`HF_HUB_OFFLINE=1` (or `scripts/serve.sh --offline`) skips the update check on startup.

## API

JSON over HTTP, served by `scripts/serve.sh` next to the UI (login only with `RAG_API_TOKEN`: HTTP Basic, the token is the password; the server answers only to the
host names in `RAG_ALLOWED_HOSTS`, and sends no CORS headers, so other web pages can't call it from a browser).

| Endpoint | Returns |
|---|---|
| `GET /api/search` | pages ranked for a query, with their best fragments |
| `GET /api/answer` | the same search, then an LLM answer with [n] citations, streamed as NDJSON |
| `GET /api/llm` | whether answer generation is available, the default and all chat models |
| `GET /api/products` | the values for the `product` filter |
| `GET /api/pages` | every indexed page with its Confluence link |
| `GET /api/pages/{id}` | one article as the sync stored it (Markdown without the title, plus title, breadcrumbs, `updated`, `created`, link); 404 when the id is not synced |

`GET /api/search?q=...&k=10&mode=hybrid|vector|fts&product=...&rerank=false`

Quotes and backslashes in `q` are ignored by the BM25 side (no phrase queries); if the BM25 index is
missing (an interrupted rebuild), hybrid search falls back to vectors only.

```json
{
  "query": "...", "took_ms": 54,
  "results": [{
    "page_id": "...", "title": "...", "url": "...", "product": "...", "breadcrumbs": ["..."],
    "updated": "...", "labels": [], "score": 0.93, "low_relevance": false,
    "chunks": [{"chunk_id": "...", "section_path": ["..."], "text": "...", "score": 0.93,
                "signals": {"vector": {"rank": 1, "score": 0.71}, "fts": {"rank": 3, "score": 9.4},
                            "rrf": 0.021, "retrieval_rank": 1, "rerank": 0.93}}]
  }]
}
```

`GET /api/answer` takes the same parameters (but `rerank` is on by default, so that unrelated pages
stay out of the prompt) plus optional `model` (an LLM id known to the LLM server; default: `RAG_LLM_MODEL`,
the model loaded in LM Studio, or `claude-sonnet-5-5`; an unknown one is a 400) and streams NDJSON (one JSON event per line).
The search runs once: its results come first, then the answer.

```
{"type": "results", "query": "...", "took_ms": 54, "results": [...]}   same as /api/search
{"type": "sources", "sources": [{"n": 1, "page_id": "...", "title": "...", "url": "..."}]}
{"type": "delta", "text": "The error means ... [1]"}                  repeated while streaming
{"type": "done", "model": "...", "took_ms": 1771, "truncated": false, "first_token_ms": 640,
 "usage": {"prompt_tokens": 1350, "completion_tokens": 210}}         or {"type": "error", "message": "..."}
```

`truncated` is true when the answer hit `RAG_LLM_MAX_TOKENS`. `usage` (token counts as the LLM server reports
them; `reasoning_tokens` when it counts thinking separately) and `first_token_ms` (time to the first answer text) are there when known. A stream that the LLM server closes
early, or an answer with no text (a reasoning model that spent the budget thinking), is an `error`.

`GET /api/llm` reports whether answer generation is available, the default model and all chat
models with their load state (`{"available", "model", "models": [{"id", "loaded"}], "error"}`).
`GET /api/products` lists the values for the `product` filter.
`GET /api/pages` lists every indexed page with its Confluence link, sorted by product and path
(pages without text, e.g. section pages that only list their children, are not indexed):

```json
{"count": 242, "pages": [{"page_id": "...", "title": "...", "url": "https://...", "product": "...",
                          "breadcrumbs": ["..."], "updated": "...", "created": "...", "labels": ["..."]}]}
```

`GET /api/pages/{id}` returns `{"page_id", "title", "url", "breadcrumbs", "product", "updated", "created", "labels", "markdown"}` for the Browse
preview. It reads the synced copy in `raw-data/` (the server holds no Confluence token), so `updated` is the page's last change at
the time of the last sync; the id is digits only.

`GET /api/pages/{id}/translate?lang=ru` streams the article translated by the Claude API (NDJSON, like `/api/answer`): an optional
`{"type": "title", "text"}` (the translated title), `{"type": "delta", "text"}` events with the Markdown, then `{"type": "done", "model",
"took_ms", "truncated"}` or `{"type": "error", "message"}`. `lang` is required: `es`, `de`, `pl` or `ru` (`en` is a 400: the articles are
English). The optional `model` must be one of `translate_models` of `GET /api/llm`, which also has `translate: true` when the Claude key is set
and `translate_model` (the default).

## Code corpus (experimental, branch `feat/code-search`)

A second index next to the Confluence one: units of Odoo addons (manifest, model cards with field help, methods,
views, data records, docs), plus a graph of dependencies and overrides. Nothing leaves the machine, and the code lives
only in `data/` (git-ignored).

```bash
scripts/module.sh add /path/to/addon --ref 19.0      # read from a git ref (--working-tree: the files as checked out); parse, graph, embed
scripts/module.sh list                               # modules in the graph, dependencies first
scripts/module.sh chain sale.order action_confirm    # every definition of a method across modules, base first
scripts/module.sh adapters export_inventory          # abstract adapter method and each connector's implementation
scripts/search.sh "ensureSettings" --corpus code     # hybrid search over code units (--module M, --mode fts|vector)
scripts/trace.sh traceback.txt --client-version X    # frames -> functions, overrides, error codes (exact lookups)
scripts/eval-traces.sh --misses                      # does the evidence contain the function the failure comes from? (eval/traces.yaml)
scripts/bench-traces.sh MODEL [MODEL...] --group real  # same evidence and prompt to several local LLMs, report in data/bench/
```

The web UI has one three-position switch per tab (Search, Browse, Code): Code takes a described problem and/or a traceback (optional: "Add traceback" next to the question opens a small window for it, which collapses to its title bar; a traceback pasted into the question box moves there), shows the frames mapped to code and, with
"AI explanation", an explanation citing the code as `[n]`. Source code is not sent to the Claude API unless
`RAG_CODE_EXTERNAL_LLM=1`. `eval/traces.yaml` is git-ignored; start from `eval/traces.example.yaml`.

A question that names a function (quoted, snake_case, camelCase or `model.method`) gets its definition by exact name, so "how does `_import_one_product` work?" quotes the method itself and the answer walks through it step by step.

The graph and the chains come from static analysis: a dynamic `_inherit` or a call by a string name is not seen.
The search index and the graph are written one after the other (`--no-index` writes only the graph). The index records the
version and git ref of each module; `module.sh list`, `/api/code/modules` and the notes of a code answer say when they
differ for a module. After a change of the index format (`CODE_INDEX_VERSION`), `module.sh add ... --reset-index` for the
first module and `add` for the rest rebuild it. The code searcher of a running server opens the index on first use, so
the first `module add` needs no restart. `module.sh rm NAME` refuses while another module depends on it (`--force`).

Repository documentation joins the same index: `scripts/docs.sh add /path/to/repo --ref 19.0` registers the repository's `docs/`
(`--dir` names another directory; a path inside a repository is taken as the docs directory itself, and a `docs` that is a
symbolic link is followed to its target). Every Markdown section becomes a `guide` unit under the name `docs:<repo>`, cited
as `[n]` next to the code with the role "documentation". A code answer takes the documentation sections that mention the
traceback's error code (`E…`) and the two best search hits, whichever module is chosen. The docs are read from the git ref, never from
the checked-out files; `scripts/docs.sh sync` after the docs grew re-reads all sources (a source whose docs directory has the same git
tree is skipped, `--force` reads it anyway). The documents are internal like the code: they stay local and follow
`RAG_CODE_EXTERNAL_LLM`.

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
- [x] The Claude API as an alternative LLM (stage 4)
- [x] Code search and traceback analysis for Odoo addons (stage 3, experimental, branch `feat/code-search`)
- [x] Relevance cut-off (rerank score threshold)
- [ ] Answer quality evaluation
- [ ] MCP server exposing search to coding agents
