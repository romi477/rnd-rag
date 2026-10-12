#!/usr/bin/env bash
# Compare LLMs on the failed-job cases of eval/traces.yaml; the report with the answers side by side is in data/bench/.
# Usage: scripts/bench-traces.sh MODEL [MODEL...] [--group real] [--limit 5]
# Code goes to the LLM: use a local one (RAG_LLM_PROVIDER=openai); the Claude provider is refused unless RAG_CODE_EXTERNAL_LLM=1.
source "$(dirname "$0")/_common.sh"

uv run rag bench-traces "$@"
