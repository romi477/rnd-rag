#!/usr/bin/env bash
# Ask a question: search + answer from the local LLM (LM Studio must be running with a model loaded).
# Usage: scripts/ask.sh "question" [--rerank] [--product "Direct Print"] [--llm-model ID]
source "$(dirname "$0")/_common.sh"

[ $# -gt 0 ] || die 'usage: scripts/ask.sh "your question" [--rerank]'
has_index || die "no search index yet: run scripts/update.sh first"
uv run rag ask "$@"
