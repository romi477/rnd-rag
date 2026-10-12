#!/usr/bin/env bash
# Ask a question: search + answer from the LLM (RAG_LLM_PROVIDER: LM Studio running with a model loaded, or the Claude API).
# Usage: scripts/ask.sh "question" [--no-rerank] [--product "Direct Print"] [--llm-model ID]
source "$(dirname "$0")/_common.sh"

[ $# -gt 0 ] || die 'usage: scripts/ask.sh "your question" [--product "Direct Print"]'
uv run rag ask "$@"
