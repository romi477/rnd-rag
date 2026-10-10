#!/usr/bin/env bash
# Search from the terminal.
# Usage: scripts/search.sh "query" [--text] [-k 5] [--mode hybrid|vector|fts] [--product "Direct Print"] [--rerank]
source "$(dirname "$0")/_common.sh"

[ $# -gt 0 ] || die 'usage: scripts/search.sh "your question" [--text]'
has_index || die "no search index yet: run scripts/update.sh first"
uv run rag search "$@"
