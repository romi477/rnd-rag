#!/usr/bin/env bash
# Build/refresh the search index from raw-data/ (only changed pages are re-embedded).
# Usage: scripts/index.sh [--force] [--model BAAI/bge-base-en-v1.5]
source "$(dirname "$0")/_common.sh"

ls raw-data/confluence/*/pages/*.md >/dev/null 2>&1 || die "no pages in raw-data/: run scripts/sync.sh first"

log "Indexing pages"
uv run rag index "$@"
