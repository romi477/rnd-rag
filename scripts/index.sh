#!/usr/bin/env bash
# Build/refresh the search index from raw-data/ (only changed pages are re-embedded).
# Usage: scripts/index.sh [--force] [--model BAAI/bge-base-en-v1.5]
source "$(dirname "$0")/_common.sh"

log "Indexing pages"
uv run rag index "$@"
