#!/usr/bin/env bash
# Measure search quality on eval/queries.yaml (hit@k, MRR).
# Usage: scripts/eval.sh [--misses] [--no-rerank] [--fts-weights 0,0.3,1] [--model ...]
source "$(dirname "$0")/_common.sh"

log "Evaluating (with rerank this takes ~1-2 min)"
uv run rag eval "$@"
