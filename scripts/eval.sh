#!/usr/bin/env bash
# Measure search quality on eval/queries.yaml (hit@k, MRR).
# Usage: scripts/eval.sh [--misses] [--no-rerank] [--fts-weights 0,0.3,1] [--model ...]
source "$(dirname "$0")/_common.sh"

has_index || die "no search index yet: run scripts/update.sh first"
[ -f "${RAG_EVAL_FILE:-eval/queries.yaml}" ] || die "no eval/queries.yaml: cp eval/queries.example.yaml eval/queries.yaml and fill it"
log "Evaluating (with rerank this takes ~1-2 min)"
uv run rag eval "$@"
