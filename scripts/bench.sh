#!/usr/bin/env bash
# Compare LLMs on eval/bench.yaml: latency, tokens/s, answers side by side (report in data/bench/).
# LM Studio must be running; models that are not loaded get loaded on the first request.
# Usage: scripts/bench.sh [MODEL ...] [--file eval/bench.yaml] [--out report.md] [--no-rerank]
source "$(dirname "$0")/_common.sh"

has_index || die "no search index yet: run scripts/update.sh first"
[ -f "${RAG_BENCH_FILE:-eval/bench.yaml}" ] || [[ " $* " == *" --file "* ]] \
  || die "no eval/bench.yaml: cp eval/bench.example.yaml eval/bench.yaml and fill it"
log "Benchmarking LLM answers (a 27B model takes ~15 s per answer)"
uv run rag bench "$@"
