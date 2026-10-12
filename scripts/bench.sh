#!/usr/bin/env bash
# Compare LLMs on eval/bench.yaml: latency, tokens/s, answers side by side (report in data/bench/).
# The LLM comes from RAG_LLM_PROVIDER: LM Studio must be running (models that are not loaded get loaded on the
# first request), or the Claude API with credits (each question is billed).
# Usage: scripts/bench.sh [MODEL ...] [--group novice] [--file eval/bench.yaml] [--no-rerank]
# A custom --out path must be git-ignored: the report quotes the internal docs.
source "$(dirname "$0")/_common.sh"

log "Benchmarking LLM answers (a 27B model takes ~15 s per answer)"
uv run rag bench "$@"
