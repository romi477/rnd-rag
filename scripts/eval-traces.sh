#!/usr/bin/env bash
# Failed-job analysis eval: is the function the failure comes from in the evidence? (no LLM)
# Usage: scripts/eval-traces.sh [--misses] [--no-search] [--file eval/traces.yaml]
source "$(dirname "$0")/_common.sh"

uv run rag eval-traces "$@"
