#!/usr/bin/env bash
# Map a failed job's traceback to the indexed code: functions, overrides, error codes (needs `scripts/module.sh add` first).
# Usage: scripts/trace.sh traceback.txt [--client-version 19.0.2.2.0] [--text]     (or pipe the traceback in)
source "$(dirname "$0")/_common.sh"

uv run rag trace "$@"
