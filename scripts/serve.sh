#!/usr/bin/env bash
# Start the web search UI + JSON API. Stop with Ctrl+C.
# Usage: scripts/serve.sh [port]        env: HOST (default 127.0.0.1), PORT (default 8000)
source "$(dirname "$0")/_common.sh"

PORT="${1:-${PORT:-8000}}"
HOST="${HOST:-127.0.0.1}"

has_index || die "no search index yet: run scripts/update.sh first"
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  die "port $PORT is busy (server already running?). Open http://$HOST:$PORT or pass another port: scripts/serve.sh 8001"
fi

log "Starting server: http://$HOST:$PORT  (model loads ~10 s; Ctrl+C to stop)"
exec uv run rag serve --host "$HOST" --port "$PORT"
