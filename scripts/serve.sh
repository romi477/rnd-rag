#!/usr/bin/env bash
# Start the web search UI + JSON API. Stop with Ctrl+C.
# Usage: scripts/serve.sh [port]        env: RAG_HOST (default 127.0.0.1), RAG_PORT (default 8000)
# Binding beyond localhost exposes the internal docs: also set RAG_ALLOWED_HOSTS to the machine's name.
source "$(dirname "$0")/_common.sh"

PORT="${1:-${RAG_PORT:-8000}}"
HOST="${RAG_HOST:-127.0.0.1}"

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  die "port $PORT is busy (server already running?). Open http://$HOST:$PORT or pass another port: scripts/serve.sh 8001"
fi

log "Starting server: http://$HOST:$PORT  (models load ~10 s in the background; Ctrl+C to stop)"
exec uv run rag serve --host "$HOST" --port "$PORT"
