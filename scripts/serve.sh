#!/usr/bin/env bash
# Start the web search UI + JSON API. Stop with Ctrl+C.
# Usage: scripts/serve.sh [port] [--offline]   env: RAG_HOST (default 127.0.0.1), RAG_PORT (default 8000)
#   --offline   don't ask Hugging Face for model updates on startup (models must be downloaded already)
# Binding beyond localhost exposes the internal docs: also set RAG_ALLOWED_HOSTS to the machine's name.
source "$(dirname "$0")/_common.sh"

PORT="${RAG_PORT:-8000}"
HOST="${RAG_HOST:-127.0.0.1}"
for arg in "$@"; do
  case "$arg" in
    --offline) export HF_HUB_OFFLINE=1 ;;
    -*) die "unknown option $arg. Usage: scripts/serve.sh [port] [--offline]" ;;
    *) PORT="$arg" ;;
  esac
done

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  die "port $PORT is busy (server already running?). Open http://$HOST:$PORT or pass another port: scripts/serve.sh 8001"
fi

log "Starting server: http://$HOST:$PORT  (models load ~10 s in the background; Ctrl+C to stop)"
[ "${HF_HUB_OFFLINE:-}" = 1 ] && log "Offline: models load from the local Hugging Face cache only"
exec uv run rag serve --host "$HOST" --port "$PORT"
