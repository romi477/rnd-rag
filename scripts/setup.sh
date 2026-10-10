#!/usr/bin/env bash
# First-time setup: install Python deps and create .env from the template.
# Usage: scripts/setup.sh
source "$(dirname "$0")/_common.sh"

log "Installing dependencies (uv sync)"
uv sync

if [ ! -f .env ]; then
  cp .env.example .env
  log "Created .env from .env.example: set CONFLUENCE_EMAIL there"
else
  log ".env already exists"
fi

cat <<'TXT'

Next steps:
  scripts/update.sh    download Confluence pages and build the search index
  scripts/serve.sh     start the web UI
TXT
