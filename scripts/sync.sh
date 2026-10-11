#!/usr/bin/env bash
# Download new/changed Confluence pages into raw-data/ (incremental).
# Usage: scripts/sync.sh [--force] [--attachments]
#        scripts/sync.sh --reconvert   offline: rebuild the markdown from the stored HTML (after a converter fix)
source "$(dirname "$0")/_common.sh"

case " $* " in *" --reconvert "*) exec uv run rag sync "$@" ;; esac

[ -f .env ] || die ".env missing: run scripts/setup.sh and set CONFLUENCE_EMAIL"

log "Syncing Confluence pages"
if [ -n "${CONFLUENCE_API_TOKEN:-}${JIRA_TOKEN_VENT:-}" ] || grep -qE '^CONFLUENCE_API_TOKEN=["'\'']?[^"'\''[:space:]#]' .env; then
  uv run rag sync "$@"
else
  # The token ($JIRA_TOKEN_VENT) is defined in the interactive zsh profile only.
  zsh -ic 'cd "$1" && shift && uv run rag sync "$@"' zsh "$ROOT" "$@"
fi
