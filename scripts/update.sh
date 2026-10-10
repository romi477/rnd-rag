#!/usr/bin/env bash
# Bring everything up to date: Confluence -> raw-data/ -> search index.
# A running server picks up the new index without restart.
# Usage: scripts/update.sh [--force]   (--force re-downloads and re-indexes everything)
source "$(dirname "$0")/_common.sh"

log "Step 1/2: sync"
"$ROOT/scripts/sync.sh" "$@"
log "Step 2/2: index"
"$ROOT/scripts/index.sh" "$@"
log "Done"
