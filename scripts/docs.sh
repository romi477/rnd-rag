#!/usr/bin/env bash
# Repository documentation (docs/ directories, Markdown) as part of the code corpus (experimental, branch feat/code-search).
# Usage: scripts/docs.sh add /path/to/repo --ref 19.0 [--dir docs] [--name NAME]   register and index the docs at a git ref
#        scripts/docs.sh sync [NAME...] [--force]    read the registered sources again (unchanged ones are skipped)
#        scripts/docs.sh list | rm NAME
source "$(dirname "$0")/_common.sh"

[ $# -gt 0 ] || die 'usage: scripts/docs.sh add|sync|list|rm ...'
uv run rag docs "$@"
