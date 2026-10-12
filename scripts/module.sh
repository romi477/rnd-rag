#!/usr/bin/env bash
# Odoo addons as a code corpus (experimental, branch feat/code-search). Subcommands: add, list, rm, chain, adapters, inspect.
# Usage: scripts/module.sh add /path/to/addon --ref 19.0 [--origin core|connector|customer] [--no-index]
#        scripts/module.sh list | rm NAME | chain MODEL METHOD | adapters METHOD | inspect /path/to/addon [--json out.json]
source "$(dirname "$0")/_common.sh"

[ $# -gt 0 ] || die 'usage: scripts/module.sh add|list|rm|chain|adapters|inspect ...'
uv run rag module "$@"
