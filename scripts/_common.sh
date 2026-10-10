# Shared by scripts/*.sh: strict mode, project root as cwd, uv check, helpers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -t 1 ]; then BOLD=$'\033[1m' BLUE=$'\033[1;34m' RED=$'\033[1;31m' RESET=$'\033[0m'
else BOLD="" BLUE="" RED="" RESET=""; fi
log() { printf '%s==>%s %s%s%s\n' "$BLUE" "$RESET" "$BOLD" "$*" "$RESET"; }
die() { printf '%serror:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || die "uv not found, install it: https://docs.astral.sh/uv/"

# True when a search index has been built (any model).
has_index() { ls data/lancedb/*/index_meta.json >/dev/null 2>&1; }
