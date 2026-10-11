# Shared by scripts/*.sh: strict mode, project root as cwd, uv check, helpers.
# Checks for the index, synced pages and question files live in the `rag` CLI, which knows the
# configured paths (RAG_DATA_DIR, RAG_RAW_DIR... from env or .env).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -t 1 ]; then BOLD=$'\033[1m' BLUE=$'\033[1;34m' RED=$'\033[1;31m' RESET=$'\033[0m'
else BOLD="" BLUE="" RED="" RESET=""; fi
log() { printf '%s==>%s %s%s%s\n' "$BLUE" "$RESET" "$BOLD" "$*" "$RESET"; }
die() { printf '%serror:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || die "uv not found, install it: https://docs.astral.sh/uv/"
