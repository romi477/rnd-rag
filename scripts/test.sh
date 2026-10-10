#!/usr/bin/env bash
# Run unit tests. Extra args go to pytest.
# Usage: scripts/test.sh [-k chunking]
source "$(dirname "$0")/_common.sh"

uv run pytest -q "$@"
