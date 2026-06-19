#!/usr/bin/env bash
# Thin wrapper -> scripts/devenv.py (Linux/macOS). See BUILDING.md.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(command -v python3 || command -v python)"
exec "$PY" "$DIR/devenv.py" "$@"
