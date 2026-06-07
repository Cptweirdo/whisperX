#!/usr/bin/env bash
# Launch a test Flask server for the Playwright e2e suite: build the SPA if
# needed, seed a demo session, then serve the JSON API + SPA single-origin.
#
# Env:
#   PYTHON  python with the web deps (flask, keyring, …). Default: python3.
#           In a dev checkout: PYTHON=.venv-web/bin/python (or your whisperx env).
#   PORT    server port (default 5099; must match playwright.config.ts).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"   # app/web/tests/e2e -> repo root
PY="${PYTHON:-python3}"

export WHISPERX_NO_WARM=1
export WHISPERX_DATA_DIR="${WHISPERX_DATA_DIR:-$(mktemp -d)}"
export HOST=127.0.0.1
export PORT="${PORT:-5099}"

if [ ! -f "$REPO/app/static/spa/index.html" ]; then
  echo "building SPA…" >&2
  (cd "$REPO/app/web" && bun run build)
fi

"$PY" "$HERE/seed.py"
cd "$REPO"
exec "$PY" -m app.server
