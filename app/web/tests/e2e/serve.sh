#!/usr/bin/env bash
# Launch the native (C++) server for the Playwright e2e suite: build the SPA if
# needed, seed a demo session, then serve the JSON API + SPA single-origin from
# build/whisperx_server. The native server is the real backend now (the Flask
# app is reference-only and lacks newer routes like the turn /split), so the
# suite drives it directly. It reads the same sessions.db + transcript.json the
# Python seeder writes, so seed.py is unchanged.
#
# Env:
#   PYTHON  python able to import app.store for the one-shot seed (stdlib only:
#           json/os/shutil/sqlite3). Default: python3. Not used for serving.
#   PORT    server port (default 5099; must match playwright.config.ts).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"   # app/web/tests/e2e -> repo root
PY="${PYTHON:-python3}"
BIN="$REPO/build/whisperx_server"

# Use a caller-supplied data dir as-is; otherwise mint a temp one we own and
# clean up on exit (only the one we created — never a dir passed in via env).
OWN_DATA_DIR=""
if [ -z "${WHISPERX_DATA_DIR:-}" ]; then
  WHISPERX_DATA_DIR="$(mktemp -d)"
  OWN_DATA_DIR="$WHISPERX_DATA_DIR"
fi
export WHISPERX_DATA_DIR
PORT="${PORT:-5099}"

SRV=""
cleanup() {
  [ -n "$SRV" ] && kill "$SRV" 2>/dev/null || true
  [ -n "$OWN_DATA_DIR" ] && rm -rf "$OWN_DATA_DIR"
}
trap cleanup EXIT INT TERM

if [ ! -f "$REPO/app/static/spa/index.html" ]; then
  echo "building SPA…" >&2
  (cd "$REPO/app/web" && bun run build)
fi

if [ ! -x "$BIN" ]; then
  echo "building whisperx_server…" >&2
  cmake --build "$REPO/build" --target whisperx_server
fi

"$PY" "$HERE/seed.py"
cd "$REPO"
# WHISPERX_STATIC_DIR points the SPA catch-all at app/static; the model warm is a
# detached, non-fatal thread, so /healthz is ready immediately even offline.
# Run in the background + wait (not exec) so the EXIT trap can remove the temp
# data dir when Playwright signals teardown.
env \
  WHISPERX_HOST=127.0.0.1 \
  WHISPERX_PORT="$PORT" \
  WHISPERX_DATA_DIR="$WHISPERX_DATA_DIR" \
  WHISPERX_STATIC_DIR="$REPO/app/static" \
  WHISPERX_LOG_LEVEL=warn \
  "$BIN" &
SRV=$!
wait "$SRV"
