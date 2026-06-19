#!/usr/bin/env bash
# Apple-Silicon lever 1 bench (SPEEDUP_FINDINGS.md "quant ggml + flash-attn"):
# sweep WHISPERX_GGML_QUANT × WHISPERX_WHISPERCPP_FLASH_ATTN on the whisper.cpp/Metal
# Stage-1 backend and record the transcribe-stage RTF + transcript per config.
#
# whisper.cpp is server-only (the pyd binds WhisperSherpa, not WhisperCpp) and both
# knobs are read at engine-build time, so each config is a fresh server process with a
# fresh WHISPERX_DATA_DIR (isolates the SQLite DB + log + sessions; the ggml/align/
# diarize ASSET cache stays shared under ~/.cache/whisperx-sherpa). The sample is
# uploaded TWICE per config — the first warms Metal pipeline compile + model load, the
# second is the clean number kept.
#
# Output per config under bench-out/: <label>.log (server log) + <label>.json
# (transcript). Analyse with scripts/bench_whispercpp_wer.py.
#
# Usage:
#   bash scripts/bench_whispercpp_metal.sh            # full 6-config matrix
#   bash scripts/bench_whispercpp_metal.sh --smoke    # 2 configs (loop validation)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="$ROOT/build/whisperx_server"
SAMPLE="$ROOT/samples/russian_2_speaker_trim.m4a"
OUT="$ROOT/bench-out"
HOST=127.0.0.1
PORT="${WHISPERX_BENCH_PORT:-8123}"
BASE="http://$HOST:$PORT"

# keg-only brew libs (curl/libarchive) — harmless if already runtime-linked.
export PKG_CONFIG_PATH="/opt/homebrew/opt/curl/lib/pkgconfig:/opt/homebrew/opt/libarchive/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-}"

[[ -x "$SERVER" ]] || { echo "missing server binary: $SERVER (cmake --build --preset server-macos)"; exit 1; }
[[ -f "$SAMPLE" ]] || { echo "missing sample: $SAMPLE"; exit 1; }
mkdir -p "$OUT"

# label  quant  flash
CONFIGS=(
  "fp16-flash0::0"
  "fp16-flash1::1"
  "q8_0-flash0:q8_0:0"
  "q8_0-flash1:q8_0:1"
  "q5_0-flash0:q5_0:0"
  "q5_0-flash1:q5_0:1"
)
if [[ "${1:-}" == "--smoke" ]]; then
  CONFIGS=("fp16-flash0::0" "q8_0-flash1:q8_0:1")
fi

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

wait_healthz() {  # $1 timeout_s
  local t=0
  until curl -fsS "$BASE/healthz" >/dev/null 2>&1; do
    sleep 1; t=$((t+1)); [[ $t -ge $1 ]] && return 1
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "  server died during boot"; return 1; }
  done
}

upload() {  # echoes session id (retries 503 loading_models)
  local resp id t=0
  while :; do
    resp="$(curl -s -H 'Expect:' \
      -F "audio=@$SAMPLE" -F "model=large-v3-turbo" -F "language=ru" \
      -F "min_speakers=2" -F "max_speakers=2" "$BASE/api/sessions")"
    id="$(printf '%s' "$resp" | python3 -c 'import sys,json;
try: print(json.load(sys.stdin).get("id",""))
except Exception: print("")' 2>/dev/null)"
    [[ -n "$id" ]] && { echo "$id"; return 0; }
    t=$((t+1)); [[ $t -ge 30 ]] && { echo "  upload failed: $resp" >&2; return 1; }
    sleep 1
  done
}

wait_done() {  # $1 session id, $2 timeout_s
  local st t=0
  while :; do
    st="$(curl -s "$BASE/api/sessions/$1" | python3 -c 'import sys,json;
try: print(json.load(sys.stdin).get("status",""))
except Exception: print("")' 2>/dev/null)"
    [[ "$st" == "done" ]] && return 0
    [[ "$st" == "error" ]] && { echo "  job errored"; return 1; }
    t=$((t+2)); [[ $t -ge $2 ]] && { echo "  job timeout (status=$st)"; return 1; }
    sleep 2
  done
}

for entry in "${CONFIGS[@]}"; do
  label="${entry%%:*}"; rest="${entry#*:}"; quant="${rest%%:*}"; flash="${rest##*:}"
  datadir="$OUT/data-$label"
  rm -rf "$datadir"; mkdir -p "$datadir"
  echo "=== $label  (quant='${quant:-fp16}' flash=$flash) ==="

  WHISPERX_DATA_DIR="$datadir" \
  WHISPERX_PORT="$PORT" \
  WHISPERX_ASR_BACKEND=whispercpp \
  WHISPERX_MODEL=large-v3-turbo \
  WHISPERX_DEVICE=cpu \
  WHISPERX_GGML_QUANT="$quant" \
  WHISPERX_WHISPERCPP_FLASH_ATTN="$flash" \
  WHISPERX_LOG_LEVEL=info \
    "$SERVER" >"$OUT/$label.boot.log" 2>&1 &
  SERVER_PID=$!

  if ! wait_healthz 120; then echo "  healthz timeout — see $OUT/$label.boot.log"; cleanup; SERVER_PID=""; continue; fi

  sid=""
  for run in 1 2; do
    id="$(upload)" || break
    if wait_done "$id" 400; then
      echo "  run $run: session $id done"; sid="$id"
    else
      echo "  run $run: did not finish"; sid=""; break
    fi
  done

  if [[ -n "$sid" ]]; then
    cp -f "$datadir/sessions/$sid/transcript.json" "$OUT/$label.json" 2>/dev/null \
      && echo "  transcript -> $OUT/$label.json" || echo "  no transcript.json for $sid"
    # the server log carries the stage=… rtf=… lines (the timing instrument)
    cp -f "$datadir/logs/whisperx-server.log" "$OUT/$label.log" 2>/dev/null || true
  fi

  cleanup; SERVER_PID=""; sleep 1
done

echo
echo "done. analyse: uv run python3 scripts/bench_whispercpp_wer.py"
