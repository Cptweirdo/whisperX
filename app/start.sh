#!/usr/bin/env bash
# Run the WhisperX web app natively (no Docker), auto-selecting the best ASR
# backend for the host:
#
#   macOS Apple Silicon (arm64) -> whisper.cpp (Metal) + MLX (Apple GPU) extras
#                                  whisper.cpp is the fastest Mac ASR path
#                                  (1.57× faster than MLX on large-v3);
#                                  select it in Settings -> Compute Device.
#   Linux x86_64                -> CUDA GPU when available, else CPU
#   anything else               -> CPU
#
# Docker on Mac runs a Linux VM with no Metal passthrough (CPU only), so run
# this on the host to get Apple-GPU acceleration. On a CUDA Linux host this
# gives GPU acceleration too (torch is CUDA-pinned for x86_64 Linux in
# pyproject's [tool.uv.sources]).
#
# Usage:  ./app/start.sh
# Then open http://localhost:5000
#   (on Mac, set Settings -> Compute Device -> whisper.cpp (Metal) for best performance).
set -euo pipefail

# Repo root = parent of this script's dir, regardless of where it's invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OS="$(uname -s)"
ARCH="$(uname -m)"

# --- Pick backend + uv extras per platform ----------------------------------
# Arrays (not strings) so empty == no extra flags. The ${arr[@]+...} guard keeps
# expanding an empty array safe under `set -u` on bash 3.2 (macOS's default).
EXTRA=(--extra gdrive)  # cloud-backup deps; build cross-platform, no markers
BACKEND="CPU"
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  EXTRA+=(--extra mlx --extra whispercpp)  # mlx-whisper + pywhispercpp (Metal)
  BACKEND="whisper.cpp (Metal) + Apple GPU (MLX)"
elif [[ "$OS" == "Linux" ]]; then
  BACKEND="CUDA GPU (if present, else CPU)"
fi
echo "Platform: $OS $ARCH  ->  ASR backend: $BACKEND"

# --- C++ engine core (docs/cpp-core-handoff.md) -----------------------------
# Route as much of the pipeline as possible to the native `whisperx_core` module
# so Python only does orchestration glue + model resolution (HF downloads). The
# `asr` token swaps faster-whisper for sherpa-onnx Whisper, which in turn lets the
# decode-once native `run_job` orchestrator drive the whole compute chain
# (decode -> silero VAD -> ASR -> align -> diarize -> assign) with no per-stage
# Python re-entry; `db`/`edits`/`writers` move the session store + output writers
# native too. Every token hasattr/isinstance-guards, so a partial build just
# degrades that stage back to Python instead of failing.
#
# Enabled by default when the audio build is present. Overrides:
#   WHISPERX_NO_CORE=1        -> force the pure-Python pipeline
#   WHISPERX_CORE_STAGES=...  -> use your own token set (respected as-is)
#
# NB: the sherpa-onnx ASR in this build is **CPU-only** (no CUDA), so on a GPU
# host the native path is slower than faster-whisper on the GPU. Set
# WHISPERX_NO_CORE=1 to keep the GPU backend.
CORE_SO="$(ls "$REPO_ROOT"/build/whisperx_core*.so 2>/dev/null | head -1 || true)"
if [[ -n "${WHISPERX_NO_CORE:-}" ]]; then
  echo "C++ core: DISABLED (WHISPERX_NO_CORE set)  ->  pure-Python pipeline"
elif [[ -n "${WHISPERX_CORE_STAGES:-}" ]]; then
  echo "C++ core: using preset WHISPERX_CORE_STAGES='${WHISPERX_CORE_STAGES}'"
  export PYTHONPATH="$REPO_ROOT/build${PYTHONPATH:+:$PYTHONPATH}"
elif [[ -n "$CORE_SO" ]]; then
  # Full native set: store + edits + the whole compute chain + the orchestrator.
  export WHISPERX_CORE_STAGES="db,edits,decode,vad,asr,align,align_onnx,align_driver,assign,diarize,writers,orchestrate"
  export PYTHONPATH="$REPO_ROOT/build${PYTHONPATH:+:$PYTHONPATH}"
  echo "C++ core: ENABLED  ->  native run_job orchestrator (ASR = sherpa-onnx, CPU)"
  echo "          module: $CORE_SO"
  echo "          (set WHISPERX_NO_CORE=1 to use the GPU faster-whisper path)"
else
  echo "C++ core: NOT BUILT  ->  running the pure-Python pipeline." >&2
  echo "  To build the native path (one-time, needs cmake + ninja + ffmpeg dev libs):" >&2
  echo "    cmake -S . -B build -G Ninja -DWHISPERX_CORE_AUDIO=ON \\" >&2
  echo "      -DPython_EXECUTABLE=\"\$(uv run --no-project python -c 'import sys;print(sys.executable)')\"" >&2
  echo "    cmake --build build" >&2
fi

# --- ffmpeg (whisperx shells out to it to decode audio; see whisperx/audio.py) ---
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found on PATH. Install it:" >&2
  case "$OS" in
    Darwin) echo "  brew install ffmpeg" >&2 ;;
    Linux)  echo "  sudo apt-get install -y ffmpeg   (or your distro's package manager)" >&2 ;;
    *)      echo "  install ffmpeg via your platform's package manager" >&2 ;;
  esac
  exit 1
fi

# --- Frontend: the Svelte SPA built into app/static/spa (gitignored). Build with
# Bun/Vite if the build is missing. See app/web/.
if [[ ! -d app/static/spa ]]; then
  if ! command -v bun >/dev/null 2>&1; then
    echo "The web client (app/static/spa) isn't built and Bun isn't installed." >&2
    echo "Install Bun from https://bun.sh, then re-run" \
         "(or build once:  cd app/web && bun install && bun run build)." >&2
    exit 1
  fi
  echo "Building the web client with Bun/Vite…"
  (cd app/web && bun install && bun run build)
fi

# --- Python deps. Idempotent. On Mac this adds the mlx extra. -----------------
echo "Syncing Python deps…"
uv sync ${EXTRA[@]+"${EXTRA[@]}"}

# Flask + keyring are app-only deps (app/requirements.txt), not part of
# pyproject, so add them ephemerally for this run instead of mutating the
# project env. keyring stores the Hugging Face token in the OS keyring.
PORT="${PORT:-5000}"
echo "Starting WhisperX web app on http://localhost:${PORT} (Ctrl-C to stop)…"
PORT="$PORT" uv run ${EXTRA[@]+"${EXTRA[@]}"} --with "Flask>=3.0" --with "keyring>=24" python -m app.server
