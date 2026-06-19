# Building WhisperX

Two build systems live in this repo:

| Stack | Tooling | What it builds |
|-------|---------|----------------|
| **Python** (`whisperx/`, `app/`) | [`uv`](https://docs.astral.sh/uv/) + `pyproject.toml` | the CLI, library, Flask web app |
| **C++** (`core/`, `adapters/`) | CMake + Ninja (`CMakePresets.json`) | the native engine core, pybind oracle, and the oat++ HTTP/SSE server |

This document covers both, and the cross-platform helper that drives the C++
build on Linux / macOS / Windows.

---

## TL;DR

```bash
# C++ engine + native server (the host-swap server)
python scripts/devenv.py doctor          # check toolchain + system deps
python scripts/devenv.py deps            # install missing system deps
python scripts/devenv.py build server    # configure + build
python scripts/devenv.py test server     # run the Catch2/CTest suite
python scripts/devenv.py run             # launch whisperx_server on :8000

# Python CLI / web app
uv sync --all-extras
uv run whisperx audio.mp3 --model large-v2
```

On Linux/macOS you can use `scripts/devenv.sh …`; on Windows
`scripts/devenv.ps1 …`. Both just forward to `devenv.py`.

---

## The C++ build

### One driver: `scripts/devenv.py`

A stdlib-only Python script (3.8+, no `pip install` to bootstrap) that wraps
`CMakePresets.json` so the build matrix has a single source of truth.

| Command | Does |
|---------|------|
| `devenv.py doctor` | Reports the toolchain + heavy-lane libraries, flags anything missing. Start here. |
| `devenv.py deps [--no-update]` | Installs system deps via the detected package manager (`apt`/`dnf`/`pacman` · `brew` · `choco`+`vcpkg`). |
| `devenv.py build [preset]` | `cmake --preset` then `cmake --build --preset`. Default preset `server`. `--target X` / `-j N` supported. |
| `devenv.py test [preset]` | `ctest --preset`. |
| `devenv.py run [-- args…]` | Launches `whisperx_server`, setting the runtime library path (oat++ + ONNX Runtime shared libs) per-OS so you don't manage `LD_LIBRARY_PATH` by hand. |

You can always bypass the script and use CMake directly:

```bash
cmake --preset server && cmake --build --preset server
```

### Build presets (`CMakePresets.json`)

| Preset | Options | Needs |
|--------|---------|-------|
| **dev** | core lib + pybind oracle + Catch2 tests | nothing beyond a C++20 compiler — light deps are fetched by CMake `FetchContent`. The fast CI lane. |
| **audio** | dev **+** `WHISPERX_CORE_AUDIO` (in-process libav* decode, sherpa-onnx VAD/ASR/align/diarize, ONNX Runtime) | system **ffmpeg** dev libs |
| **server** | audio **+** `WHISPERX_BUILD_SERVER` (the native oat++ HTTP/SSE host) | ffmpeg + **curl** + **libarchive** + the OS secret store |
| **server-vcpkg** | same as `server`, but resolves the C/C++ libs from **vcpkg** | `VCPKG_ROOT` set (the Windows / hermetic path) |
| **server-cuda** / **server-vcpkg-cuda** | server **+** `WHISPERX_ENABLE_GPU` (the CUDA ONNX Runtime EP — runs ASR/align/diarize on GPU) | CUDA toolkit 12.x + cuDNN + an NVIDIA GPU; the vcpkg variant uses **clang-cl** on Windows. **See [`docs/WINDOWS_CUDA.md`](docs/WINDOWS_CUDA.md).** |

All presets use the Ninja generator, `build/` as the binary dir,
`RelWithDebInfo`, and export `compile_commands.json`.

### System dependencies

Light deps (pybind11, Catch2, nlohmann/json, spdlog, oat++, keychain,
sherpa-onnx → ONNX Runtime) are pulled by CMake `FetchContent` — **no package
manager needed for those**. Only the heavy native libraries come from the
system (or vcpkg):

| Library | Linux (apt) | macOS (brew) | Windows |
|---------|-------------|--------------|---------|
| ffmpeg (libav*) | `libav{format,codec,util}-dev`, `libswresample-dev` | `ffmpeg` | vcpkg |
| curl | `libcurl4-openssl-dev` | system (CommandLineTools) | vcpkg |
| libarchive | `libarchive-dev` | `libarchive` | vcpkg |
| OS secret store | `libsecret-1-dev` | Security.framework (built-in) | wincred (built-in) |
| CUDA (GPU build only) | CUDA toolkit 12.x + cuDNN + driver | — | CUDA toolkit 12.x + cuDNN + driver |

**GPU / CUDA:** the `server-cuda` / `server-vcpkg-cuda` presets build against
sherpa-onnx's CUDA ONNX Runtime. On Windows build with **clang-cl** (the preset selects
it) so the `-ffp-contract=off` parity flag is honored. Full walkthrough + the GPU-dispatch
validation checklist: [`docs/WINDOWS_CUDA.md`](docs/WINDOWS_CUDA.md).

`devenv.py deps` installs the right set for your platform. On Fedora/Arch it
maps to `dnf`/`pacman`; the exact ffmpeg-dev package name varies by distro, so
if `doctor` still flags a lib after `deps`, install the `-devel`/`-dev` package
for it manually.

#### Windows / vcpkg

The heavy libs are painful to get system-wide on Windows, so use vcpkg:

```powershell
git clone https://github.com/microsoft/vcpkg
.\vcpkg\bootstrap-vcpkg.bat
$env:VCPKG_ROOT = "$PWD\vcpkg"
python scripts\devenv.py deps                 # vcpkg install (reads vcpkg.json)
python scripts\devenv.py build server-vcpkg
```

`vcpkg.json` is the manifest listing those deps. The same path works on
Linux/macOS if you prefer hermetic deps over system packages — set `VCPKG_ROOT`
and use the `server-vcpkg` preset.

### Running the server

`devenv.py run` sets the shared-library path and a few dev defaults
(`WHISPERX_DATA_DIR=./.devdata`, `WHISPERX_MODEL=tiny`, `WHISPERX_PORT=8000`)
then launches the binary. Assets lazy-download from the public mirrors on first
use — no env vars required. Override anything by exporting it first:

```bash
WHISPERX_MODEL=base WHISPERX_PORT=5000 python scripts/devenv.py run
```

Key env vars: `WHISPERX_DATA_DIR`, `WHISPERX_HOST`/`WHISPERX_PORT`,
`WHISPERX_MODEL`, `WHISPERX_LOG_LEVEL`, `WHISPERX_MAX_UPLOAD_MB`, the backup vars
(`WHISPERX_BACKUP_BACKEND`, `WHISPERX_BACKUP_DIR`, `WHISPERX_BACKUP_INTERVAL`)
and `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` for Drive. See
`docs/cpp-core-host-swap-status.md` for the full list.

---

## The Python build

Uses `uv` (not plain pip for dev):

```bash
uv sync --all-extras          # project + dev deps (pytest)
uv run pytest tests/ -v       # tests
uv lock --check               # CI gate: lockfile up to date
uv run whisperx audio.mp3 --model large-v2 --diarize
```

`torch` is pinned to the CUDA 12.8 wheel index (`cu128`) on x86_64 Linux and the
CPU index elsewhere — see `[tool.uv.sources]` in `pyproject.toml`. GPU needs
CUDA toolkit 12.8; cuDNN load failures are common (`CUDNN_TROUBLESHOOTING.md`).

Web app (`app/`): `cd app/web && bun install && bun run dev` for the SPA; see
`CLAUDE.md` for the full front-end workflow.

---

## CI parity

- Python: `.github/workflows/` test Python 3.10–3.13 and enforce `uv lock --check`
  + a bare `import whisperx` smoke test.
- C++: the `dev` preset is the dep-free fast lane (no ffmpeg/ORT); `audio`/`server`
  add the heavy deps. Tests build with ASan + UBSan (`WHISPERX_CORE_SANITIZE`).
