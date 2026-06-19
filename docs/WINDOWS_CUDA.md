# Building & Testing the Native Server with CUDA (Windows)

This guide builds the native C++ server (`whisperx_server`) with the **CUDA ONNX
Runtime execution provider** so the ASR / align / diarize stages can run on an NVIDIA
GPU, and walks through validating that the GPU path actually works (the spike from
`GPU_INTEGRATION.md`).

The runtime **cpu⇄cuda device switch** is already implemented (`POST /api/device`,
`WHISPERX_DEVICE`, status `cuda_available`); it is *inert* in a CPU build. This guide
produces the GPU build that lights it up.

> For the general (CPU) build matrix and the `devenv.py` driver, see
> [`BUILDING.md`](../BUILDING.md). This doc is the GPU/Windows-specific path.

---

## Why clang-cl (read this first)

The CMake build applies `-ffp-contract=off` to the parity-sensitive stages
(diarize speaker-assignment, alignment log-softmax) so results match the Python/torch
reference bit-for-bit. **MSVC `cl` silently ignores `-ffp-contract=off`** (emits warning
`D9002` and continues), which lets the compiler fuse multiply-adds and drift the
numbers — diarization clustering and parity tests can then diverge.

`clang-cl` honors `-ffp-contract=off` (and `-Wall`/`-Wextra`). **Build with clang-cl.**
The `server-vcpkg-cuda` preset selects it automatically on Windows (the hidden
`windows-clang` preset). clang-cl still uses the MSVC STL + Windows SDK, so you install
the Visual Studio Build Tools either way.

---

## Prerequisites

| Component | Notes |
|-----------|-------|
| **NVIDIA driver** | Recent enough for your CUDA toolkit. Check with `nvidia-smi`. |
| **CUDA Toolkit 12.x** | Matches what sherpa-onnx's GPU ONNX Runtime expects. Adds `nvcc`, sets `CUDA_PATH`. |
| **cuDNN** | Version matching the CUDA toolkit; its DLLs must be on `PATH` at runtime (see Troubleshooting / [`CUDNN_TROUBLESHOOTING.md`](../CUDNN_TROUBLESHOOTING.md)). |
| **Visual Studio Build Tools** | Provides the MSVC STL + Windows SDK that clang-cl links against. "Desktop development with C++" workload. |
| **LLVM / clang-cl** | The compiler. Install LLVM (e.g. `choco install llvm`) and ensure `clang-cl` is on `PATH`. |
| **CMake ≥ 3.24, Ninja, Git** | `choco install cmake ninja git` (or via the VS installer). |
| **vcpkg** | Supplies the heavy C/C++ libs (ffmpeg, curl, libarchive, onnxruntime). |

Run all commands from a shell where the MSVC environment + `clang-cl` are on `PATH`
(e.g. the **x64 Native Tools Command Prompt for VS**, or a PowerShell that has run
`vcvars64.bat`), so clang-cl finds the STL/SDK.

---

## 1. Set up dependencies

```powershell
# vcpkg (one-time)
git clone https://github.com/microsoft/vcpkg
.\vcpkg\bootstrap-vcpkg.bat
$env:VCPKG_ROOT = "$PWD\vcpkg"

# install build tools (choco) + the C/C++ libs (vcpkg.json manifest)
python scripts\devenv.py deps

# sanity-check the toolchain + that CUDA is visible
python scripts\devenv.py doctor
```

`doctor` reports an optional **GPU / CUDA** section — confirm `nvcc` and `nvidia-smi`
show `ok` before building the GPU preset. (They are optional for the CPU lanes, required
for `-cuda`.)

---

## 2. Build (GPU)

```powershell
python scripts\devenv.py build server-vcpkg-cuda
```

This runs `cmake --preset server-vcpkg-cuda` then builds. The preset:
- sets `WHISPERX_ENABLE_GPU=ON` → flips `SHERPA_ONNX_ENABLE_GPU` ON, so sherpa-onnx
  fetches the **CUDA** ONNX Runtime (this is a larger download + longer first configure),
- resolves heavy deps from vcpkg (`VCPKG_ROOT`),
- selects **clang-cl** (the `windows-clang` preset, Windows-only).

Equivalent raw CMake (if you bypass the driver):

```powershell
cmake --preset server-vcpkg-cuda
cmake --build --preset server-vcpkg-cuda
```

> **CPU comparison build** (for parity testing in step 4e): use the `server-vcpkg`
> preset — same toolchain, CPU ORT.

---

## 3. Run

```powershell
# launch on the GPU directly
$env:WHISPERX_DEVICE = "cuda"
python scripts\devenv.py run
```

`devenv.py run` sets `PATH` to include the build's shared libs (oat++, ONNX Runtime).
You must **also** have the CUDA + cuDNN DLLs on `PATH` (the toolkit's `bin` and cuDNN's
`bin`); otherwise ORT fails to load the CUDA provider — see Troubleshooting.

Two ways to choose the device:
- **At boot:** `WHISPERX_DEVICE=cuda` (env). Precedence is *persisted setting > env > cpu*.
- **At runtime:** `POST /api/device {"device":"cuda"}` — the live switch (no restart).

Defaults from `devenv.py run`: `WHISPERX_PORT=8000`, `WHISPERX_MODEL=tiny`,
`WHISPERX_DATA_DIR=.\.devdata`. Override any by exporting first.

---

## 4. Test

### CTest suite (must stay green)

```powershell
python scripts\devenv.py test server-vcpkg-cuda
```

### GPU dispatch spike checklist

This is the real validation — confirming the GPU build *dispatches to CUDA* and stays
correct. The big risk (per `GPU_INTEGRATION.md`) is a **silent CPU-node fallback**: the
build links the GPU ORT but a graph quietly runs on CPU.

Assume the server is running on `:8000`.

- **(a) Capability detected.** `GET /api/models` reports `"cuda_available": true`.
  (False here means the GPU ORT/driver isn't actually loadable — fix before continuing.)

  ```powershell
  curl http://127.0.0.1:8000/api/models
  ```

- **(b) Switch accepted.** `POST /api/device {"device":"cuda"}` → **200**, and the
  returned status shows `"device":"cuda"`.

  ```powershell
  curl -X POST http://127.0.0.1:8000/api/device -H "Content-Type: application/json" -d "{\"device\":\"cuda\"}"
  ```

- **(c) Actually on the GPU.** While a transcription job runs, watch `nvidia-smi`
  (`nvidia-smi -l 1`) for GPU utilization + memory growth. Additionally enable verbose
  ORT/sherpa logging to catch nodes assigned to the CPU provider — **no GPU util during
  a job ⇒ silent fallback**, investigate before trusting the build.

- **(d) End-to-end.** Upload a real audio file and run a full transcribe→align→diarize
  job on `cuda`; confirm it completes and the transcript is sane.

- **(e) Parity vs CPU.** Run the same audio on a `server-vcpkg` (CPU) build and compare:
  - Whisper transcription WER/CER (text isn't bit-stable; judge by error rate),
  - wav2vec2 alignment word timestamps — test **both** model graph types
    (`layer_norm`/batchable **and** `group_norm`/per-segment),
  - diarization DER + speaker count.
  Small drift is expected (GPU kernels aren't bit-identical); large divergence —
  especially in diarization clustering — means a parity flag was lost (re-check you
  built with **clang-cl**, not `cl`).

- **(f) Switch-stress / leak check.** Cycle `cpu→cuda→cpu` many times via `/api/device`
  while watching `nvidia-smi`: VRAM should return to baseline after each switch back to
  cpu. A monotonic climb means the eviction/rebuild in `ModelManager::set_device` leaks
  CUDA context/VRAM across switches.

- **(g) Cold-start cost.** Note the latency of the **first** job after switching to cuda
  (CUDA context creation + cuDNN autotune) — it's a one-time hit; confirm it doesn't blow
  the first job's ETA unacceptably.

### Runtime-switch behavior

- `cpu → cuda → cpu`: after each switch, `GET /api/models` reflects the new `device`, and
  a job completes on whichever device is active.
- **Busy gate:** start a long job, then `POST /api/device` mid-job → expect **409**
  (`"busy"`); the running job is unaffected (engine borrows are `shared_ptr`s).
- **Unknown/unavailable:** `{"device":"gpu0"}` → 400 "Unknown device"; on a CPU build,
  `{"device":"cuda"}` → 400 "CUDA device not available in this build."

---

## Troubleshooting

- **`cuda_available:false` in a GPU build.** ORT can't load the CUDA provider — almost
  always a missing/mismatched DLL. Ensure the CUDA toolkit `bin` and cuDNN `bin` are on
  `PATH`; verify cuDNN matches the CUDA version. See
  [`CUDNN_TROUBLESHOOTING.md`](../CUDNN_TROUBLESHOOTING.md).
- **DLL not found at startup.** `devenv.py run` adds the build libs to `PATH` but not
  CUDA/cuDNN — add those `bin` dirs yourself.
- **Silent CPU fallback (GPU util stays at 0).** Enable ORT verbose logging; check
  whether the Whisper / wav2vec2 / pyannote graphs report nodes on `CPUExecutionProvider`.
  This is the headline spike unknown — a graph may not be fully CUDA-supported in the
  pinned sherpa-onnx `v1.13.2`.
- **Parity tests / diarization drift.** You likely built with MSVC `cl`, which dropped
  `-ffp-contract=off`. Rebuild with **clang-cl** (`server-vcpkg-cuda` selects it; confirm
  `clang-cl` is on `PATH` and re-configure from a clean `build/`).
- **`clang-cl` not found.** Install LLVM and open a shell that has both the MSVC
  environment (`vcvars64`) and LLVM on `PATH`.
- **`WHISPERX_ENABLE_GPU requires WHISPERX_CORE_AUDIO`.** Use the `-cuda` presets (they
  pair them); don't set the GPU option on the dep-free lanes.
- **Stale CMake cache after switching presets.** GPU vs CPU presets share `build/`;
  delete `build/` if a reconfigure complains about changed cache variables.
