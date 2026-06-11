# CUDA Whisper Decode — Handoff Brief (SPEEDUP_FINDINGS item 1a)

Status: **not started** — this brief hands off the diagnosis + fix of the
host-bound CUDA decode path discovered while landing item 1 (batched decode).
Written 2026-06-11 on branch `cpp-core/host-swap-server` (item 1 uncommitted
on that branch at the time of writing).

## Mission

Serial CUDA transcription runs at **RTF 0.443 with a ~20% sawtooth GPU
utilization** (3080 Ti, `large-v3-turbo` fp32, 10 min audio, native server).
That is several-fold slower than the hardware should allow — whisper.cpp does
RTF 0.070 on an Apple laptop (`MAC_PERFORMANCE.md`), batched CUDA pipelines do
0.01–0.03 (`SPEEDUP_FINDINGS.md` corroboration table). The GPU being idle
~80% of the time means the bottleneck is host-side (sync, allocation, CPU
phases), not FLOPs. **Find where the per-chunk wall time actually goes, fix
the dominant cost, then re-evaluate the already-landed batch decode.**

## Hard data (measured 2026-06-11, scratch server :8011)

Setup: 3080 Ti 12 GB, `large-v3-turbo` (sherpa fp32 ONNX from the
KonstantK/whisper-onnx-sherpa mirror), 607 s test file (golden
`en_dialog.wav` tiled ×10 — recipe below), `WHISPERX_DEVICE=cuda`.

| Config | transcribe wall | RTF | VRAM | GPU util |
|---|---|---|---|---|
| `WHISPERX_ASR_BATCH_SIZE=1` (serial) | 269.25 s | 0.443 | ~4.1 GB | sawtooth 1–30% |
| `WHISPERX_ASR_BATCH_SIZE=8` (batched) | 319.31 s | 0.526 | ~7.2 GB | spikes to 55%, valleys 1–10% |

Other stages, same run (for proportion): align RTF 0.002, diarize 0.062 (CPU),
decode 0.000. Transcribe is ~79% of job wall time — the only stage worth
optimizing on CUDA right now.

Two facts to explain, one suspicious bonus fact:
1. Serial RTF 0.443 is ~10× worse than a rough FLOPs/bandwidth estimate for
   this card (encoder ~2 TFLOP/window, decoder ~150 weight-bound steps/window).
2. Batch 8 does ~8× fewer decoder `Run`s yet is **slower** — so the dominant
   per-step cost must scale ≥ linearly with batch size, or the bottleneck is
   outside the decoder loop entirely.
3. The batch-8 VRAM delta (+3.1 GB) is far larger than the tensors require
   (~0.8 GB) — consistent with ORT CUDA arena over-extension / churn.

## What is ALREADY true in the code (verified — don't re-derive)

**Disproven hypothesis, do not chase it:** "sherpa round-trips the KV caches
through host memory every step." It does not. sherpa v1.13.2 already does
CUDA IOBinding in the whisper model wrapper. Verified anchors (all in
`third_party/sherpa-onnx-patches/sherpa-onnx/csrc/offline-whisper-model.cc`,
which is byte-identical to upstream in these regions):

- `InitCudaIOBinding()` (~line 431): `use_cuda_iobinding_ = provider == cuda`;
  `cuda_mem_info_ = ("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault)`.
- `ForwardEncoder` (~line 127): binds both outputs (cross-K/V) to
  `cuda_mem_info_` → **cross-KV lives on GPU** for the whole decode.
- `ForwardDecoder` (~line 153): per step constructs a **fresh
  `Ort::IoBinding`**, `BindInput` ×6, binds logits → CPU, out self-K/V → GPU,
  then `SynchronizeInputs()` → `Run` → `SynchronizeOutputs()`. Self-KV
  output of step *i* is the (GPU-resident) input of step *i+1*.
  **Cross-KV is not a graph output** — it's passed back as the moved C++
  inputs (lines ~200–205). Only logits (~200 KB/step serial, ~1.7 MB at
  batch 8) cross PCIe per step, plus the tiny tokens/offset uploads.

So the per-token loop is already "device-resident KV"; what remains per step:
fresh IoBinding construction, 6 BindInput calls, **two stream synchronizations**
(`SynchronizeInputs` + `SynchronizeOutputs`), a fresh GPU allocation for each
self-KV output (18 MB serial / 147 MB at batch 8, every step), logits DtoH,
CPU argmax.

## Ranked suspects (what profiling should confirm/kill)

1. **Per-step GPU output allocations → arena churn.** Every `Run` allocates
   new out self-K/V buffers from the ORT CUDA arena (`BindOutput(mem_info)`
   semantics). 147 MB/step at batch 8 explains both the regression (cost
   scales with batch) and the +3.1 GB VRAM. Fixes: pre-allocate two
   ping-pong buffer pairs once per decode and `BindOutput(name, Ort::Value)`
   instead of `BindOutput(name, MemoryInfo)`; and/or set CUDA provider arena
   options (`arena_extend_strategy=kSameAsRequested`, sized
   `initial_chunk_size`) in sherpa's session options (`session.cc` /
   `GetSessionOptions`).
2. **Two `cudaStreamSynchronize` + fresh `IoBinding` per token.** ~3000 steps
   serial on the 10-min file; even 1–2 ms of fixed overhead per step ≈ tens
   of seconds. Fixes: hoist one `Ort::IoBinding` per decode loop and rebind
   only what changes; drop `SynchronizeInputs()` (inputs are device-resident
   or tiny CPU tensors staged on the compute stream); keep a single sync
   before the CPU logits read. Caveat: this suspect alone can't explain the
   batch-8 regression (batch has 8× fewer steps) — if profiling shows this
   dominates serial, something else still has to explain batch 8.
3. **fp32 encoder with unfused attention is just slow.** The encoder burst is
   the 30–55% spike; if profiling shows encoder ≫ decoder, jump straight to
   SPEEDUP_FINDINGS items 4 (fp16 conversion) + 5 (ORT transformers
   optimizer / fused attention), which shrink every buffer and kernel.
4. **CPU phases between decodes** (kaldi-fbank mel extraction per stream,
   `NormalizeFeatures`, `Transpose12`, stream setup) — serialized with GPU
   work today. Only worth attacking after 1–3, e.g. by overlapping feature
   extraction of window *i+1* with decode of window *i*.

## How to profile (do this first; ~an hour)

Pick one:
- **ORT profiling JSON** (easiest, per-kernel + per-Run gaps): in the patch
  copy of `offline-whisper-model.cc`, add
  `sess_opts_.EnableProfiling(ORT_TSTR("whisper-prof"))` in the `Impl`
  constructor before the sessions are created; rebuild; run one job; open the
  emitted `whisper-prof*.json` in `chrome://tracing` / Perfetto. The files
  land in the server's CWD.
- **Nsight Systems**: `nsys profile --trace=cuda,osrt -o whisper.nsys-rep`
  wrapping the server process (start it directly with the right PATH — see
  the devenv note below — or `nsys launch`). Look for: gaps between kernels
  (sync/alloc), `cudaMalloc` calls during steady-state decode (arena
  extension), HtoD/DtoH volume per step.
- **Poor-man's timers**: spdlog around `ForwardEncoder`, the per-step `Run`,
  and both `Synchronize*` calls in the patch files; coarse but enough to
  split encoder vs decoder vs sync.

Workload recipe (same as the measured table):
```powershell
# ~10 min test file
.venv\Scripts\python -c "import wave; r=wave.open(r'golden\clips\en_dialog.wav','rb'); p=r.getparams(); d=r.readframes(r.getnframes()); w=wave.open(r'build\e2e_long.wav','wb'); w.setparams(p); [w.writeframes(d) for _ in range(10)]; w.close()"
# scratch server — NEVER the user's :8000 instance (see memory: server-verification-workflow)
$env:VCPKG_ROOT="D:\playground\vcpkg"; $env:WHISPERX_DATA_DIR="D:\playground\whisperX\build\e2e-data"
$env:WHISPERX_PORT="8011"; $env:WHISPERX_MODEL="large-v3-turbo"; $env:WHISPERX_DEVICE="cuda"
$env:WHISPERX_ASR_BATCH_SIZE="1"   # and "8" for the comparison run
python scripts\devenv.py run       # devenv prepends build/bin to PATH (sherpa DLL) — don't launch the exe bare
curl.exe -F "audio=@build\e2e_long.wav" http://127.0.0.1:8011/api/sessions
# numbers: "runner: stage=transcribing elapsed=…s rtf=…" in build\e2e-data\logs\whisperx-server.log
```
Models resolve from the shared cache `~\.cache\whisperx-sherpa` (no downloads).

## Where the code lives / how the patch mechanism works

- **Edit only `third_party/sherpa-onnx-patches/sherpa-onnx/csrc/*`** — the
  CMake sherpa block (`CMakeLists.txt`, "whisperX batched-whisper patch"
  comment) copies these over the FetchContent checkout
  (`build/_deps/sherpa_onnx-src/...`) on every configure
  (`file(COPY_FILE … ONLY_IF_DIFFERENT)`). Editing the `_deps` copy gets
  overwritten. Re-diff the patch files when bumping the sherpa `GIT_TAG`
  (pinned v1.13.2; upstream whisper still has no batch decode as of
  2026-06-11).
- Patched files so far (item 1): `offline-recognizer-whisper-impl.h`
  (batched `DecodeStreams` + serial fallback), `offline-whisper-greedy-
  search-decoder.{h,cc}` (`DecodeBatch`, lockstep), `offline-whisper-model.
  {h,cc}` (`GetInitialSelfKVCache(n)`).
- Our side: `core/asr/whisper_sherpa.cpp` (window flattening, batch groups,
  recognizer-level language pinning — `apply_options`; `detect()` resets
  language to "" or jobs leak the previous language),
  `adapters/server/models/model_manager.cpp::build_asr_engine` (batch only on
  Cuda), `adapters/server/config.{hpp,cpp}` (`WHISPERX_ASR_BATCH_SIZE`,
  default 1 = off), `adapters/py/whisperx_core.cpp` (`batch_size` kwarg).

## Verification gates (all must stay green)

1. Build: `python scripts\devenv.py build server-vcpkg-cuda`
   (`$env:VCPKG_ROOT` per session). `ctest --test-dir build` → 201/201.
2. Parity (catches text drift from any decode-loop change):
   `cmake --build build --target whisperx_core` (pyd is pinned to the
   project venv's CPython 3.12), then
   `$env:PYTHONPATH="D:\playground\whisperX\build"; $env:RUN_MIRROR="1";
   uv run pytest bindings/test/test_asr_sherpa_parity.py -v` (23 tests:
   WER/CER baseline, language detection + leak test,
   `test_batched_decode_matches_serial`). Requires `uv sync --all-extras`
   once — without it `uv run pytest` silently falls back to anaconda's
   Python and everything skips.
3. E2E: the workload recipe above, batch 1 vs 8 RTF + `nvidia-smi` during the
   stage; two consecutive jobs in different languages must each detect their
   own language. Clean up `build\e2e-data` + `build\e2e_long.wav` after.
4. VRAM budget: 12 GB card that also runs a browser — keep batch-8 footprint
   well under ~8 GB or lower the recommended batch.

## Definition of done

- Profiling artifact (trace or timer table) attributing serial transcribe
  wall time to encoder / decoder-step / sync / alloc / CPU phases.
- The dominant cost fixed in the sherpa patch (or sherpa session options),
  serial RTF substantially below 0.443 on the reference workload.
- Batch sweep (1/2/4/8) re-measured; `WHISPERX_ASR_BATCH_SIZE` default
  flipped to the winner only if it beats serial; `SPEEDUP_FINDINGS.md` item
  1/1a updated with the new numbers (and this brief marked superseded).
