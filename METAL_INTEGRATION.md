# Metal / Apple-Silicon Integration Brief — Native C++ Pipeline (Stages 1–3)

Companion to `GPU_INTEGRATION.md` (the CUDA brief, whose plumbing has since landed:
`Device{Cpu,Cuda}`, runtime `set_device()`, provider threaded to every engine,
`WHISPERX_GPU_BUILD`, `server-cuda` preset). This brief answers: **what can we
accelerate with Metal on Apple Silicon, ranked by ROI and by ease/risk?**

## General overview

On CUDA there was **one lever** — the ORT CUDA execution provider, threaded everywhere
through the now-landed `Device` plumbing. On Apple Silicon there are **two levers, and
they are not the same kind of thing**:

- **Route A — provider swap: ORT CoreML EP.** Slots into the *existing* `Device`
  plumbing almost for free (sherpa-onnx already parses the `"coreml"` provider string,
  and the macOS ORT prebuilt already ships the CoreML EP). Applies to all three stages.
  But the research record says CoreML EP is **frequently a wash or a regression** for
  transformer ASR graphs — treat it as a cheap, timeboxed spike, not a plan.
- **Route B — engine swap: whisper.cpp + GGML Metal, Stage 1 only.** A different
  runtime with different model assets (ggml `.bin`, not ONNX), so it is *not* a
  `Device` — it's a new ASR backend dimension. It is also the only route with strong
  evidence of large real-world speedups for Whisper on Apple GPUs.

**Headline ranking (easy/low-risk first, then ROI):**

1. **Phase 0 (free):** macOS arm64 **CPU** build + RTF baseline. Multi-threaded CPU via
   the landed `threads_for()` (`model_manager.cpp:32-36`, half the cores) on NEON/MLAS
   is genuinely strong — this is what Metal must beat, not the old 1-thread strawman.
2. **Phase 1 (easy, low-risk, low expected win):** `Device::CoreML` spike — near-pure
   reuse of the CUDA plumbing. Ship only if measurably ≥ CPU; evidence predicts it won't be.
3. **Phase 2 (highest ROI):** whisper.cpp Metal backend for Stage 1 — the only route
   with credible ~2× end-to-end impact, since Stage 1 dominates wall-clock.
4. **Phase 3 (optional):** whisper.cpp's `WHISPER_COREML` ANE encoder on top of Phase 2.
5. **Not now:** ORT WebGPU EP, MLX.

Pipeline (same as the CUDA brief), annotated with which route applies:

```
decode (ffmpeg, CPU — out of scope)
  └─ Stage 1  ASR        Whisper enc/decoder   sherpa-onnx   RTF ~0.10   ← Route A or B
  └─ Stage 2  Align      wav2vec2-CTC          direct ORT    RTF ~0.15   ← Route A only
  └─ Stage 3  Diarize    segmentation+embed    sherpa-onnx   RTF ~0.15   ← Route A only
```

VAD (silero) stays CPU-pinned by design — `vad_silero.cpp:36-39` documents it as
deliberately outside the device knob (tiny net, serialized 512-sample drain loop, poor
GPU occupancy). Decode, trellis Viterbi, log-softmax, speaker assignment, writers: same
out-of-scope list as the CUDA brief.

### Cross-cutting enablers

1. **Phase 0 — macOS build viability.** Nothing in this tree has ever been compiled on
   macOS. `CMakePresets.json` has only Linux/Windows presets
   (`dev/audio/server/server-vcpkg/server-cuda/server-vcpkg-cuda`). Add a
   `server-macos` preset (plain `server` should nearly work — `config.cpp` already has
   the `__APPLE__` data-dir branch) and confirm the whole `whisperx_server` target +
   tests build on arm64. sherpa-onnx v1.13.2 handles the ORT side automatically: its
   `cmake/onnxruntime-osx-arm64.cmake` downloads the **official ORT 1.24.4 osx-arm64
   prebuilt**, which **already includes the CoreML EP**
  ([ORT CoreML EP docs](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html),
  [sherpa pin](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.2/cmake/onnxruntime-osx-arm64.cmake)).
   So unlike CUDA there is **no `SHERPA_ONNX_ENABLE_GPU`-style build flag** for Route A
   — the EP is compiled in, gated only by sherpa's `__APPLE__` + ORT-version checks.
2. **Honest CPU baseline.** Record RTF medians (`runner.cpp::stage_rtf()`) on the
   target Mac with the CPU build *before* any Metal work. Every later claim is measured
   against this, not against historical single-thread numbers.
3. **Two config dimensions, not one.**
   - Route A extends the existing enum: `Device { Cpu, Cuda, CoreML }`
     (`adapters/server/config.hpp:23` — its comment already anticipates extension) plus
     `parse_device`/`to_string` cases (`config.cpp:114,123`; `to_string` returning
     `"coreml"` *is* the sherpa provider string). Everything downstream —
     `ModelManager::set_device()` with rollback, the provider threading at
     `model_manager.cpp:168-171/244-245/284`, `POST /api/device` validation + 409 busy
     gate (`api_controller.hpp:466-500`), the persisted setting, `WHISPERX_DEVICE` —
     is reused unchanged.
   - Route B is **not a Device**. whisper.cpp replaces the Stage 1 *engine*, while
     stages 2–3 stay on ORT and still need their own cpu/coreml provider choice.
     Conflating it into the Device enum (e.g. `Device::Metal`) would break that. Add an
     orthogonal `enum class AsrBackend { Sherpa, WhisperCpp }` + `WHISPERX_ASR_BACKEND`
     env + persisted setting, surfaced either as a new `POST /api/asr_backend` or an
     extended `/api/device` payload.
4. **Honest status.** `status()` already emits the placeholders this brief fills in:
   `{"mlx_available", false}` and `{"whispercpp_available", false}`
   (`model_manager.cpp:85-86`) next to the real `cuda_available()` probe. Add
   `coreml_available` (probe `Ort::GetAvailableProviders()` for
   `"CoreMLExecutionProvider"`, `#ifdef __APPLE__`-guarded — the analog of
   `ort_cuda_available()`, `wav2vec2_onnx.cpp:11-23`) and flip `whispercpp_available`
   to a compile-time `WHISPERX_WHISPERCPP_BUILD` define + asset-presence check.
5. **CoreML compile cache.** The CoreML EP recompiles the model at session creation
   unless `ModelCacheDirectory` is set, which "may cost significant time" per the ORT
   docs. This interacts badly with `set_device()`'s evict-and-rebuild semantics: every
   cpu→coreml switch pays the compile for every resident engine unless the cache dir is
   configured (and sherpa's append path exposes no way to set it — see Route A risks).

---

## Route A — CoreML EP via the existing Device plumbing (all 3 stages)

**The cheap spike. Cost: ~1 day of plumbing. Risk: low (worst case it's slower and we
don't ship it). Expected ROI: low, possibly negative — set expectations accordingly.**

### How it would be implemented

- **Stages 1 & 3 (sherpa-engined): zero engine changes.** sherpa v1.13.2's
  [`provider.cc`](https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v1.13.2/sherpa-onnx/csrc/provider.cc)
  maps the string `"coreml"` → `Provider::kCoreML`, and
  [`session.cc`](https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v1.13.2/sherpa-onnx/csrc/session.cc)
  appends the CoreML EP under `__APPLE__`. `WhisperSherpa` and `SherpaDiarizer`
  already take the provider string (`whisper_sherpa.hpp:50-54`,
  `diarize_sherpa.hpp:40-43`); `to_string(Device::CoreML) == "coreml"` does the rest.
- **Stage 2 (direct `Ort::Session`):** add a CoreML branch in `make_options()`
  (`core/align/wav2vec2_onnx.cpp:36-52`) next to the existing
  `#ifdef WHISPERX_GPU_BUILD` CUDA branch, guarded by `__APPLE__` only (no special ORT
  build). Use the **modern** options API here — we control this call site:
  `opts.AppendExecutionProvider("CoreML", {{"ModelFormat","MLProgram"},
  {"MLComputeUnits","CPUAndGPU"}, {"ModelCacheDirectory", …}})`.
- Capability detection + status as per enabler #4; `/api/device` accepts `"coreml"`
  once `parse_device` does.

### Why expectations must be low

- **sherpa uses the legacy append API.** `session.cc` calls
  `OrtSessionOptionsAppendExecutionProvider_CoreML(opts, /*flags=*/0)` — the API
  deprecated in ORT 1.20, which selects the older **NeuralNetwork** model format, not
  MLProgram (the better-covered path). We cannot choose MLProgram for stages 1 & 3
  without patching sherpa.
- **Dynamic shapes are CoreML EP's weak spot.** Whisper's decoder/KV-cache loop,
  wav2vec2's variable-length padded batches, and the diarization streams are all
  dynamic-shape; CoreML EP either degrades or pushes those nodes back to CPU, causing
  CPU↔CoreML partition ping-pong
  ([onnxruntime#14212](https://github.com/microsoft/onnxruntime/issues/14212),
  [onnxruntime#16934](https://github.com/microsoft/onnxruntime/issues/16934)).
- **Field report directly on point:**
  [sherpa-onnx#2910](https://github.com/k2-fsa/sherpa-onnx/issues/2910) measured CoreML
  ~10% **slower** than CPU for a transformer ASR model on an M2 Max.

### Per-stage sub-verdicts

- **Stage 1 (Whisper):** the encoder is fixed-shape (30 s mel) so it has the best shot
  at full CoreML placement; the autoregressive decoder is dynamic → likely partial
  placement and ping-pong. Net effect unpredictable; measure.
- **Stage 2 (wav2vec2):** we control the session options (MLProgram, static-shape
  experiments possible) but batched dynamic shapes are the EP's worst case, and this
  stage's batching is the thing that makes it fast.
- **Stage 3 (diarize):** two small models + CPU clustering → marginal even if placement
  is clean. Also re-check diarization parity: GPU drift in embeddings can nudge
  cluster assignments (same caveat as the CUDA brief).

### Verification spike

On an Apple Silicon machine: run the RTF harness (`runner.cpp::stage_rtf()` medians)
with `WHISPERX_DEVICE=coreml` vs `cpu` on the same clips, and read ORT verbose logs for
**node-assignment counts** per graph — silent CPU fallback is invisible otherwise.
Timebox; keep only if measurably ≥ CPU.

---

## Route B — whisper.cpp + GGML Metal for Stage 1 (the recommended investment)

**Cost: medium (new engine class + asset family + CMake). Risk: medium (new dependency,
output drift). Expected ROI: high — the only credible 2×+ lever on the dominant stage.**

### Why

Metal is whisper.cpp's first-class backend — enabled by default on Apple Silicon, full
inference on the GPU with unified memory (no H2D/D2H copies), flash-attention
(`whisper_context_params.flash_attn`) and quantized models (Q5_0 etc.)
([whisper.cpp README](https://github.com/ggml-org/whisper.cpp)). Reported speedups run
30–60%+ over CPU and widen with model size; large-v3 reaches ~2–3× realtime on
M3/M4-class machines
([benchmark roundup](https://www.promptquorum.com/local-llms/apple-silicon-whisper-metal-benchmark),
[fazm.ai](https://fazm.ai/blog/whisper-cpp-metal-apple-silicon)). Since Stage 1 is the
heaviest stage, this is where Metal actually moves end-to-end wall-clock.

### How it would be implemented

- **No ASR interface exists to inherit** — `WhisperSherpa` is concrete. Its surface is
  small and whisper.cpp's C API covers it 1:1:
  - `transcribe(AudioBuffer, spans, language, task) → vector<AsrChunk>`
    (`whisper_sherpa.hpp:66-69`): feed each pre-VAD'd span of 16 kHz float PCM —
    exactly what the transcribe closure in `runner.cpp` already produces — to
    `whisper_full()` with greedy params; build `AsrChunk{text, avg_logprob}` from
    segments + `whisper_token_data.p`. Stage 2 re-times every word, so whisper.cpp's
    segment timestamps only need to be approximately right (no DTW/token-timestamps
    needed).
  - `detect_language(AudioBuffer)` (`whisper_sherpa.hpp:73`): `whisper_lang_auto_detect`
    over the first 30 s.
- New concrete `core/asr/whisper_cpp.{hpp,cpp}` mirroring those method shapes (reuse
  `AsrChunk` + `strip_blank_audio` from `whisper_sherpa.hpp`), constructed via
  `whisper_init_from_file_with_params` (`use_gpu=true`, optionally `flash_attn`).
- Branch in `ModelManager::build_asr_engine()` (`model_manager.cpp:157`) selected by
  the new `AsrBackend`; the runner's `Steps.transcribe` closure shape is unchanged.
  Since the manager caches/returns `shared_ptr<WhisperSherpa>` today, the cleanest cut
  is a small variant/wrapper (or an interface extracted at this point) so the cache and
  closure are backend-agnostic.
- **CMake:** `FetchContent` whisper.cpp pinned to a release tag, behind a new
  `option(WHISPERX_ENABLE_WHISPERCPP)` + `WHISPERX_WHISPERCPP_BUILD` define (the
  pattern at `CMakeLists.txt:45,181-183`); `GGML_METAL` defaults on for Apple. Lands
  next to the sherpa FetchContent (`CMakeLists.txt:198-201`).

### Asset implication (don't skip this)

ggml `.bin` models (HF `ggerganov/whisper.cpp`) are a **parallel asset family** to the
ONNX encoder/decoder/tokens catalog in `adapters/server/models/assets.{hpp,cpp}` — new
catalog entries, new disk layout, and roughly doubled download for a user who flips
between backends. The model picker / `whisper_model_names()` mapping must resolve to
per-backend assets.

### Risks

- **Text drift vs the sherpa backend.** Mitigated by design: the goldens are already
  decoupled — `whisper_sherpa.hpp:8-11` states Whisper text is not byte-stable across
  decoders and this stage is judged by **WER/CER, not exact text**. The same standard
  applies to whisper.cpp; hallucination/repetition behavior still differs and needs a
  listen-through on the parity clip set.
- **ggml + sherpa-onnx coexistence** in one binary: a new shared dep next to the
  carefully-shared ORT (`CMakeLists.txt:189-197`); symbol collisions unlikely but the
  link must be verified on macOS.
- **Cold start:** Metal shader compile on first run (cached by the OS afterwards) —
  same class of concern as CUDA's cuDNN autotune note in the GPU brief.
- **Per-span serial loop** (same as today): one `whisper_full` per VAD span; unified
  memory makes the per-call overhead much smaller than CUDA's, but it's still serial.

### Optional add-on — `WHISPER_COREML` ANE encoder

whisper.cpp can run the *encoder* on the Apple Neural Engine via a precompiled
`*-encoder.mlmodelc` (README claims ≥3× encoder speedup vs CPU-only). Yet another asset
artifact per model; treat as a Phase 3 experiment after the Metal baseline works.

### Verification spike

Before writing any engine code: build stock `whisper-cli` on the target Mac and
benchmark `ggml-<active-model>.bin` against the sherpa CPU RTF baseline from Phase 0 on
the same clips. If the standalone numbers don't clear the bar, stop here.

---

## Routes considered and rejected

- **ORT WebGPU EP (Metal via Dawn) for stages 2–3:** exists upstream, but ships as a
  separate package, not in the `onnxruntime-osx-arm64` prebuilt sherpa pins — adopting
  it means a custom ORT build replacing sherpa's download, for graphs it is unproven
  on. Reject for now; revisit if sherpa moves to an ORT that bundles it.
- **MLX:** notable because mlx_whisper measured ~2× faster than whisper.cpp Metal on
  large-v3-turbo
  ([benchmark](https://notes.billmill.org/dev_blog/2026/01/updated_my_mlx_whisper_vs._whisper.cpp_benchmark.html)),
  and [mlx-c](https://github.com/ml-explore/mlx-c) is an official C API — but there is
  no maintained first-party C/C++ Whisper implementation on MLX (the reference is
  Python), it would be a third model-asset family, and it's Apple-only build
  complexity. Verdict: **watch, don't build** — the `mlx_available:false` placeholder
  stays honest. *Update:* `MLX_PORT.md` (component study of Lightning-SimulWhisper)
  softens this — MLX is C++-native and the decoder is a bounded ~300-line hand-roll;
  feasible, but justified only by a future *streaming* feature, not this batch roadmap.
- **Custom Metal kernels / native CoreML conversion of wav2vec2 & pyannote:**
  kernel-authoring cost for the two cheapest stages — same logic as the CUDA brief's
  out-of-scope list.

---

## Summary of required changes

| Concern | Location | Route |
|---|---|---|
| `Device::CoreML` + parse/to_string ("coreml") | `adapters/server/config.{hpp,cpp}:23,114,123` | A |
| `coreml_available` probe + status field | `model_manager.cpp:65-67,83-86`, `wav2vec2_onnx.cpp:11-23` (pattern) | A |
| CoreML branch in `make_options()` (modern API, MLProgram, cache dir) | `core/align/wav2vec2_onnx.cpp:36-52` | A (stage 2) |
| Accept `"coreml"` in `/api/device` (falls out of `parse_device`) | `api_controller.hpp:466-500` | A |
| `AsrBackend` enum + `WHISPERX_ASR_BACKEND` + setting + endpoint | `config.{hpp,cpp}`, `api_controller.hpp` | B |
| `WhisperCpp` engine class | new `core/asr/whisper_cpp.{hpp,cpp}` | B |
| Backend branch in `build_asr_engine` (variant/interface for the cache) | `model_manager.cpp:157` | B |
| ggml model asset family | `adapters/server/models/assets.{hpp,cpp}` | B |
| whisper.cpp FetchContent + `WHISPERX_ENABLE_WHISPERCPP`/`WHISPERX_WHISPERCPP_BUILD` | `CMakeLists.txt` (pattern at `:45,181-183,198-201`) | B |
| Real `whispercpp_available` | `model_manager.cpp:86` | B |
| `server-macos` preset | `CMakePresets.json` | all |

**Out of scope (CPU stays):** silero VAD (deliberately pinned, `vad_silero.cpp:36-39`),
ffmpeg decode, trellis Viterbi, emission log-softmax, speaker assignment, clustering,
writers.

---

## Ranked roadmap (the answer)

1. **Phase 0 — free:** `server-macos` preset; build + test on arm64; record CPU RTF
   baselines with `threads_for()` multithreading. No acceleration code at all.
   *Status: speculatively implemented (preset added; never run on a Mac) — see
   `docs/MACOS_COREML.md` for the validation run-book.*
2. **Phase 1 — easy/low-risk, low expected win:** `Device::CoreML` spike. Almost pure
   reuse of the landed CUDA plumbing; timebox to days. The evidence
   (sherpa#2910, ORT dynamic-shape issues, sherpa's legacy flags=0/NeuralNetwork path)
   predicts a wash or regression — ship only if it measurably beats CPU.
   *Status: speculatively implemented, unvalidated — `Device::CoreML`,
   `coreml_available` probe, stage-2 MLProgram branch (+ env knobs), per-stage RTF
   logging in the runner; benchmark procedure in `docs/MACOS_COREML.md`.*
3. **Phase 2 — highest ROI:** whisper.cpp + Metal for Stage 1, gated by the standalone
   `whisper-cli` benchmark spike. This is the recommended investment.
4. **Phase 3 — optional:** `WHISPER_COREML` ANE encoder on top of Phase 2; retune
   Stage 2 batch budget if CoreML stuck anywhere.
5. **Not now:** WebGPU EP, MLX (watch list).

---

## Unknowns requiring an Apple Silicon machine (dev box is Linux)

- Does the `whisperx_server` target build on macOS at all (Phase 0)?
- Node-assignment fraction when the CoreML EP loads each of the five graphs (Whisper
  enc/dec, wav2vec2, pyannote-seg, CAM++) — silent CPU fallback is invisible without
  ORT verbose logs on-device.
- Measured RTFs on the same clip set: CPU-multithread vs CoreML vs whisper.cpp-Metal;
  diarization parity drift under CoreML embeddings.
- CoreML compile-cache behavior across `set_device()` evict-and-rebuild cycles; Metal
  shader-compile cold start for whisper.cpp.
- ggml + sherpa-onnx link coexistence in one macOS binary.
