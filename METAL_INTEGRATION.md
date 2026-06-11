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
   the landed `threads_for()` (`models/model_manager.cpp:33`, half the cores) on NEON/MLAS
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
  └─ Stage 3  Diarize    segmentation+embed    sherpa-onnx   RTF ~0.15   ← Route A moot*
```

\* Since this brief was first drafted, Stage 3 was **hard-pinned to the CPU provider
regardless of `device_`** (`models/model_manager.cpp:278-289`, a CUDA-era finding:
hundreds of tiny forwards starve the GPU — measured `cuda+8thr` 737s vs `cpu+8thr` 98s
on a 23-min file, see `SPEEDUP_FINDINGS.md` §1). So `Device::CoreML` never reaches the
diarizer today; the only diarize-on-Metal lever is the conv-heavy **embedding extractor**
on the ANE (a clean cheap measurement — `SPEEDUP_FINDINGS.md` §3), which would require
un-pinning that one model, not the whole stage.

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
2. **Honest CPU baseline.** Record RTF medians (`jobs/runner.cpp::stage_rtf()`) on the
   target Mac with the CPU build *before* any Metal work. Every later claim is measured
   against this, not against historical single-thread numbers.
3. **Two config dimensions, not one.**
   - Route A extends the existing enum: `Device { Cpu, Cuda, CoreML }`
     (`adapters/server/config.hpp:25` — landed) plus `parse_device`/`to_string` cases
     (`config.cpp:125,132`; `to_string` returning `"coreml"` *is* the sherpa provider
     string). Everything downstream — `ModelManager::set_device()` with rollback, the
     provider threading at `models/model_manager.cpp:200-202` (ASR) and `:328` (align),
     `POST /api/device` validation + 409 busy gate
     (`http/api_controller.hpp:466-487`), the persisted setting, `WHISPERX_DEVICE` —
     is reused unchanged. (Note Stage 3 diarize at `:286-287` is *not* device-threaded —
     it hard-codes `provider="cpu"`; see the diagram footnote.)
   - Route B is **not a Device**. whisper.cpp replaces the Stage 1 *engine*, while
     stages 2–3 stay on ORT and still need their own cpu/coreml provider choice.
     Conflating it into the Device enum (e.g. `Device::Metal`) would break that. Add an
     orthogonal `enum class AsrBackend { Sherpa, WhisperCpp }` + `WHISPERX_ASR_BACKEND`
     env + persisted setting, surfaced either as a new `POST /api/asr_backend` or an
     extended `/api/device` payload.
4. **Honest status.** `status()` emits `{"mlx_available", false}` and
   `{"whispercpp_available", false}` (`models/model_manager.cpp:106-107`) next to the
   real `cuda_available()` probe. `coreml_available` is **landed**
   (`:76-77,105` → `wal::ort_coreml_available()`, which probes
   `Ort::GetAvailableProviders()` for `"CoreMLExecutionProvider"` under `#ifdef
   __APPLE__` — `core/align/wav2vec2_onnx.cpp:28-39`). Still to do (Route B): flip
   `whispercpp_available` to a compile-time `WHISPERX_WHISPERCPP_BUILD` define +
   asset-presence check.
5. **CoreML compile cache.** The CoreML EP recompiles the model at session creation
   unless `ModelCacheDirectory` is set, which "may cost significant time" per the ORT
   docs. This interacts badly with `set_device()`'s evict-and-rebuild semantics: every
   cpu→coreml switch pays the compile for every resident engine unless the cache dir is
   configured (and sherpa's append path exposes no way to set it — see Route A risks).

---

## Route A — CoreML EP via the existing Device plumbing (all 3 stages)

**The cheap spike. Cost: ~1 day of plumbing. Risk: low (worst case it's slower and we
don't ship it). Expected ROI: low, possibly negative.**

> **MEASURED 2026-06-11 (Apple M4): worse than predicted — non-functional.** CoreML
> cannot complete a transcription. The Stage 1 Whisper model fails to even load under
> the CoreML EP in fp16 (ORT `SimplifiedLayerNormFusion` vs precision-cast crash) and
> fp32 (external-weights `model_path` lost in CoreML partitioning), and the int8 export
> loads but **crashes the process mid-decode** (`Context leak detected`). Stage 2/3
> CoreML were unreachable (global device knob, Stage 1 blocks the pipeline). Verdict:
> do **not** expose the CoreML device; go Route B. Full failure table + the
> `model_manager.cpp:183` precision-default bug in `docs/MACOS_COREML.md` Phase 1.

### How it would be implemented

- **Stages 1 & 3 (sherpa-engined): zero engine changes.** sherpa v1.13.2's
  [`provider.cc`](https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v1.13.2/sherpa-onnx/csrc/provider.cc)
  maps the string `"coreml"` → `Provider::kCoreML`, and
  [`session.cc`](https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v1.13.2/sherpa-onnx/csrc/session.cc)
  appends the CoreML EP under `__APPLE__`. `WhisperSherpa` and `SherpaDiarizer`
  already take the provider string (`whisper_sherpa.hpp:50-54`,
  `diarize_sherpa.hpp:40-43`); `to_string(Device::CoreML) == "coreml"` does the rest.
- **Stage 2 (direct `Ort::Session`):** **landed** — `make_options()`
  (`core/align/wav2vec2_onnx.cpp:53-86`) has a CoreML branch under `#ifdef __APPLE__`
  (line 66) next to the existing `#ifdef WHISPERX_GPU_BUILD` CUDA branch, using the
  **modern** options API: `opts.AppendExecutionProvider("CoreML", {{"ModelFormat",
  "MLProgram"}, …, {"ModelCacheDirectory", …}})` with the cache dir env-driven.
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

- **Stage 1 (Whisper):** the encoder is fixed-shape (30 s mel) so it had the best shot
  at full CoreML placement; the autoregressive decoder is dynamic → likely partial
  placement and ping-pong. **Measured (M4, 2026-06-11): never got that far — the model
  won't load (fp16/fp32) or crashes the process at decode (int8) under the CoreML EP.
  Dead end with the current turbo exports.**
- **Stage 2 (wav2vec2):** we control the session options (MLProgram, static-shape
  experiments possible) but batched dynamic shapes are the EP's worst case, and this
  stage's batching is the thing that makes it fast.
- **Stage 3 (diarize):** **currently CPU-pinned** (`models/model_manager.cpp:286-287`,
  `provider="cpu"` hard-coded) — the device knob doesn't reach it, by a deliberate
  CUDA-era decision (`SPEEDUP_FINDINGS.md` §1). Two small models + CPU clustering →
  marginal even if placement were clean. The one credible Metal angle is moving *only*
  the conv-heavy **embedding extractor** onto the ANE/CoreML (it maps far better than
  transformers; `SPEEDUP_FINDINGS.md` §3) — a cheap targeted measurement, not the whole
  stage. Also re-check diarization parity: accelerator drift in embeddings can nudge
  cluster assignments (same caveat as the CUDA brief).

### Verification spike

On an Apple Silicon machine: run the RTF harness (`jobs/runner.cpp::stage_rtf()` medians)
with `WHISPERX_DEVICE=coreml` vs `cpu` on the same clips, and read ORT verbose logs for
**node-assignment counts** per graph — silent CPU fallback is invisible otherwise.
Timebox; keep only if measurably ≥ CPU.

---

## Route B — whisper.cpp + GGML Metal for Stage 1 (the recommended investment)

**Cost: medium (new engine class + asset family + CMake). Risk: medium (new dependency,
output drift). Expected ROI: high — the only credible 2×+ lever on the dominant stage.**

> **LANDED + MEASURED 2026-06-11 (Apple M4). The win is real.** whisper.cpp v1.8.6 +
> GGML Metal is wired as a second ASR backend (`AsrBackend::WhisperCpp` /
> `WHISPERX_ASR_BACKEND=whispercpp`, `WHISPERX_ENABLE_WHISPERCPP` build). Warm run, 300 s
> Russian clip, `large-v3-turbo` fp16 ggml:
>
> | Stage | sherpa CPU (int8) | whisper.cpp/Metal (fp16) | speedup |
> |---|---|---|---|
> | transcribing (Stage 1) | 70.85 s · RTF 0.236 | **21.9 s · RTF 0.073** | **3.2×** |
> | aligning (Stage 2, ORT CPU) | 21.3 s | 20.9 s | — |
> | diarizing (Stage 3, CPU-pinned) | 17.0 s | 16.9 s | — |
> | **end-to-end** | 109 s · RTF 0.364 | **59.8 s · RTF 0.199** | **1.83×** |
>
> Stage 1 drops from 65% → ~37% of wall-clock (align is now the largest stage). Transcript
> parity sane (coherent Russian, correct 2-speaker diarization, `ru` auto-detected). ggml +
> sherpa-onnx ORT coexist cleanly in one macOS binary (the link risk is cleared). The
> implementation below matches what shipped.

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
    exactly what the transcribe closure in `jobs/runner.cpp` already produces — to
    `whisper_full()` with greedy params; build `AsrChunk{text, avg_logprob}` from
    segments + `whisper_token_data.p`. Stage 2 re-times every word, so whisper.cpp's
    segment timestamps only need to be approximately right (no DTW/token-timestamps
    needed).
  - `detect_language(AudioBuffer)` (`whisper_sherpa.hpp:73`): `whisper_lang_auto_detect`
    over the first 30 s.
- New concrete `core/asr/whisper_cpp.{hpp,cpp}` mirroring those method shapes (reuse
  `AsrChunk` + `strip_blank_audio` from `whisper_sherpa.hpp`), constructed via
  `whisper_init_from_file_with_params` (`use_gpu=true`, optionally `flash_attn`).
- Branch in `ModelManager::build_asr_engine()` (`models/model_manager.cpp:178`) selected
  by the new `AsrBackend`; the runner's `Steps.transcribe` closure shape is unchanged.
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

Status legend: ✅ landed · ⬜ to do.

| Concern | Location | Route | Status |
|---|---|---|---|
| `Device::CoreML` + parse/to_string ("coreml") | `adapters/server/config.hpp:25`, `config.cpp:125,132` | A | ✅ |
| `coreml_available` probe + status field | `models/model_manager.cpp:76-77,105`, `core/align/wav2vec2_onnx.cpp:28-39` | A | ✅ |
| CoreML branch in `make_options()` (modern API, MLProgram, cache dir) | `core/align/wav2vec2_onnx.cpp:53-86` | A (stage 2) | ✅ |
| Accept `"coreml"` in `/api/device` (falls out of `parse_device`) | `http/api_controller.hpp:466-487` | A | ✅ |
| Per-stage RTF logging in the runner | `adapters/server/jobs/runner.cpp:49,144` | A | ✅ |
| Diarize stays CPU (device knob doesn't reach Stage 3) | `models/model_manager.cpp:286-287` | A | ✅ (by design) |
| `AsrBackend` enum + `WHISPERX_ASR_BACKEND` + setting + `/api/asr_backend` | `config.{hpp,cpp}`, `http/api_controller.hpp` | B | ✅ |
| `AsrEngine` interface (sherpa + whisper.cpp behind one handle) | new `core/asr/asr_engine.hpp` | B | ✅ |
| `WhisperCpp` engine class | new `core/asr/whisper_cpp.{hpp,cpp}` | B | ✅ |
| Backend branch in `build_asr_engine` + `set_asr_backend` switch | `models/model_manager.cpp` | B | ✅ |
| ggml model asset family (resolver + downloader, ggerganov repo) | `models/assets.{hpp,cpp}`, `assets/downloader.{hpp,cpp}` | B | ✅ |
| whisper.cpp FetchContent (v1.8.6) + `WHISPERX_ENABLE_WHISPERCPP`/`WHISPERX_WHISPERCPP_BUILD` | `CMakeLists.txt` | B | ✅ |
| Real `whispercpp_available` (compile-time probe) | `models/model_manager.cpp` | B | ✅ |
| `server-macos` preset (whisper.cpp ON) | `CMakePresets.json` | all | ✅ (builds + `ctest` 206/206 on M4) |

**Out of scope (CPU stays):** silero VAD (deliberately pinned, `vad_silero.cpp:36-39`),
ffmpeg decode, trellis Viterbi, emission log-softmax, speaker assignment, clustering,
writers.

---

## Ranked roadmap (the answer)

1. **Phase 0 — free:** `server-macos` preset; build + test on arm64; record CPU RTF
   baselines with `threads_for()` multithreading. No acceleration code at all.
   *Status: **DONE on Apple M4 (2026-06-11)** — builds one-pass, `ctest` 204/204, CPU
   E2E RTF 0.364 (Stage 1 = 65%). Run-book + numbers in `docs/MACOS_COREML.md`.*
2. **Phase 1 — easy/low-risk, low expected win:** `Device::CoreML` spike. Evidence
   (sherpa#2910, ORT dynamic-shape issues, legacy flags=0/NeuralNetwork path) predicted
   a wash or regression.
   *Status: **DONE / killed (2026-06-11)** — CoreML is **non-functional**: the Stage 1
   Whisper model won't load under the CoreML EP (fp16 fusion crash, fp32 external-weights
   path loss) and the int8 export crashes the process mid-decode. Worse than a wash. Do
   not expose the device; the plumbing stays inert. Details in `docs/MACOS_COREML.md`.*
3. **Phase 2 — highest ROI:** whisper.cpp + Metal for Stage 1.
   *Status: **DONE / measured (2026-06-11)** — whisper.cpp v1.8.6 backend landed
   (`AsrBackend::WhisperCpp`); Stage 1 **3.2×** faster, E2E **1.83×** (M4, see Route B
   box). The only viable Apple-GPU route — and it delivers. `ctest` 206/206.*
4. **Phase 3 — optional:** `WHISPER_COREML` ANE encoder on top of Phase 2; retune
   Stage 2 batch budget if CoreML stuck anywhere.
5. **Not now:** WebGPU EP, MLX (watch list).

---

## Unknowns requiring an Apple Silicon machine

*Resolved 2026-06-11 on an Apple M4 (build, Phase-0 baseline, Route A killed, Route B
landed). Full run-book + numbers in `docs/MACOS_COREML.md`.*

- ~~Does the `whisperx_server` target build on macOS at all (Phase 0)?~~ **Yes** —
  builds one-pass + `ctest` 206/206 on M4 (needed two toolchain fixes: a C++20
  `std::result_of` patch reorder and a macOS-libc++ ASan `detect_container_overflow=0`).
- ~~CPU-multithread RTF baseline~~ **Done**: E2E RTF **0.364** on M4 (`large-v3-turbo`
  int8, 300 s clip) — Stage 1 ASR is 65% of wall-clock. This is the bar.
- ~~whisper.cpp-Metal RTF vs the CPU baseline~~ **Done**: Stage 1 RTF **0.073** (3.2×),
  E2E **0.199** (1.83×) — see the Route B box.
- ~~ggml + sherpa-onnx link coexistence in one macOS binary~~ **Done**: links + runs
  clean; ggml's Metal backend and sherpa's ORT share no symbols.
- CoreML EP node-assignment fractions / diarization parity drift — **moot** (Route A is
  non-functional; not worth the on-device dig). The diarize-embedding-on-ANE idea
  (`SPEEDUP_FINDINGS.md` §3) remains the only open CoreML thread, and it's optional.
- CoreML compile-cache behavior across `set_device()` evict-and-rebuild cycles; Metal
  shader-compile cold start for whisper.cpp.
- ggml + sherpa-onnx link coexistence in one macOS binary.
