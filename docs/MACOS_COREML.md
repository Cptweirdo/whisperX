# macOS / CoreML — build & benchmark run-book

Companion to `METAL_INTEGRATION.md`. Phases 0–1 of that brief were written and
unit-tested on Linux. **Build + test + boot on Apple Silicon now confirmed** (arm64,
macOS 15 / Darwin 25.5, 2026-06-11): `whisperx_server` links ORT 1.24.4, boots,
`GET /api/models` reports `coreml_available: true`, and `ctest --preset server-macos`
is **204/204 green**. Still **not** done on a Mac: the CPU RTF baseline and the
CPU-vs-CoreML side-by-side. This is the checklist for that: get the honest CPU
baseline, then run CoreML side-by-side and decide whether the CoreML device is worth
keeping.

Expectation setting up front: the evidence (sherpa-onnx#2910, ORT dynamic-shape
issues #14212/#16934, sherpa's legacy `flags=0`/NeuralNetwork append path) predicts
CoreML will be **a wash or a regression** for these graphs. The spike exists to
replace that prediction with numbers. Ship the device only if it measurably beats CPU.

## What's already wired (no code needed on the Mac)

- `Device::CoreML` — accepted by `WHISPERX_DEVICE`, `POST /api/device`,
  onboarding, and the persisted setting; the string `"coreml"` is passed straight
  through as the sherpa/ORT provider for all three stages.
- `coreml_available` in `GET /api/models` — true iff the linked ORT exposes
  `CoreMLExecutionProvider` (always false off-Apple; the gate returns 400 then).
- Stages 1 & 3 (sherpa): sherpa appends the CoreML EP itself under `__APPLE__` —
  via the **legacy** API, i.e. NeuralNetwork model format, no knobs.
- Stage 2 (wav2vec2, direct ORT): modern provider-options API — **MLProgram**
  format, env-tunable compute units + compile cache (below).
- Per-stage wall-clock logging in the job runner:
  `stage=<name> elapsed=<s> rtf=<elapsed/audio-duration> device=<dev>` — this is
  the benchmark instrument; there is no other timing harness.

## Env knobs

| Env | Default | Meaning |
|---|---|---|
| `WHISPERX_DEVICE` | `cpu` | `cpu` \| `cuda` \| `coreml` (persisted setting wins after first boot) |
| `WHISPERX_COREML_COMPUTE_UNITS` | `CPUAndGPU` | Stage 2 only: `CPUAndGPU` \| `CPUAndNeuralEngine` \| `ALL` \| `CPUOnly` |
| `WHISPERX_COREML_CACHE_DIR` | `<data_dir>/coreml-cache` | Stage 2 only: CoreML compile cache — without it every `set_device()` rebuild recompiles the model |
| `WHISPERX_ORT_VERBOSE` | unset | `1` → stage 2's ORT env logs per-node EP assignments (the only way to see silent CPU fallback) |

Note the asymmetry: the compute-units/cache/verbose knobs reach only stage 2
(wav2vec2), because that's the one session we construct ourselves. Stages 1 & 3 go
through sherpa's internal append — fixed NeuralNetwork format, no cache dir, no
verbose toggle, short of patching sherpa.

## Session checklist

### Phase 0 — build + CPU baseline

1. Deps: `brew install ffmpeg ninja cmake`. `curl`/`libarchive` are keg-only in
   Homebrew, so point CMake/pkg-config at them (CURL otherwise resolves to the Xcode
   SDK tbd, which also works; LibArchive needs the brew prefix):
   ```bash
   export PKG_CONFIG_PATH="/opt/homebrew/opt/curl/lib/pkgconfig:/opt/homebrew/opt/libarchive/lib/pkgconfig:$PKG_CONFIG_PATH"
   export CMAKE_PREFIX_PATH="/opt/homebrew/opt/curl:/opt/homebrew/opt/libarchive:$CMAKE_PREFIX_PATH"
   ```
2. ```bash
   cmake --preset server-macos
   cmake --build --preset server-macos        # target whisperx_server links cleanly
   ctest --preset server-macos                 # NOT yet run on a Mac
   ```
   Confirmed (2026-06-11): the server target builds in **one pass** and boots. The
   one gotcha hit + fixed: `simple-sentencepiece`'s `threadpool.h` uses `std::result_of`
   (gone in C++20) — libc++ hard-errors where libstdc++ tolerated it, and the failure
   masquerades as bogus `<filesystem>` "undeclared identifier 'path'" errors. The
   `CMakeLists.txt` patch that rewrites it to `std::invoke_result_t` now runs right
   after sherpa's `add_subdirectory` (so the file exists), making it a clean one-pass
   build. `ctest --preset server-macos` passes 204/204. One macOS-only ASan quirk
   handled: `detect_container_overflow` fires false positives on the nlohmann::json
   turn-split path because Apple's system libc++ isn't ASan-instrumented (std::vector
   annotations break crossing that boundary). The `server-macos` test preset sets
   `ASAN_OPTIONS=detect_container_overflow=0` for that reason; Linux keeps full
   detection. No real memory bug — confirmed by the rest of ASan/LSan/UBSan staying on.
3. Pick 2–3 benchmark clips (suggest: ~1 min speech, ~10 min multi-speaker, one
   noisy/musical) and keep them fixed for every run.
4. Run each clip through the server on **cpu**. Collect the `stage=… rtf=…` lines
   from the server log (`<data_dir>/logs/`). Run each clip twice; keep the second
   (warm) numbers. Record machine, model size, clip durations.
   This baseline is multi-threaded NEON/MLAS CPU — the bar CoreML must clear.

### Phase 1 — CoreML side-by-side — RUN 2026-06-11, **verdict: non-functional**

The side-by-side never produced a comparison: **CoreML cannot complete a transcription
on this stack** (M4, sherpa-onnx v1.13.2, ORT 1.24.4, `large-v3-turbo` exports). Stage 1
(Whisper) fails in all three precision variants, each differently:

| `WHISPERX_ASR_PRECISION` | Encoder asset | CoreML outcome |
|---|---|---|
| `fp16` (the device's auto-default — see bug below) | `turbo-encoder.fp16.onnx` | **load fail** — `graph_utils.cc:30 GetIndexFromName … InsertedPrecisionFreeCast_…/attn_ln/… SimplifiedLayerNormFusion`: ORT's layernorm fusion collides with the fp16 converter's precision-cast nodes under the CoreML EP. |
| `fp32` | `turbo-encoder.onnx` | **load fail** — `initializer.cc:45 … model_path must not be empty`: CoreML partitioning re-serializes the subgraph and loses the external-weights file path. |
| `int8` | `turbo-encoder.int8.onnx` | **loads, then the process crashes mid-`transcribing`** — `Context leak detected, CoreAnalytics returned false`, hard abort (CoreML has no int8 kernels → CPU-fallback path that dies). |

Because Stage 1 can't run, Stage 2 (wav2vec2 on the MLProgram path we control) and the
node-placement/compute-unit/cache experiments (steps 7–10 below) were **never reached** —
the device knob is global, so there's no way to put align on CoreML while Whisper stays
on CPU. Those steps stay open *iff* Stage 1 is ever made to load.

**Code bug surfaced:** `models/model_manager.cpp:183` picks precision as
`dev == Cpu ? Int8 : asr_precision_`, lumping CoreML in with CUDA — so CoreML inherits
the **fp16** default that was tuned for CUDA. On CoreML fp16 doesn't just regress, it
fails to load. Even fixing the default doesn't rescue CoreML (all three variants fail),
but the device shouldn't auto-select a variant that hard-crashes.

Steps 7–10 (kept for if/when Stage 1 loads): node placement via `WHISPERX_ORT_VERBOSE=1`
on the wav2vec2 graph; `WHISPERX_COREML_COMPUTE_UNITS=ALL`/`CPUAndNeuralEngine` variants;
transcript/diarization parity; compile-cache behavior across cpu↔coreml switches.

### Decision — taken 2026-06-11

CoreML is **worse than the predicted wash — it's broken end-to-end**. Recommendation:
**do not expose the CoreML device** until the Whisper-asset/EP issues are resolved
(re-export without the offending fusions / with inline weights, or patch sherpa's EP
append). The plumbing stays (inert, gated by `coreml_available` + the precision bug
note). **Move to Phase 2 — whisper.cpp + Metal for Stage 1**, the route with the real
ROI and no dependence on the ORT CoreML EP. This corroborates `METAL_INTEGRATION.md`
Route A's low-ROI framing, only more strongly than expected.

## Recorded CPU baseline (Phase 0 step 4)

Apple **M4** (4P+6E, 10-core), 16 GB, macOS 15 / Darwin 25.5; `large-v3-turbo` **int8**
(the CPU default precision), `threads_for(cpu)=5`. Clip: `samples/russian_2_speaker_trim.m4a`
(300 s, mono, 2 speakers, Russian, `language=ru`). Warm run (align model resident):

| Stage | elapsed | RTF | share |
|---|---|---|---|
| decoding (ffmpeg) | 0.14 s | ~0.000 | — |
| **transcribing (Stage 1 ASR)** | **70.85 s** | **0.236** | **65%** |
| aligning (Stage 2 wav2vec2) | 21.33 s | 0.071 | 20% |
| diarizing (Stage 3, CPU-pinned) | 17.00 s | 0.057 | 16% |
| **end-to-end** | **~109 s** | **~0.364** | 100% |

This is the bar. **Stage 1 is 65% of wall-clock** → confirms the brief's premise that
whisper.cpp+Metal (Route B, Stage 1 only) is the real ROI lever; CoreML on stages 2–3
can only chip at the remaining ~36%, and Stage 3 is CPU-pinned anyway. Cold run (first
of the session, before align resident) was within noise on the compute stages — only
`loading_align` differs (18 s cold vs 0 s warm), which is one-time per process.

## Phase 2b — quant + flash-attn on the whisper.cpp/Metal backend (Phase 2 follow-up)

Run 2026-06-11 (M4), same clip. With Route B landed, the next Apple lever
(`SPEEDUP_FINDINGS.md` lever 1) was a pure measurement — both knobs already ship. Swept
`WHISPERX_GGML_QUANT` × `WHISPERX_WHISPERCPP_FLASH_ATTN` via
`scripts/bench_whispercpp_metal.sh`; WER is drift vs the fp16/flash-off transcript
(`scripts/bench_whispercpp_wer.py`, gate ≤ 0.03). Stage 1 only — align/diarize unchanged.

| quant | flash | Stage-1 RTF | ×fp16 | WER drift | verdict |
|---|---|---|---|---|---|
| fp16 | off | 0.063 | 1.00× | 0.000 | baseline |
| fp16 | **on** | 0.055 | 1.15× | 0.013 | flash is the lever |
| q8_0 | off | 0.062 | 1.02× | 0.007 | quant alone ≈ nothing |
| **q8_0** | **on** | **0.050** | **1.26×** | **0.011** | **recommended** |
| q5_0 | off | 0.057 | 1.11× | 0.046 | over gate — reject |
| q5_0 | on | 0.050 | 1.26× | 0.062 | over gate — reject |

**Finding:** flash-attention does the work; q8_0 quantization barely speeds Stage 1 alone
(turbo's 4-layer decoder makes the bandwidth-bound decode a small slice — the
lightning-whisper-mlx "quant = decode throughput" thesis didn't hold here). q8_0 stacks a
small near-lossless bonus on flash; q5_0 garbles Russian (drops/swaps words) and breaks
the WER gate. **Recommended Mac flags: `WHISPERX_WHISPERCPP_FLASH_ATTN=1` +
`WHISPERX_GGML_QUANT=q8_0`** (1.26× Stage 1, ≈4× over the int8 CPU baseline above), or
fp16+flash to skip the 874 MB q8_0 download for ~9% less Stage-1 speed. E2E gain is only
~5% — Stage 1 is now 37%, so the real time is in align+diarize (levers 2–3).

Env knobs (whisper.cpp backend, read at engine-build → set before boot):

| Env | Default (Apple / other) | Meaning |
|---|---|---|
| `WHISPERX_ASR_BACKEND` | `whispercpp` / `sherpa` | `sherpa` \| `whispercpp` (Metal Stage 1) |
| `WHISPERX_GGML_QUANT` | `q8_0` / *(empty = fp16)* | `q8_0` (recommended) \| `q5_0` (rejected, WER) — picks `ggml-<model>[-quant].bin` from `ggerganov/whisper.cpp` |
| `WHISPERX_WHISPERCPP_FLASH_ATTN` | `1` / off | `1` enables flash-attention — the actual speed lever |

**Now the Apple default (2026-06-11, `config.cpp::load_config`):** on `__APPLE__`,
`WHISPERX_ASR_BACKEND` / `WHISPERX_GGML_QUANT` / `WHISPERX_WHISPERCPP_FLASH_ATTN` default
to `whispercpp` / `q8_0` / `1` — so a fresh Mac install runs Stage 1 on whisper.cpp/Metal
with the measured-best flags out of the box. It degrades safely to sherpa if the build
lacks `WHISPERX_WHISPERCPP_BUILD` (`asr_backend_available` gate), and any explicit env var
or persisted `/api/asr_backend` choice still wins. Non-Apple defaults are unchanged
(sherpa, fp16, no flash).

## What to bring back (fills METAL_INTEGRATION.md "Unknowns")

- ~~Did `server-macos` build/test cleanly, and what needed fixing?~~ **Done**: builds
  one-pass, boots, `ctest` 204/204. Needed: keg-only brew env (step 1), the threadpool
  C++20 patch reorder, and `ASAN_OPTIONS=detect_container_overflow=0` in the macOS test
  preset (libc++ ASan false positive).
- ~~CPU RTF baseline~~ **Done** (table above): E2E RTF 0.364 on M4, Stage 1 dominates.
- Per-stage RTF table: **coreml** vs the cpu baseline (× compute-unit variants for stage 2).
- wav2vec2 node-assignment fraction under CoreML.
- Diarization parity verdict.
- Compile-cache behavior across device switches.
