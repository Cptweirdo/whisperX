# macOS / CoreML — build & benchmark run-book

Companion to `METAL_INTEGRATION.md`. Phases 0–1 of that brief are **speculatively
implemented** on this branch (written and unit-tested on Linux, never executed on a
Mac). This is the checklist for the first Apple Silicon session: get a build, get an
honest CPU baseline, then run the CPU-vs-CoreML side-by-side and decide whether the
CoreML device is worth keeping.

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

1. Deps: `brew install ffmpeg ninja cmake` (curl/libarchive come with macOS; if
   CMake can't find them, `brew install curl libarchive`).
2. ```bash
   cmake --preset server-macos
   cmake --build --preset server-macos
   ctest --preset server-macos
   ```
   This is the **first ever macOS build of this tree** — expect small breakage
   (linker flags, keychain backend, case-sensitive includes) before anything runs.
3. Pick 2–3 benchmark clips (suggest: ~1 min speech, ~10 min multi-speaker, one
   noisy/musical) and keep them fixed for every run.
4. Run each clip through the server on **cpu**. Collect the `stage=… rtf=…` lines
   from the server log (`<data_dir>/logs/`). Run each clip twice; keep the second
   (warm) numbers. Record machine, model size, clip durations.
   This baseline is multi-threaded NEON/MLAS CPU — the bar CoreML must clear.

### Phase 1 — CoreML side-by-side

5. Switch: `POST /api/device {"device":"coreml"}` (or restart with
   `WHISPERX_DEVICE=coreml` on a fresh data dir). First load pays CoreML model
   compile — note it, then ignore it (warm runs only).
6. Same clips, same procedure. Record per-stage RTFs next to the CPU numbers.
7. Node placement: rerun one clip with `WHISPERX_ORT_VERBOSE=1` and count
   CoreML-vs-CPU node assignments for the wav2vec2 graph in the log. Mostly-CPU
   placement explains a wash instantly. (Stages 1 & 3 can't be inspected this way —
   infer from their RTF deltas.)
8. Stage 2 variants (cheap, env-only): repeat the align-heavy clip with
   `WHISPERX_COREML_COMPUTE_UNITS=ALL` and `CPUAndNeuralEngine`.
9. Parity spot-checks: transcripts should be unchanged (whisper text identical;
   word timestamps within tolerance); diarization may drift — same caveat as CUDA
   (embedding drift can flip cluster assignments). Listen-through one
   multi-speaker clip.
10. Cache behavior: flip device cpu→coreml→cpu→coreml; the second coreml switch
    should be fast (compile cache hit in `<data_dir>/coreml-cache`). Stages 1 & 3
    have no cache dir — if switches are slow, that's why.

### Decision

- **CoreML ≥ CPU on stage RTFs** → keep the device selectable, document the win.
- **Wash or regression (predicted)** → record the numbers in `METAL_INTEGRATION.md`,
  leave the plumbing in (it's inert and gated), and move to Phase 2 — whisper.cpp +
  Metal for stage 1, the route with the real ROI.

## What to bring back (fills METAL_INTEGRATION.md "Unknowns")

- Did `server-macos` build/test cleanly, and what needed fixing?
- Per-stage RTF table: cpu vs coreml (× compute-unit variants for stage 2).
- wav2vec2 node-assignment fraction under CoreML.
- Diarization parity verdict.
- Compile-cache behavior across device switches.
