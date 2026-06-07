# C++ Engine Core + SPA — Architecture & Pipeline Streamlining

> Architecture for rewriting the WhisperX pipeline as a **headless C++ engine
> core** with thin adapters — an HTTP/SSE server behind a web **SPA** (desktop +
> cloud) and **FFI** for mobile/Flutter. Builds on
> [`pipeline-reference.md`](./pipeline-reference.md) (the stage spec),
> [`single-language-runtime-options.md`](./single-language-runtime-options.md)
> (why C++ is the native substrate), and the
> [Flutter plan](./flutter-migration.md) (the alternative delivery).

## 1. Context & goal

WhisperX today is a Python library + Flask web app bound to PyTorch / CTranslate2 /
pyannote / ffmpeg — heavy, and impossible to ship lean. As
[`single-language-runtime-options.md`](./single-language-runtime-options.md) found,
**C++ is the native substrate for every dependency**: the inference runtimes
(ONNX Runtime, sherpa-onnx, GGML) *are* C++. So a C++ rewrite isn't "reimplement
the models" — sherpa-onnx/ORT already provide them — it's **owning the
orchestration + alignment + glue in C++ and exposing the result through thin
adapters.**

**Principles**
- One **headless engine core** (no UI, no transport) — the reusable artifact.
- **Adapters**, not a monolith: HTTP/SSE server (→ SPA, desktop + cloud), FFI
  (→ mobile/Flutter), pybind11 (→ keep Python `app/` + `tests/` as the oracle).
- **All four models on one runtime (ONNX Runtime)** — collapse Python's four ML
  stacks into one.
- The Python pipeline remains the **spec and parity oracle**, not a dependency.

## 2. The engine core + adapters

```
                ┌──────────────────────────────────────────┐
                │      C++ ENGINE CORE  (libwhisperx)        │
                │  decode · VAD · ASR · align · diarize ·    │
                │  assemble · write   — all on ONNX Runtime  │
                └──────────────────────────────────────────┘
                  ▲              ▲                ▲
   ┌──────────────┘   ┌──────────┘     ┌──────────┘
   │ HTTP/SSE server  │ FFI (JNI /     │ pybind11
   │ (Crow/Drogon/…)  │  Swift interop)│ (reference/tests)
   ▼                  ▼                ▼
 Web SPA           Mobile UI         Python app/ + pytest
 (desktop+cloud)   (native/Flutter)  (golden-vector oracle)
```

The core knows nothing about HTTP, FFI, or UI. Each adapter is a thin shell. The
result `schema.py` shapes become the **API/FFI contract**.

## 3. Two delivery models (the desktop/cloud-vs-mobile split)

"C++ backend + SPA" is the **desktop/cloud** delivery. Mobile reuses the *same
core* but with a different adapter — because an embedded localhost HTTP server is
awkward on mobile.

| | Server + SPA | FFI + UI |
|---|---|---|
| Targets | Windows/macOS/Linux desktop, cloud/self-host | Android, iOS |
| Transport | HTTP + SSE (progress) | direct function calls (JNI / Swift) |
| UI | web SPA in a webview (Tauri/CEF/system) or browser | native, Flutter, or WebView |
| Fit | ✅ excellent | ✅ the right mobile path |
| Anti-pattern | — | ❌ don't run a localhost HTTP server in-app; ❌ remote server breaks offline |

**One core, two front doors.** Desktop bundles get lean (native binary + static SPA
assets) — no `python-build-standalone`, no torch/CUDA, no cuDNN matching, so the
bulk of [`windows-port-options.md`](./windows-port-options.md) /
[`MACOS_INSTALLER.md`](../MACOS_INSTALLER.md) packaging pain disappears. The
existing Tauri shell (`packaging/macos/tauri`) just points at the C++ server
instead of Flask.

## 4. Pipeline streamlining

A faithful port would inherit structure that exists only because of Python's deps.
The C++ core lets us drop most of it. Items map to the current code in
[`pipeline-reference.md`](./pipeline-reference.md).

### A. Eliminate redundant work
- **Decode once.** Today the file is decoded up to **3×** — ASR (`transcribe.py:148`),
  alignment for multi-file (`alignment.py:135`), and diarization **always**
  (`diarize.py:131-132`). Decode once → one float32 16 kHz mono buffer → pass
  **zero-copy spans** to every stage. Only Whisper needs mel; VAD/wav2vec2/diarize
  all want the same raw buffer.
- **One segmentation, not two.** The audio is segmented twice — the WhisperX VAD for
  ASR batching (`vads/`) *and* pyannote's internal segmentation in diarization. When
  diarization is on, derive ASR-batching windows from its speech regions and skip
  the standalone VAD model; when off, run a light silero pass. *(Caveat:
  pyannote-seg ≠ silero, so the sharing flows diarization→ASR, not the reverse.)*
- **Models resident** + *(optional)* **stage pipelining.** Python loads→runs→frees
  each model sequentially for memory (`transcribe.py:124-234`). A server keeps all
  models loaded (**adopted** — no reload churn; also makes sleep/resume instant, see
  the memory decision below). **Overlapping stages** (align chunk N-1 while ASR runs
  chunk N) is a further throughput option but is **deferred for simplicity** — the
  committed model is one job at a time, run to completion (see
  [`cpp-core-migration-briefs.md`](./cpp-core-migration-briefs.md) §
  "Memory management"). Revisit pipelining only if single-job latency becomes a
  bottleneck.

### B. Collapse four ML stacks into one runtime
Python uses **four** inference stacks: CTranslate2 (Whisper), torch/HF (wav2vec2),
pyannote/torch (diarization), torch.hub (silero). Run **all four on ONNX Runtime**
(sherpa-onnx already does 3/4; add wav2vec2 as an ORT model). One runtime → one
threading model, one memory arena, one set of EPs (Core ML/NNAPI/CUDA/DirectML).

This also **deletes heavy transitive deps**:

| Drop | Replace with |
|---|---|
| `pandas` (alignment `:325,395`, diarize `:170`) | plain structs + loops |
| `nltk` punkt (`alignment.py:189`) | ICU / small punctuation-rule splitter |
| `torch`, `torchaudio`, `transformers`, `faster-whisper`, `ctranslate2`, `pyannote-audio` | ORT + sherpa-onnx |

Dependency graph shrinks from dozens of packages to ≈ **ORT + audio decoder + your code**.

### C. Make alignment efficient (the un-optimized stage)
Alignment carries an explicit `TODO: ...batched inference here` (`alignment.py:245`)
and is per-segment, sequential, un-batched.
- **Batch the wav2vec2 forward passes** across segments (pad to common length, one
  ORT batch) instead of one call per segment.
- **Compute emissions over larger contiguous spans once and slice per segment**,
  removing per-call overhead and the <400-sample re-padding (`alignment.py:248`).
- The pandas char→word→sentence assembly (`alignment.py:298-411`) becomes tight
  struct loops — no per-segment DataFrame allocation.

### D. Reduce model count where quality allows
- **Default to silero VAD; drop the heavier pyannote VAD model.**
- **Optional: drop wav2vec2 alignment** if Whisper-native timestamps suffice — a
  config switch, not a default (wav2vec2 accuracy is *why* WhisperX exists).

### E. Keep these — already good
- `IntervalTree` + `assign_word_speakers` (`diarize.py:14,185`) — O(log n), ~228×
  over linear scan. Port as-is.
- Viterbi `get_trellis`/`backtrack` (`alignment.py:425-490`) — algorithm is right;
  just batch the *emissions* feeding it.
- Wildcard/OOV char handling (`alignment.py:272-283`) — clean.

### Streamlined shape
```
decode once → raw 16k buffer (shared, zero-copy spans)
   ├─ [if diarize] segmentation → speech regions ─┐ (feeds both)
   ├─ [else] silero VAD ──────────────────────────┤
   │                                              ▼
   │                              ASR windows (≤30s, packed)
   │                                  └─ Whisper (ORT) ─→ text segments
   ├─ wav2vec2 emissions over spans (ORT, batched) ─→ Viterbi ─→ word times
   └─ diarization embeddings (ORT) ─→ turns ─→ IntervalTree assign
                                                     ▼
                                        assemble result · write
```
One runtime · one decode · models resident · stages pipelined · glue in tight C++
— versus four ML stacks, up-to-3× decode, sequential load/unload, pandas/nltk today.

## 5. Tech choices

| Concern | Options | Note |
|---|---|---|
| Inference | **ONNX Runtime** (C++) + **sherpa-onnx** (VAD/ASR/diarize) | one runtime; EPs for GPU/ANE/NNAPI |
| wav2vec2 align | ORT model + own Viterbi | the one DIY model |
| Audio decode | dr_libs / miniaudio / libsndfile (+ ffmpeg libs for compressed) | **no ffmpeg subprocess**; link libs, decode to float in-memory |
| HTTP/SSE server | **Drogon** or **oat++** (perf, WebSocket/SSE) · **cpp-httplib** (header-only, simple) | SSE for progress, mirroring `app/` |
| SPA | React / Svelte / SolidJS | true client-rendered SPA over a JSON/SSE API |
| Build | CMake + vcpkg/Conan; cross-compile per platform | the main ergonomic cost |
| Mobile adapter | JNI (Android) · Swift C-interop (iOS) · or via Flutter FFI | reuses the core, no server |
| Python oracle | pybind11 | keep `app/` + `tests/` as parity reference |

> **Note on the SPA:** the current `app/` frontend is **htmx + SSE (server-rendered),
> not a true SPA** (see CLAUDE.md htmx/Shoelace notes). "Move to an SPA" is therefore
> also a frontend rewrite (a JS framework + build) that decouples UI from the backend
> language — a second workstream, not free.

## 6. Proposed structure

```
core/                         # libwhisperx (C++) — no transport/UI
  audio/      decode, resample, mel
  vad/        silero + chunk-merge (ported from vads/)
  asr/        sherpa/ORT Whisper wrapper
  align/      wav2vec2 ORT + trellis/backtrack/words (ported)
  diarize/    sherpa diarization + interval-tree assign (ported)
  schema/     result structs (mirror schema.py)
  pipeline/   orchestrator (mirror transcribe_task), progress events
adapters/
  server/     HTTP/SSE (Drogon/oatpp) → SPA
  ffi/        C ABI for JNI/Swift/Flutter
  py/         pybind11 module (oracle)
web/          SPA (React/Svelte)
bindings/test golden vectors + parity tests
```

## 7. Golden-parity strategy (unchanged)

Same discipline as the Flutter plan: dump intermediates from Python `whisperx`
(VAD/merged chunks, CTC emissions/trellis, backtrack path, word timings,
diarization turns, speaker labels) into golden JSON; assert the C++ core matches
within tolerance (`merge_chunks` boundaries, trellis path, speaker labels exact,
writers byte-identical). The **pybind11 adapter** makes this trivial — call the C++
functions directly from the existing pytest oracle.

## 8. Key decisions to confirm

| # | Decision | Recommendation |
|---|---|---|
| C1 | **Primary delivery** — server-centric (desktop+cloud) vs device-centric (offline mobile) | If desktop+cloud lead, this architecture is cleanest; if mobile-first, Flutter+FFI is more direct and the server buys less |
| C2 | **Inference runtime** — ✅ *resolved: all ONNX Runtime* | **All-ORT** (ONNX Runtime + sherpa-onnx) is the committed runtime for every stage and platform — one runtime, one threading model, transformer-friendly, and sherpa-onnx supplies 3/4 models off-the-shelf. LiteRT is dropped. |
| C3 | **Diarization-drives-VAD coupling** vs independent VAD | Coupling saves a segmentation pass; keep independent if simplicity preferred |
| C4 | **SPA framework + webview shell** (reuse Tauri vs CEF vs system) | Reuse the existing Tauri shell; pick one SPA framework |
| C5 | **Own engine in C++** vs consume sherpa-onnx as-is | You already get a C++ core free via sherpa; bespoke C++ adds value mainly for the server, alignment, and pybind oracle |

## 9. Roadmap

| Phase | Goal | Exit criteria |
|---|---|---|
| 0 | Core skeleton | CMake builds `libwhisperx`; decode-once + sherpa Whisper transcribes a WAV |
| 1 | Transcribe + VAD | silero VAD + **merge_chunks port**; matches Python segment text on golden clips |
| 2 | Alignment | wav2vec2 ORT (batched) + **Viterbi port**; golden word-timing parity *(highest risk — early)* |
| 3 | Diarization | sherpa diarize + **interval-tree assign port**; shared-segmentation option |
| 4 | Server + SPA | Drogon/oatpp HTTP+SSE; SPA transcribe→progress→export; reuse Tauri shell on desktop |
| 5 | pybind oracle + CI | pybind11 module; golden-parity tests green in CI |
| 6 | Mobile FFI | C-ABI consumed via JNI/Swift (or Flutter FFI); on-device run |
| 7 | Package | desktop installers (lean), signing; cloud image |

## 10. Risks

| Risk | Mitigation |
|---|---|
| C++ memory safety / build complexity (CMake, 5-platform cross-compile) | vcpkg/Conan; sanitizers; keep the core small and pure |
| wav2vec2 ONNX export / drift | validate early (phase 2); golden emission tests; pin opset |
| Mobile server anti-pattern | use FFI on mobile, not HTTP |
| SPA = extra frontend rewrite | scope it; reuse Tauri shell; or keep htmx initially behind the same server |
| Diarization quality vs full pyannote | sherpa pyannote-seg + CAM++; A/B on real clips |
| Losing Python reference | pybind11 keeps `app/`+`tests/` as oracle |

## 11. References

- **Migration plan (the *how*)**: [`cpp-core-migration-plan.md`](./cpp-core-migration-plan.md) — strangler-fig via pybind11, session-DB compatibility, headless test + timing suite, build tooling
- Pipeline spec: [`pipeline-reference.md`](./pipeline-reference.md)
- Why C++ is the substrate: [`single-language-runtime-options.md`](./single-language-runtime-options.md)
- Alternative delivery: [`flutter-migration.md`](./flutter-migration.md)
- Per-platform context: [synthesis](./packaging-shared-vs-bespoke.md) · [windows](./windows-port-options.md) · [`MACOS_INSTALLER.md`](../MACOS_INSTALLER.md)
- ONNX Runtime (C++ API, EPs): <https://onnxruntime.ai/docs/>
- sherpa-onnx (C++ VAD/ASR/diarization): <https://github.com/k2-fsa/sherpa-onnx>
