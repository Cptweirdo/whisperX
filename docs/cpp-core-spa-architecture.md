# C++ Engine Core + SPA — Architecture & Pipeline Streamlining

> Architecture for rewriting the WhisperX pipeline as a **headless C++ engine
> core** with thin adapters — primarily an HTTP/SSE server behind a web **SPA**
> (desktop + cloud), plus a pybind11 oracle and an optional FFI adapter for
> embedding. Builds on [`pipeline-reference.md`](./pipeline-reference.md) (the
> stage spec) and [`single-language-runtime-options.md`](./single-language-runtime-options.md)
> (why C++ + ORT). The *how* lives in
> [`cpp-core-migration-plan.md`](./cpp-core-migration-plan.md) and the per-phase
> [`cpp-core-migration-briefs.md`](./cpp-core-migration-briefs.md).

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
- **Adapters**, not a monolith: HTTP/SSE server (→ SPA, desktop + cloud),
  pybind11 (→ keep Python `app/` + `tests/` as the oracle), and an optional C-ABI
  FFI adapter for embedding.
- **Models on one runtime (ONNX Runtime)**, with ASR pluggable (§4B) — collapse
  Python's four ML stacks toward one.
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
   │ HTTP/SSE server  │ pybind11       │ FFI (C ABI)
   │ (Crow/Drogon/…)  │ (reference/    │ (optional,
   ▼                  │  tests)        │  embedding)
 Web SPA              ▼                ▼
 (desktop+cloud)   Python app/ +     embedding host
                   pytest oracle
```

The core knows nothing about HTTP, FFI, or UI. Each adapter is a thin shell. The
result `schema.py` shapes become the **API/FFI contract**.

## 3. Delivery: server + SPA (desktop + cloud)

The primary delivery is **C++ core + HTTP/SSE server + web SPA**, run as a desktop
app (server bundled behind a webview) or a self-hosted/cloud service. Desktop
bundles get **lean** — a native binary + static SPA assets, with **no
`python-build-standalone`, no torch/CUDA, no cuDNN matching** — so the existing
packaging pain largely disappears; an existing webview shell just points at the C++
server instead of Flask.

The same core also exposes an **optional C-ABI FFI adapter** for embedding the
engine directly in another host (no localhost server). That's a secondary path and
not the focus here.

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

### B. Collapse ML stacks toward one runtime (ORT), with a pluggable ASR backend
Python uses **four** inference stacks: CTranslate2 (Whisper), torch/HF (wav2vec2),
pyannote/torch (diarization), torch.hub (silero). Run **VAD, alignment, and
diarization on ONNX Runtime** (sherpa-onnx already does VAD + diarization; add
wav2vec2 as an ORT model). One runtime for those → one threading model, one memory
arena, one set of EPs (Core ML/NNAPI/CUDA/DirectML).

**Caveat — ASR is a pluggable backend, not folded into ORT.** ORT has **no Metal
EP**, and Whisper-on-Apple-GPU is exactly where ORT is weak, so the existing
**whisper.cpp/GGML (Metal)** backend stays as the Apple-Silicon ASR path (default
remains sherpa-onnx Whisper on ORT for cross-platform). So "one runtime" really
means **one runtime for VAD/align/diarize + a pluggable ASR engine** — and
**one runtime ≠ one accelerator**: on Mac you get Metal for ASR (GGML) and
CoreML-EP-or-CPU for the ORT stages. See
[`cpp-core-migration-briefs.md`](./cpp-core-migration-briefs.md) §"Runtime &
acceleration" for the backend interface and the Apple-Silicon acceleration map.

This still **deletes heavy transitive deps**:

| Drop | Replace with |
|---|---|
| `pandas` (alignment `:325,395`, diarize `:170`) | plain structs + loops |
| `nltk` punkt (`alignment.py:189`) | ICU / small punctuation-rule splitter |
| `torch`, `torchaudio`, `transformers`, `faster-whisper`, `ctranslate2`, `pyannote-audio` | ORT + sherpa-onnx (+ whisper.cpp/GGML for the Metal ASR backend) |

Dependency graph shrinks from dozens of packages to ≈ **ORT + sherpa-onnx +
whisper.cpp + audio decoder + your code**.

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
One runtime (ASR pluggable) · one decode · models resident · one job at a time ·
glue in tight C++ — versus four ML stacks, up-to-3× decode, sequential
load/unload, pandas/nltk today.

## 5. Tech choices

| Concern | Options | Note |
|---|---|---|
| Inference | **ONNX Runtime** (C++) + **sherpa-onnx** (VAD/ASR/diarize) | one runtime; EPs for GPU/ANE/NNAPI |
| wav2vec2 align | ORT model + own Viterbi | the one DIY model |
| Audio decode | **ffmpeg libraries** (`libavformat`/`libavcodec`/`libswresample`), linked in-process | **no ffmpeg subprocess** (Option B): one universal decode path for all formats incl. M4A/AAC/MP4/video; decode to float in-memory; LGPL build |
| HTTP/SSE server | **Drogon** or **oat++** (perf, WebSocket/SSE) · **cpp-httplib** (header-only, simple) | SSE for progress, mirroring `app/` |
| SPA | ✅ **Svelte 5** (Bun + Vite, `app/web/`) — built | true client-rendered SPA over the JSON/SSE API; already shipped |
| Build | CMake + vcpkg/Conan; per-desktop-OS builds | the main ergonomic cost |
| FFI adapter (optional) | C ABI for embedding | reuses the core, no server |
| Python oracle | pybind11 | keep `app/` + `tests/` as parity reference |

> **Note on the SPA (done — was a planned workstream):** the `app/` frontend is now a
> standalone **Bun + Vite + Svelte 5 SPA** (`app/web/`), and the Flask backend is a
> **pure JSON API + SSE** (no Jinja/htmx; see CLAUDE.md). The "move to an SPA" this doc
> once scoped as "a second workstream, not free" is **already complete**, which
> *de-risks the endgame*: the SPA is transport-agnostic — it talks to a stable
> **JSON `/api/*` + SSE** contract, not to Flask specifically. So the eventual C++
> HTTP/SSE server adapter is a pure backend swap behind that contract; the SPA is
> reused unchanged (same as the Tauri webview shell). **The JSON/SSE API is now a
> migration contract** the C++ server must reproduce — the transport-layer analogue of
> the session-DB contract (migration plan §2), but only relevant at the (out-of-scope)
> host-swap, not during the strangler phases. **Full route-by-route spec:**
> [`api-reference.md`](./api-reference.md).

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
  ffi/        C ABI (optional, embedding)
  py/         pybind11 module (oracle)
app/web/      SPA — Svelte 5 + Vite + Bun (already built; reused as-is)
bindings/test golden vectors + parity tests
```

## 7. Golden-parity strategy (unchanged)

The golden-vector discipline: dump intermediates from Python `whisperx`
(VAD/merged chunks, CTC emissions/trellis, backtrack path, word timings,
diarization turns, speaker labels) into golden JSON; assert the C++ core matches
within tolerance (`merge_chunks` boundaries, trellis path, speaker labels exact,
writers byte-identical). The **pybind11 adapter** makes this trivial — call the C++
functions directly from the existing pytest oracle.

## 8. Key decisions to confirm

| # | Decision | Recommendation |
|---|---|---|
| C2 | **Inference runtime** — ✅ *resolved: ONNX Runtime* | **ORT + sherpa-onnx** for VAD/align/diarize; **ASR pluggable** (sherpa-onnx default, whisper.cpp/GGML for Apple-Silicon Metal, §4B). One runtime, transformer-friendly, sherpa supplies 3/4 models. LiteRT dropped. |
| C3 | **Diarization-drives-VAD coupling** vs independent VAD | Coupling saves a segmentation pass; keep independent if simplicity preferred |
| C4 | **SPA framework + webview shell** — ✅ *resolved/done* | **Svelte 5 SPA shipped** (`app/web/`, Bun + Vite) against the JSON/SSE API; Tauri webview shell reused. No longer a workstream. |
| C5 | **Own engine in C++** vs consume sherpa-onnx as-is | You already get a C++ core free via sherpa; bespoke C++ adds value mainly for the server, alignment, and pybind oracle |

## 9. Roadmap

> The **authoritative, compat-aware roadmap** lives in
> [`cpp-core-migration-plan.md`](./cpp-core-migration-plan.md) §6, with detailed
> per-phase execution briefs in
> [`cpp-core-migration-briefs.md`](./cpp-core-migration-briefs.md). In short:
> **0** scaffold + golden generator + decision gate · **1** session store
> (SQLiteCpp, replaces all of `store.py` — DB layer + edits/translation sidecars +
> `app/edits.py`) · **2** decode-once + VAD/`merge_chunks` **landed** (2A
> `merge_chunks`/`vad`; 2B in-process `libav*` decode + ORT silero/`decode`) · **3** alignment
> (batched wav2vec2 + Viterbi — *highest risk, early*) · **4a** ASR backends /
> **4b** diarize + assign · **5** writers + end-to-end · **6** timing gates.
> The **SPA itself is already built** (Svelte, `app/web/`); only the C++ HTTP/SSE
> **server** + packaging (and the optional FFI adapter) follow once the engine is
> green. (This supersedes an earlier draft roadmap that numbered the phases
> differently — see the migration plan as the single source of truth.)

## 10. Risks

| Risk | Mitigation |
|---|---|
| C++ memory safety / build complexity (CMake, per-desktop-OS builds) | vcpkg/Conan; sanitizers; keep the core small and pure |
| wav2vec2 ONNX export / drift | validate early (phase 2); golden emission tests; pin opset |
| ~~SPA = extra frontend rewrite~~ — ✅ done | Resolved: the Svelte SPA (`app/web/`) already ships against the JSON/SSE API; no longer a risk. The remaining transport work is the C++ server reproducing that API contract. |
| Diarization quality vs full pyannote | sherpa pyannote-seg + CAM++; A/B on real clips |
| Losing Python reference | pybind11 keeps `app/`+`tests/` as oracle |

## 11. References

- **Migration plan (the *how*)**: [`cpp-core-migration-plan.md`](./cpp-core-migration-plan.md) — strangler-fig via pybind11, session-DB compatibility, headless test + timing suite, build tooling
- **Per-phase briefs**: [`cpp-core-migration-briefs.md`](./cpp-core-migration-briefs.md)
- Pipeline spec: [`pipeline-reference.md`](./pipeline-reference.md)
- Why C++ + ORT: [`single-language-runtime-options.md`](./single-language-runtime-options.md)
- ONNX Runtime (C++ API, EPs): <https://onnxruntime.ai/docs/>
- sherpa-onnx (C++ VAD/ASR/diarization): <https://github.com/k2-fsa/sherpa-onnx>
