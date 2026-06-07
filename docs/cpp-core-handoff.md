# C++ Core Migration — Implementation Handoff Prompt

> Paste-ready kickoff for the engineer/agent implementing the WhisperX C++ core.
> The design is settled; this is the build phase. Read the four linked docs, then
> start at Phase 0.

---

## Mission

Reimplement the WhisperX transcription pipeline as a **headless C++ engine core**
(`libwhisperx`) on **ONNX Runtime + sherpa-onnx**, replacing the Python
`whisperx/` internals **stage by stage** while the Python `app/` keeps running as
the host and parity oracle. The deliverable is a lean, fast, drop-in-compatible
engine — not a rewrite of the app.

## Read first (authoritative, in this order)

1. [`pipeline-reference.md`](./pipeline-reference.md) — what each stage does + exact code/line refs (the spec).
2. [`cpp-core-spa-architecture.md`](./cpp-core-spa-architecture.md) — the engine-core + adapters design and pipeline streamlining.
3. [`cpp-core-migration-plan.md`](./cpp-core-migration-plan.md) — strategy (strangler-fig via pybind11), the **session-DB compatibility contract** (§2), test/timing approach, build tooling, and the **roadmap (§6 = single source of truth)**.
4. [`cpp-core-migration-briefs.md`](./cpp-core-migration-briefs.md) — **per-phase execution briefs** (Context / Goals / Validation / Unknowns). Start with "How to read" (3 cross-cutting facts + tolerances), then Phase 0.

[`single-language-runtime-options.md`](./single-language-runtime-options.md) is background (why C++ + ORT).

## Already decided — do **not** re-open

- **Runtime:** ORT + sherpa-onnx for VAD/align/diarize; **ASR is a pluggable backend** (sherpa-onnx default; whisper.cpp/GGML for Apple-Silicon Metal). LiteRT dropped.
- **Migration:** strangler-fig — Python stays the host until Phase 5; each C++ stage swaps in behind the **pybind11 `whisperx_core`** module, gated by `WHISPERX_CORE_STAGES`.
- **Validation (decoupled goldens):** Whisper text isn't byte-stable across engines → test align/diarize/writers against a **fixed transcript input**; judge ASR by **WER/CER**. "Byte-identical" only ever means *given identical input*.
- **Audio:** **ffmpeg libraries linked in-process** (Option B), no subprocess, all formats.
- **DB:** **replace** `app/store.py` with a C++ `SessionStore` (SQLiteCpp) — full API parity, honoring the §2 compatibility contract byte-for-byte.
- **Memory:** jobs **run to completion** (no mid-job cancel); mmap model weights + a per-job `std::pmr` arena reset at job end + mimalloc/jemalloc + **ASan/LSan in CI**.
- **Build:** **CMake + Ninja + vcpkg**, CTest, Catch2 (tests) + Google Benchmark (bench).
- **Dataset:** English + German + Russian, from **Common Voice Spontaneous Speech** (CC0) + LibriSpeech (EN). Two committed scripts in `golden/`: `fetch_datasets.py` pulls single-speaker ASR/align clips (`en_libri`; `ru_cv_*` for HF align loader + Cyrillic) + one m4a for ffmpeg decode; `synthesize_dialog.py` concats distinct speakers into **4-speaker synthetic dialogs** (`en/de/ru_dialog`, seed-pinned, byte-identical) with ground-truth RTTM for diarization. Pin lib + model revisions; diarization parity goldens from the **vendored** `app/models/speaker-diarization-community-1.3533c8cf/` checkpoint (no HF token). *Gap:* no real-overlap diarization (synthetic dialogs are sequential, no overlap) — add AMI later if needed.

## Phase 0 — landed (status + findings)

Phase 0 is **complete** (branch `cpp-core/phase0-scaffold`). The scaffold, goldens,
and decision-gate measurements are in; the findings below are now *settled facts*
for later phases — treat them like the "Already decided" list.

**What exists now**
- **`whisperx_core` pybind module** — root `CMakeLists.txt` (CMake+Ninja), pure-algo
  lib in `core/`, the module in `adapters/py/`. Light deps (pybind11/Catch2/
  nlohmann-json) via **FetchContent** so a fresh checkout builds with no vcpkg
  bootstrap; `vcpkg.json` seeds the heavy runtime deps for the production toolchain.
- **Tests/CI** — Catch2 + CTest under **ASan/UBSan** (`core/tests/`); the Python
  **parity oracle** `bindings/test/test_core_parity.py` (C++ `edit_distance` ==
  Python, bit-for-bit, incl. Cyrillic); CI in `.github/workflows/cpp-core.yml`
  (build + CTest + parity, no heavy `whisperx` sync).
- **Goldens** (`golden/`) — EN/DE/RU clips + `manifest.json`; the timing/WER baseline
  (`tests/test_baseline_golden.py` → `baseline.json`); and the **per-stage tensor
  intermediates** (`dump_goldens.py` → `intermediates/`: merged VAD chunks, CTC
  emissions, backtrack path, char-segments, words, **transcript-as-separate-input**),
  guarded torch-free by `tests/test_golden_intermediates.py`.

**Decision-gate findings (carry into later phases)**
- **wav2vec2 → ONNX works** (`WAV2VEC2_ASR_BASE_960H`, **opset 17**) and runs under
  ORT — the highest-risk Phase-3 step is de-risked early. `golden/measure_ort_tolerance.py`.
- **Tolerance budget is measured, not guessed:** ORT-vs-torch emission drift = **max
  2.8e-3 / mean 2.3e-4**; cross-process torch run-noise ≈ **2.9e-3** (float
  reduction-order). ⇒ **`emission_atol = 0.006`** (2× max drift), timings ±1 frame,
  scores ±0.01 (`golden/intermediates/manifest.json`, `tolerance_report.json`).
- **Emissions are never byte-equal** — only `atol`-compared. Reinforces the
  decoupled-goldens rule; applies to *any* fp32 tensor golden.
- **Trellis is not committed** (deterministically recomputable from emission+tokens).
  The exact-match parity gates are `path` / `char_segments` / `merged_chunks` /
  `speaker_labels` / `tokens`; emissions are the only fp32-`atol` golden.
- **Wildcard column:** align.py appends a wildcard label column when a transcript char
  is outside the model dictionary (e.g. `en_libri`: model emits 29 labels, golden is
  30). The C++ aligner must reproduce this extension (max non-blank score per frame)
  before trellis/backtrack.

**Remaining for later phases (not Phase 0):** open a PR for the branch; wire the heavy
vcpkg deps (ORT/sherpa-onnx) as the stages that need them land; port the existing
`tests/test_pipeline_contract.py` writer byte-goldens + constants to Catch2.

## Where to start

**Phase 0 is a decision gate, not just CMake setup.** Deliver: the `whisperx_core`
pybind scaffold (CMake+Ninja+vcpkg, ASan/LSan on), the **golden-vector generator**
over the EN/DE/RU clip set (transcript dumped as a *separate* artifact), pinned
manifest, CTest+Catch2 green, and a CI job. Then follow the critical path:

```
0 scaffold+goldens+decisions ─┬─► 1 DB (replace store.py)        [parallelizable]
                              └─► 2 decode+VAD ─► 3 alignment ─► 4a ASR / 4b diarize ─► 5 writers+e2e
                                                   (Python ASR feeds 3 until 4a lands)
6 timing gates accrue from Phase 2 on.
```

**Phase 3 (alignment) is the highest risk — front-loaded deliberately.** Validate
wav2vec2 ONNX export + the Viterbi port to golden parity *early*; if it's going to
fail, fail here. See the Phase 3 brief's parallelization section (batch inference
bucket-by-length + **mask padded frames** — masking is a golden-parity gate).

## Working agreement

- A phase is **done** when its stage matches Python on the golden set within the
  brief's tolerances, with the `WHISPERX_CORE_STAGES` flag on. Each brief's
  **Validation** section is the definition of done.
- Keep the **Python pipeline runnable** as the oracle the whole way (pybind makes
  this in-process). Never break `import whisperx` or the existing pytest suite.
- Port pure algorithms **verbatim** where the brief says so (Viterbi, IntervalTree,
  merge_chunks, writers) and golden-test them in isolation.
- Surface any **decide-during** question (per-phase Unknowns) before guessing —
  e.g. ONNX export opset, `default_asr_options` mapping, float/JSON formatting,
  CI timing-noise gating, the diarization CoreML-EP-vs-CPU benchmark. None block
  starting.

## First three concrete tasks

1. Stand up the CMake/Ninja/vcpkg project producing an importable `whisperx_core`
   pybind module (one real function, ASan/LSan on, CTest green in CI).
2. Write the golden generator: run the live Python pipeline over the EN/DE/RU
   clips, dump per-stage intermediates **+ the transcript separately**, with a
   pinned manifest (lib + model revisions + sha256).
3. Lock the **tolerance budget** by measuring torch-vs-ORT emission drift on one
   clip; record the chosen epsilon/frame values in the briefs.

When in doubt, the migration plan §6 roadmap and the per-phase briefs are the
contract; deviations should be raised, not silently taken.
