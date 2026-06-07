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
- **Store:** **replace** `app/store.py` with a C++ `SessionStore` (SQLiteCpp) — full API parity, honoring the §2 compatibility contract byte-for-byte. Scope is the **whole** store: the SQLite layer (token `db`) **and** the file-backed sidecars + the `app/edits.py` turn algorithms (token `edits`); both **landed** in Phase 1 (composable `WHISPERX_CORE_STAGES` tokens).
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

## Phase 1 — landed (full session-persistence layer)

The whole of `app/store.py` is ported to C++ — the SQLite layer **and** the
file-backed sidecars (edits/undo overlay + per-language translation files, with the
`app/edits.py` algorithms) — gated behind two composable `WHISPERX_CORE_STAGES`
tokens. Full detail in the Phase 1 brief; the settled facts for later phases:

- **The C++ store is full-surface.** `core/db/session_store.{hpp,cpp}` (on
  **SQLiteCpp**, bundled-SQLite via FetchContent) owns SQLite — CRUD/lifecycle,
  settings, speaker_names, the `translations` column, snapshot/swap — **plus** the
  file-backed sidecars: `load_result`/`load_edits`/`current_segments`/
  `save_turn_edit`/`save_turn_reassign`/`undo_turn_edit` + translation-file I/O,
  delegating the turn algorithms to `core/edits/edits.{hpp,cpp}` (the port of
  `app/edits.py`). Only the **pure path helpers** stay on the Python facade.
- **Strangler seam = Python facade, two composable flags.** `app.store.SessionStore`
  forwards its DB-method group to `self._db` and its file-method group to
  `self._edits`; each is the pure-Python `_PyStore` (default, the oracle) or
  `whisperx_core.SessionStore` (C++), picked by **`db`** / **`edits`** in
  `WHISPERX_CORE_STAGES` (either alone, or `db,edits` — one shared C++ store when
  both are on). `app/edits.py` is **also** a facade: `group_turns`/
  `distinct_speakers`/`next_speaker_key` (imported directly by `app/server.py` +
  `app/render.py`) route through C++ when `edits` is on (re-wrapping the dicts back
  into `Turn`), so the store and the server share one turn-grouping implementation
  (no turn-index desync). This is the pattern every later stage's swap follows.
- **Parity is proven both directions.** `bindings/test/test_store_parity.py` (DB) +
  `test_edits_parity.py` (per-function, **exact ==** on interpolated floats, plus a
  difflib-adversarial section) + `test_store_edits_parity.py` (edit sequences +
  on-disk overlay round-trips) assert behavioural parity **and** that
  `sessions.db` / `transcript.edits.json` written by either implementation read back
  identically through the other. The pure-stdlib `tests/test_turn_edits.py` oracle
  passes identically with the flags unset, `edits`, and `db,edits` (a CI matrix);
  the full pytest suite is green flag-off. C++ side: 34 Catch2 cases under ASan/UBSan.
- **The difflib port is the load-bearing piece.** `realign_words` needs
  `difflib.SequenceMatcher(autojunk=False)`'s **greedy matching blocks** (not a generic
  LCS) → a verbatim CPython `find_longest_match`/`get_matching_blocks` port in
  `core/text/sequence_matcher.*` (autojunk popularity-pruning branch omitted; confirmed
  via a >200-element adversarial parity case).
- **Float parity:** `core/edits/edits.cpp` is compiled with **`-ffp-contract=off`**
  (per-TU) so the gap-interpolation borrow arithmetic matches Python's never-fused
  IEEE-754 doubles bit-for-bit; words model timed-vs-untimed by **key presence**
  (`.contains()`, emitting neither start nor end), never null.
- **JSON parity is semantic** (parse-on-read), not byte-identical to `json.dumps`.
- **Threading:** the C++ store serializes the DB with its own `std::mutex` and the
  sidecar writes with a separate `files_lock_`; pybind calls keep the GIL (behaviour
  == today's Python store). GIL-release is a later tuning.

## Phase 2 — landed (2A `merge_chunks`/`vad` + 2B in-process decode/silero VAD/`decode`)

Phase 2 is **split into two composable slices** (the Phase 1 pattern), **both landed**.
2A is the pure, dep-free algorithm IP; 2B is the heavy in-process decode + ORT VAD.

- **`merge_chunks` is native (`vad` token).** `core/vad/merge_chunks.{hpp,cpp}` is a
  verbatim port of `whisperx/vads/vad.py::Vad.merge_chunks` (`whisperx::vad`, all
  `nlohmann::json`), in the always-built `whisperx_core_lib` (no new deps), with the
  audio hyperparameters in `core/audio/audio_constants.hpp`. `whisperx/vads/vad.py` is
  now a **facade**: `Vad.merge_chunks` routes to C++ when **`vad`** ∈
  `WHISPERX_CORE_STAGES` (keeping `_py_merge_chunks` as the oracle), covering both the
  Silero and Pyannote backends (they delegate to it). `WHISPERX_CORE_STAGES` now carries
  `db` / `edits` / `vad`, composably.
- **VAD parity is decoupled (settled).** silero ≠ pyannote and torch silero ≠ ORT
  silero, so live VAD segments aren't byte-comparable across engines. The **raw
  pre-merge segments** are dumped as a *fixed input* golden (`dump_goldens.py --vad-only`
  augments each `golden/intermediates/*.vad.json`, drift-guarded against the committed
  `merged_chunks`, leaving the tensor goldens untouched); `merge_chunks` is gated
  **exactly** on that input, the model's own segments only by a loose tolerance.
- **Parity proven.** `core/tests/test_merge_chunks.cpp` (Catch2/ASan) +
  `bindings/test/test_vad_parity.py` (per-function + golden-replay, exact). 40/40 CTest,
  73 bindings, full pytest green (226) across `WHISPERX_CORE_STAGES` ∈ {unset, `vad`,
  `db,edits,vad`}.
- **Slice 2B landed (`decode` token).** In-process `libav*` decode
  (`core/audio/decode.{hpp,cpp}` + the shared `AudioBuffer`) replaces the ffmpeg
  **subprocess**; `whisperx/audio.py::load_audio` is now a facade routing to
  `whisperx_core.load_audio` under **`decode`** (`_py_load_audio` is the oracle). The ORT
  **silero VAD** (`core/audio/vad_silero.cpp`, **sherpa-onnx**) emits the raw segments 2A's
  `merge_chunks` consumes; `whisperx/vads/silero.py` facades to it under `decode` (torch.hub
  stays the default/oracle — decoupled, smoke-only). Built behind a **`WHISPERX_CORE_AUDIO`**
  CMake option (default OFF → the dep-free fast lane is untouched): ffmpeg via `pkg-config`
  (system dev libs locally/CI; vcpkg ffmpeg for prod), sherpa-onnx vendored via FetchContent
  (it brings its own ONNX Runtime; built **shared** so ORT's C++ internals stay behind the C
  API). Pinned silero ONNX in `models/silero_vad.onnx`.
- **Two non-obvious 2B build gotchas (settled):** (1) sherpa bundles its own `nlohmann_json`
  and unconditionally `add_subdirectory()`s it → guard that one line so it reuses ours
  (see `CMakeLists.txt`). (2) Link ORT **shared**, not the static archive — the prebuilt
  static `libonnxruntime.a` is compiled against an older (glibc2_17) libstdc++ and corrupts
  the heap (`free(): invalid pointer` in `DeviceDiscovery`'s `std::regex`) when pulled into a
  system-libstdc++ binary; `BUILD_SHARED_LIBS ON` (sherpa scope) selects the shared/C-API path.
- **Gates met:** PCM **sample-for-sample** parity vs the subprocess on every golden clip incl.
  a committed **m4a/AAC** (wavs bit-exact, m4a ≤2 LSB) — `bindings/test/test_decode_parity.py`;
  VAD smoke (`test_vad_smoke.py`); decode-once contract (`tests/test_pipeline_contract.py`);
  40/40 CTest; full `uv run pytest tests/` green (226) across `WHISPERX_CORE_STAGES` ∈ {unset,
  `decode`, `vad,decode`, `db,edits,vad,decode`}; bench RTF (`bench/bench_audio`, decode ≈ VAD
  RTF ≪ 1). New vcpkg-less CI job (`cpp-core.yml` `audio-stage`, system ffmpeg + cached sherpa).

## Phase 3 — slice 3A landed (alignment: Viterbi core + assembly + native splitter)

Phase 3 (the highest-risk stage) is **split into two composable slices**; **3A is
landed**. 3A is the pure, dep-free algorithm IP behind a new composable **`align`**
token; **3B** (the heavy ONNX wav2vec2 forward) is the remaining slice.

- **Native Viterbi + assembly (`align` token).** `core/align/trellis.{hpp,cpp}`
  (`get_trellis`/`backtrack`/`merge_repeats`/`merge_words`) + `core/align/align.{hpp,cpp}`
  (the char→word→sentence assembly, replacing pandas with struct loops; `interpolate.hpp`
  ports `interpolate_nans`) live in the always-built `whisperx_core_lib` — **no new deps**.
  The CTC **emission is the fixed fp32 input** (decoupled-golden): the torch model forward
  + log_softmax + char-cleaning + wildcard extension + tokenization **stay in Python** (the
  facade); only the extended emission + tokens cross the seam. `whisperx/alignment.py::align`
  facades to `whisperx_core.align_assemble` under `align` (the Python body is the oracle,
  unchanged — the seam at `tests/test_pipeline_contract.py` is untouched since `align()`'s
  signature is preserved).
- **nltk punkt → native sentence splitter.** `core/text/sentence_split.{hpp,cpp}` (rule-based
  + vendored Moses `nonbreaking_prefixes`, embedded at build time; `core/text/utf8.hpp` for
  the codepoint cursor) replaces punkt. Evaluated + rejected: FreeLing (GPLv3 + weight), ICU
  (heavy dep, not abbrev-aware), a punkt port (no C++ one; moot after re-baseline). It
  **reproduces punkt exactly on every golden transcript** (so `words.json` needed **no**
  re-baseline) and agrees on 80% of a deliberately adversarial **en+ru+de+fr** corpus — the
  divergences are documented (stricter capital-after-period, case-sensitive prefixes) and
  in several cases the native splitter is the better call (`No. 5`, `e.g.`, `«…!»`).
  **German ordinals** ("1." = 1st) get a de-gated, ≤2-digit suppression rule so "Am 1.
  Januar" stays one sentence while year-final ("…1990. Danach…") + decimals ("1.23") still
  split; de/fr abbreviations + accented capitals are covered in `test_sentence_split.cpp`.
- **Gates met.** Viterbi **path + char_segments exact** vs the committed emissions on all 8
  clips (`bindings/test/test_align_parity.py`, torch-free); `align_assemble` word output
  matches `words.json` within ±1 frame / ±0.01 score; the splitter is pinned against a
  committed **punkt baseline** (`bindings/test/sentence_split_baseline.json`, generated by
  `golden/sentence_split_corpus.py`); native Catch2 (`test_trellis.cpp`,
  `test_sentence_split.cpp`, 62/62 CTest under ASan/UBSan); `tests/test_word_timestamp_interpolation.py`
  green under the `align` token; full `uv run pytest tests/` green (226) across
  `WHISPERX_CORE_STAGES` ∈ {unset, `align`, `vad,align`, `db,edits,vad,align`}; dep-free
  build (`WHISPERX_CORE_AUDIO=OFF`) unaffected. Goldens gained per-segment `dictionary` +
  `clean_cdx` (`golden/dump_goldens.py --align-io`) so the replay is torch-free.
- **3B remaining:** the C++ ONNX wav2vec2 forward (raw `Ort::Session` on 2B's vendored
  sherpa ORT, batched + length-masked), emissions vs golden within `emission_atol = 0.006`,
  model-asset export-on-demand. Inherits the 2B build facts: link ORT **shared**, the
  sherpa-`nlohmann_json` guard, decode-once `AudioBuffer::slice`.

## Where to start

**Phase 0 is a decision gate, not just CMake setup.** Deliver: the `whisperx_core`
pybind scaffold (CMake+Ninja+vcpkg, ASan/LSan on), the **golden-vector generator**
over the EN/DE/RU clip set (transcript dumped as a *separate* artifact), pinned
manifest, CTest+Catch2 green, and a CI job. Then follow the critical path:

```
0 scaffold+goldens+decisions ─┬─► 1 store.py (DB + edits sidecars) [parallelizable]
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
