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
- **Phase 3 slice 3B — landed (native ONNX wav2vec2 forward, `align_onnx` token).** The
  torch model forward (`alignment.py:278-285`) now runs in C++ under a raw `Ort::Session`
  (2B's vendored sherpa ORT, linked **shared**): `core/align/wav2vec2_onnx.{hpp,cpp}`
  (pImpl, bucket-by-length + padded **attention_mask** batched forward, trimmed by the
  graph's `frame_lengths` output) + the pure `core/align/emission_post.{hpp,cpp}`
  (log_softmax + OOV wildcard column + tokenization — "path 2", so the mirror-lang align
  path is **torch-free** forward→post→assemble). `whisperx/alignment.py` facades under
  **`align_onnx`**: `load_align_model` pulls the parity-pinned `.onnx`+`meta.json` from the
  mirror and builds a `Wav2Vec2Onnx`; `align()` does one batched forward over all segments,
  then per-segment `align_emission_post` → `align_assemble` (3A). **Batching is masked +
  data-gated:** only `layer_norm` extractors are batched (`meta["batchable"]`); the
  torchaudio **`group_norm`** bundles normalize over time, so padded batches corrupt valid
  frames (eager drift ~6 ≫ atol) → they run **per-segment**. Mirror re-exported to
  **contract v2** (dynamo, opset 18, inputs `waveform`+`attention_mask`, outputs
  `emissions`+`frame_lengths`). **Gates met:** C++ forward+post reproduces every golden
  `seg{i}_emission` within `emission_atol = 0.006` **per-segment and batched** + exact
  tokens across all 8 clips / both loaders (`bindings/test/test_align_onnx_forward_parity.py`,
  `RUN_MIRROR=1`); `align()` under `align_onnx` matches `words.json` **exactly** (en + ru
  112-word dialog); Catch2 `test_emission_post` (6 cases) under ASan/UBSan; 66/66 CTest;
  `import whisperx` + dep-free build unaffected. Inherits the 2B facts (ORT **shared**,
  sherpa-`nlohmann_json` guard, decode-once `AudioBuffer::slice`).
  - *Known gap:* `<400-sample` segments are padded to the conv minimum + masked (now
    correct for batchable models); none in the goldens, so untested.
- **3B model assets — the ONNX mirror (landed, pre-3B).** The aligner consumes a *path
  to a parity-valid `.onnx`*, produced **offline** (PyTorch is build/CI tooling, never the
  runtime) and hosted on **`KonstantK/wav2vec2-align-onnx`** (HF public repo — free, CDN,
  reuses the `huggingface_hub` cache/sha/revision-pin path the original weights already
  use). `golden/export_align_onnx.py` reuses `load_align_model` + the two
  `DEFAULT_ALIGN_MODELS_{TORCH,HF}` tables (the model registry) and a `loader-type→wrapper`
  dispatch, so **any of the 43 languages exports via `--lang <code>` with no code change**;
  `model.onnx` emits **raw logits** (opset 17, dynamic axes; consumer applies log_softmax +
  the OOV wildcard column) and `meta.json` ships the `dictionary`/`blank_id` so tokenization
  needs no torch. Each export is **self-checked against the golden emissions before upload**;
  re-runs **skip already-published models** (keyed on opset + upstream revision) unless
  `--force`. Validated round-trip (export→upload→**download**→ORT) by the opt-in
  `bindings/test/test_align_onnx_mirror.py` (`RUN_MIRROR=1`): every golden `seg{i}_emission`
  reproduced within `emission_atol` across all 8 clips / both loader families (en+de
  torchaudio, ru HF). Producer is swappable (torch-export now → CI-hosted later) with the
  C++ seam unchanged. Phase-5 packaging decides which langs bundle vs lazy-download.

## Phase 4 — slice `assign` landed (speaker-assignment glue)

Phase 4 swaps the last two models — **ASR** (4a) and **diarization** (4b) — and,
like every prior phase, the **pure dep-free algorithm IP ships first** behind its own
composable token. **Slice `assign` is landed**: the 4b analog of 3A.

- **Native `IntervalTree` + `assign_word_speakers` (`assign` token).**
  `core/diarize/interval_tree.{hpp,cpp}` + `core/diarize/assign_speakers.{hpp,cpp}`
  (verbatim ports of `diarize.py:14,185`) live in the always-built `whisperx_core_lib`
  — **no new deps**. Pure JSON segments in/out (carries arbitrary seg/word fields
  untouched; only `"speaker"` keys are written). `whisperx/diarize.py::assign_word_speakers`
  facades to `whisperx_core.assign_word_speakers` under **`assign`** (the pandas body
  `_py_assign_word_speakers` stays the oracle). The pyannote **DataFrame stays Python**
  — the facade extracts it to `(start,end,speaker)` turns and the C++ takes structs
  ("replace the DataFrame with structs"); `speaker_embeddings` passthrough stays Python
  (a dict copy, no algorithm).
- **Mutate-and-return contract.** Callers rely on **both** in-place mutation
  (`dump_goldens.py`, `tests/test_baseline_golden.py` ignore the return) **and** the
  return value (`transcribe.py:233`, `app/pipeline.py:563`). The C++ returns fresh
  dicts, so the facade writes the speaker labels **back into the original seg/word
  dicts in place** and returns — both contracts hold.
- **Load-bearing parity = the tie-breaks.** `max(dict.items(), key=…)` first-wins on a
  tie (insertion order from the IntervalTree query) and `np.argmin` first-min in
  `find_nearest`, plus `searchsorted(side='left')`; all ported exactly, `-ffp-contract=off`.
- **Model-independent golden.** The committed `words.json` speakers are entangled with
  the live `community-1` turns, so the isolation gate uses a **new** `*.assign.json`
  (`golden/dump_goldens.py --assign`, align models only — no whisper/diarize): the
  synthetic dialogs' **CC0 ground-truth RTTM** turns fed once through the Python assign
  oracle over the pre-diarize aligned segments (`{turns, segments_in, segments_out}`).
  The C++ replay is therefore torch/pandas/model-free.
- **Gates met.** C++ `assign_word_speakers` reproduces the oracle's segment **and** word
  speaker labels **exactly** on all 3 dialogs (`bindings/test/test_assign_parity.py`);
  Catch2 `test_assign_speakers` (9 cases: ties, no-overlap, `fill_nearest`, untimed
  word, empties, argmin/searchsorted) under ASan/UBSan; 78/78 CTest; the facade's C++
  path matches the oracle end-to-end incl. in-place mutation + embeddings passthrough;
  `test_pipeline_contract.py` seam untouched; full `uv run pytest tests/` green (226)
  across `WHISPERX_CORE_STAGES` ∈ {unset, `assign`, `db,edits,vad,align,assign`};
  `import whisperx` + dep-free build unaffected.
- **Remaining Phase 4 slices (later):** the **diarization model** swap (sherpa
  pyannote-seg-3.0 + CAM++) is **A/B, not parity** — Python pyannote `community-1` keeps
  feeding the turns this glue consumes (strangler).

## Phase 4 — slice `asr` landed (sherpa-onnx Whisper ASR backend)

The first 4a backend. **Whisper** moves to a **pluggable backend** behind a new
composable **`asr`** token; faster-whisper (CTranslate2) stays the **default + WER/CER
oracle** (strangler). No pure-IP sub-slice — ASR is all model forward + decode, so this
is one heavy slice. **whisper.cpp/GGML** (Apple-Silicon Metal) remains an explicit
follow-on.

- **Native sherpa-onnx Whisper (`asr` token).** `core/asr/whisper_sherpa.{hpp,cpp}`
  (pImpl, built only under `WHISPERX_CORE_AUDIO`) wraps sherpa's high-level
  **`OfflineRecognizer`** (mel + encoder/decoder forward + greedy detokenize + language
  ID all inside sherpa) — reuses the **already-vendored** sherpa-onnx (2B), not a fresh
  ORT setup, so it's a thin batched front-end over the VAD spans (the wav2vec2 raw
  `Ort::Session` pattern was **not** needed — decoupled WER/CER makes greedy-vs-beam a
  non-issue). `WhisperSherpa::transcribe(AudioBuffer, spans, lang, task)` slices the
  shared buffer per VAD span (zero-copy, 2B's `slice`); `detect_language` reads sherpa's
  `<|xx|>` over the first 30 s. `avg_logprob` is best-effort (sherpa's `ys_log_probs`;
  0.0 when the build doesn't expose them — a populated field only, never load-bearing).
- **Python facade.** `whisperx/asr_sherpa.py` (`SherpaWhisperPipeline` +
  `load_sherpa_model`, sibling of `asr_mlx`/`asr_whispercpp`) reuses the identical VAD
  serial loop + `merge_chunks`, swapping the decode engine for `whisperx_core.WhisperSherpa`
  — same `transcribe() -> {segments, language}` contract, so align/diarize are untouched.
  `whisperx/asr.py::load_model` routes to it under `_core_asr_enabled()` (the `asr` token,
  `hasattr`-guarded); signature preserved so `test_pipeline_contract.py` is untouched.
- **Model assets — mirror + sha-pin (3B mechanism), with a fallback.**
  `golden/mirror_whisper_onnx.py` re-hosts sherpa's pre-exported Whisper ONNX
  (`encoder`/`decoder`/`tokens`, already ONNX — **no torch export**) to a repo we control
  (`KonstantK/whisper-onnx-sherpa`, public), sha-pinned, with a `meta.json` (asset
  filenames + `feature_dim` + shas). **Published: tiny / base / small** (the common
  CPU/interactive set; tiny is the golden/CI model, matches `baseline.json`). The runtime
  resolver (`asr_sherpa._resolve_sherpa_assets`) tries, in order: (1) a local dir
  (`WHISPERX_SHERPA_WHISPER_DIR`, dev/CI), (2) our sha-pinned mirror via `huggingface_hub`
  (the `load_align_model` cache path), (3) **sherpa-onnx's official release tarball**
  (cached under `~/.cache/whisperx-sherpa`) for any model not on the mirror — so large-v2
  etc. still load without us hosting GBs. Bundle-vs-lazy is Phase-5 packaging.
- **Decoupled gate (settled fact 3).** Whisper text isn't byte-stable across decoders, so
  the gate is **WER/CER vs the faster-whisper tiny baseline** (`golden/baseline.json`),
  not exact text. `bindings/test/test_asr_sherpa_parity.py` runs the native backend over
  each clip's **committed VAD spans** (`*.vad.json` merged_chunks — model-free) and asserts
  WER ≤ baseline + 0.30 / CER ≤ baseline + 0.20 (measured worst-case greedy-vs-beam drift
  ≈ +0.18 WER on a short CV clip), exact language detection, and the facade pipeline glue
  shape (via a stub VAD, torch-free). **Gates met:** 15/15 on the en/de/ru golden set;
  the seam (`test_pipeline_contract.py`) green under `asr`; full `uv run pytest tests/`
  green (226) across `WHISPERX_CORE_STAGES` ∈ {unset, `asr`, `db,edits,vad,align,assign,asr`};
  `import whisperx` clean (facade `hasattr`-guarded); dep-free `WHISPERX_CORE_AUDIO=OFF`
  build unaffected (the TU only joins `whisperx_core_audio`). CI: a torch-free
  `WhisperSherpa` smoke in the `audio-stage` lane (cached sherpa tiny).
- **Remaining Phase 4 slices (later):** the **diarization model** swap (sherpa
  pyannote-seg-3.0 + CAM++, **A/B not parity**) and the **whisper.cpp/GGML** ASR backend
  (4a follow-on, Metal-shader/CMake build).

## Phase 4 — slice `diarize` landed (sherpa-onnx speaker diarization)

The diarization **model** moves to a pluggable backend behind a new composable
**`diarize`** token; pyannote `community-1` (Python/torch) stays the **default
oracle** (strangler). **A/B, not parity — by construction:** community-1 is
pyannote.audio 4.0 (its own segmentation + embedding + **VBxClustering/PLDA**),
sherpa uses pyannote-segmentation-3.0 + a speaker-embedding extractor + cosine
**FastClustering** — different models ⇒ different turns. The landed `assign`
glue consumes either model's turns unchanged.

- **Native sherpa-onnx diarization (`diarize` token).** `core/diarize/diarize_sherpa.{hpp,cpp}`
  (pImpl, built only under `WHISPERX_CORE_AUDIO`) wraps sherpa's high-level
  **`OfflineSpeakerDiarization`** (segmentation + embedding + FastClustering inside
  sherpa) — reuses the vendored sherpa-onnx/ORT (2B). `diarize(AudioBuffer, num_clusters)`
  returns `{start,end,speaker:int}` turns; `embeddings(AudioBuffer, segments)` runs a
  **second `SpeakerEmbeddingExtractor` pass** (the diarization result exposes no
  vectors) feeding each cluster's turns into one stream → a per-speaker vector
  (sherpa's result has no embeddings, unlike pyannote). Inherits the 2B facts (ORT
  **shared**, sherpa-`nlohmann_json` guard, `AudioBuffer::slice`).
- **Python facade.** `whisperx/diarize_sherpa.py` (`SherpaDiarizationPipeline` +
  `load_sherpa_diarize_model`, sibling of `asr_sherpa.py`) emits the **same pandas
  DataFrame** `[segment,label,speaker,start,end]` the pyannote path does (cluster int →
  `SPEAKER_xx`), so `assign_word_speakers` and all callers are untouched; preserves the
  optional `return_embeddings` `(df, {speaker: vec})` return. `whisperx/diarize.py::DiarizationPipeline`
  facades under `_core_diarize_enabled()` (the `diarize` token, `hasattr`-guarded):
  `__init__` builds the sherpa pipeline into `self._impl` and `__call__` delegates, else
  the pyannote oracle path runs unchanged. **Speaker-count controls → FastClustering's
  single `num_clusters`:** precedence `num_speakers → max_speakers → min_speakers → auto`
  (sherpa has no native min/max range); `num_clusters` is a clustering *target* — the
  pyannote impl's frame-level finalization can yield fewer — not a hard guarantee.
- **Model assets — mirror + sha-pin (3B mechanism), with a fallback.**
  `golden/mirror_diarize_onnx.py` re-hosts sherpa's pre-exported **pyannote-segmentation-3.0**
  + **wespeaker_en_voxceleb CAM++** ONNX (already ONNX — no torch export; MIT + Apache-2.0)
  to `KonstantK/diarize-onnx-sherpa` (public), sha-pinned, with a `meta.json` (asset names +
  `embedding_dim` + shas + license). The runtime resolver (`diarize_sherpa._resolve_diarize_assets`)
  tries: (1) a local dir (`WHISPERX_DIARIZE_ONNX_DIR`, dev/CI), (2) our mirror via
  `huggingface_hub`, (3) sherpa-onnx's official release assets (cached). The embedding model
  is **English-VoxCeleb CAM++** (~28 MB, 512-dim) — best language family for en/de/ru at
  small/fast size (the zh-cn CAM++ would be the wrong family). Bundle-vs-lazy is Phase-5.
- **A/B gate (not parity).** The synthetic dialogs are concatenated CommonVoice clips and
  are *hard for any diarizer* — measured, the incumbent **community-1 itself scores DER
  ≈ 32 % (en) / 32 % (de) / 64 % (ru)** with the wrong speaker count on them; sherpa is
  comparable (≈ 43/32/45 %, and better than community-1 on ru). So there is **no `count==4`
  or low-DER claim**: `bindings/test/test_diarize_sherpa.py` (opt-in,
  `WHISPERX_DIARIZE_ONNX_DIR`/`RUN_MIRROR`) gates **DER ≤ 0.70** vs the CC0 RTTM (a
  regression bound), ≥ 2 speakers, the **assign-glue contract** (its DataFrame feeds
  `assign_word_speakers` → labels land on segments + words), per-speaker embedding dim, and
  a speaker-count-control sanity. **Gates met:** 5/5 on en/de/ru; the seam
  (`test_pipeline_contract.py`) green under `diarize`; full `uv run pytest tests/` green
  across `WHISPERX_CORE_STAGES` ∈ {unset, `diarize`, `db,edits,vad,align,assign,diarize`};
  `import whisperx` clean (facade `hasattr`-guarded); dep-free `WHISPERX_CORE_AUDIO=OFF`
  build unaffected. CI: a torch-free `SherpaDiarizer` smoke in the `audio-stage` lane.
- **Remaining Phase 4 slice — deferred/skipped:** the **whisper.cpp/GGML** ASR backend
  (4a follow-on, Metal-shader/CMake build) **and its Apple-Silicon Metal acceleration** are
  **explicitly out of scope for now** (build cost not yet worth it). Phase 4 has no active
  remaining slice; sherpa-onnx Whisper is the ASR backend on all platforms until/unless the
  GGML path is picked up.

## Phase 5 — slice `writers` landed (native output writers)

Phase 5 ports the output writers and (later) a native end-to-end orchestrator. Per the
prior-phase pattern the **pure dep-free writer IP ships first** behind a new composable
**`writers`** token; the **`orchestrate`** slice (a fully-native `run_job`) is **deferred**
(see below).

- **Native writers (`writers` token).** `core/writers/writers.{hpp,cpp}` (verbatim port of
  `whisperx/utils.py`'s `ResultWriter` family + `SubtitlesWriter.iterate_result`) live in the
  always-built `whisperx_core_lib` — **no new deps** (pure `nlohmann::json` in / `std::string`
  out, the `merge_chunks`/`assign` pattern). All **six** formats: `write_srt`/`write_vtt`/
  `write_txt`/`write_tsv`/`write_aud`/`write_json` + `format_timestamp` + the full
  `iterate_result` (the word-grouping cue loop incl. the CLI-live `highlight_words` `<u>` path
  and `max_line_width`/`max_line_count` line-splitting, `LANGUAGES_WITHOUT_SPACES` join,
  speaker `[SPK]:` prefix, the no-`words` segment fallback). `whisperx/utils.py` is a **facade**:
  `ResultWriter.__call__` routes to `whisperx_core.write_<fmt>` under `writers` (one routing
  point covers all six subclasses; file naming + open stay Python, `get_writer`'s signature
  unchanged), keeping the Python bodies as the oracle.
- **Byte-parity is provable (the brief's float-repr worry dissolves).** srt/vtt/txt/tsv
  timecodes are `round(seconds*1000)` **integer-ms** math (`nearbyint`, banker's), so they're
  pure integer + string formatting → **byte-identical**. `aud` prints raw floats via a
  Python-`repr`-compatible formatter (the increasing-`%.{p}g` shortest-round-trip loop + the
  repr `.0` rule) → byte-identical on the timestamp domain. `json` is `dump()` — **semantic**
  round-trip parity (compact/sorted-key OK; values + structure round-trip — the Phase-1
  store-JSON precedent; the contract test checks `json.loads`, not bytes). **`SubtitlesProcessor.py`
  is NOT ported** — it is dead code (imported nowhere in `whisperx/`/`app/`); revisit if a
  caller appears.
- **Gates met.** `bindings/test/test_writers_parity.py` — C++ `write_<fmt>` vs the Python
  `write_result` oracle, **byte-exact** for srt/vtt/txt/tsv/aud and **round-trip-equal** for
  json, across 5 result shapes × 4 option-sets (defaults / highlight / line-width / both) +
  `format_timestamp` parity (**106 cases**); Catch2 `test_writers.cpp` (7 cases: timecode
  rounding/hours gating, the `iterate_result` branch matrix, the `<u>` wrap) under ASan/UBSan;
  **85/85 CTest**; the existing `test_pipeline_contract.py` writer byte-goldens + json
  round-trip stay green under `writers`; full `uv run pytest tests/` green (226) across
  `WHISPERX_CORE_STAGES` ∈ {unset, `writers`}; `import whisperx` clean; dep-free build
  unaffected (the writers TU is in `whisperx_core_lib`, no audio deps).
- **Slice `orchestrate` — deferred; next up is the align driver.** A *fully-native* `run_job`
  was the intended second slice, but the **align stage has no native driver**:
  `alignment.py::align`'s Phase-1 preprocessing (per-language char-clean →
  `clean_char`/`clean_cdx`, the OOV/wildcard handling, the gather + batched-forward loop,
  sentence spans) stays Python — only the native *pieces* `Wav2Vec2Onnx.forward` /
  `emission_post` / `align_assemble` exist, and they consume already-preprocessed inputs. So a
  "no Python re-entry" orchestrator isn't achievable for align without first porting that
  driver. **Decided: the next slice ports the align driver** (making `align()` a true native
  entrypoint), *then* the 100%-native orchestrator builds on it — **not** the hybrid
  Python-callback shape. (Watch: `align()` still calls **nltk punkt** for `sentence_spans` even
  though `align_assemble` already carries the native splitter — the driver port must reconcile
  that.)

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
