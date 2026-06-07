# C++ Core Migration Plan

> The *how* for the [`cpp-core-spa-architecture.md`](./cpp-core-spa-architecture.md)
> decision. That doc argued **what** to build (a headless C++ engine core +
> adapters) and **why**; this one is the migration plan: how to get there from
> today's Python pipeline **without a big-bang rewrite**, how to **preserve
> session-DB compatibility** for an in-place upgrade, what **headless test +
> timing** suite proves the port correct, which **C++ libraries replace the
> Python counterparts**, and what **build tooling** to standardize on.
>
> Grounded by reading the real code: `app/store.py` (DB), `app/pipeline.py::run_job`
> (orchestration), `tests/` (pytest — engine + JSON-API + SSE), the SPA tests
> (`app/web/tests/`: Vitest units + a **Playwright end-to-end suite** under
> `tests/e2e/` that drives the real Svelte SPA against a seeded Flask backend), and
> the six pipeline stages in `whisperx/` (see
> [`pipeline-reference.md`](./pipeline-reference.md)).

## 1. Migration strategy — strangler-fig via pybind11

A from-scratch C++ rewrite that's only validated at the end is the failure mode.
Instead we **strangle** the Python engine one stage at a time, each validated
against the live Python stage before moving on. The cpp-core doc already names
**pybind11 as the parity oracle**; here it doubles as the *migration vehicle*.

- **Keep `app/` (Flask + SQLite + SSE) running as the host.** Replace the engine
  internals (`whisperx/`) stage by stage with C++ compiled into a pybind11 module
  (`whisperx_core`). `app/pipeline.py::run_job` keeps its exact shape —
  `_stage("decoding"/"transcribing"/"loading_align"/"aligning"/"diarizing")`,
  `on_duration`, `progress`, `cancel_event` — and the `schema.py` result shapes
  are unchanged. The host doesn't know a stage went native.
- **Stage order = alignment-first for risk** (matching the cpp-core roadmap):
  decode → VAD/`merge_chunks` → ASR → **wav2vec2 align + Viterbi (highest risk,
  do early)** → diarize + `IntervalTree` assign → writers. Front-load the one
  stage with no off-the-shelf binding so its risk is retired first.
- **Side-by-side diffing.** Each ported stage sits behind a flag
  (`WHISPERX_CORE_STAGES=decode,align,…`). With it off, Python runs; on, C++
  runs. Run both on real audio and diff outputs until the stage is trusted, then
  flip the default. This is the strangler seam.
- **Endgame.** Once every stage is C++, the pybind module *is* `libwhisperx`;
  attach the server/FFI adapters from the cpp-core doc. Whether `app/` stays (a
  thin Python host over the lib) or is replaced by the C++ HTTP/SSE server is a
  **later** decision — not part of this migration, and the pybind path lets us
  defer it without blocking.

Why this order works: the schema and the orchestration contract (`run_job`) are
the stable spine; we swap the muscle underneath them incrementally, and the
Python pipeline stays runnable as the oracle the entire way.

## 2. Session-DB compatibility contract

For a true **in-place upgrade** (a user's existing `sessions.db` + `sessions/<id>/`
just keep working), the C++ core must honor this contract **byte-for-byte**, not
merely "the same schema on paper." Sourced from `app/store.py`, `app/paths.py`,
`app/backup/`.

**Database file**
- `sessions.db`, SQLite 3, **`PRAGMA journal_mode=WAL`** (keep the `-wal`/`-shm`
  sidecars and their semantics).
- Location from `data_dir()`: `WHISPERX_DATA_DIR` → macOS `~/Library/Application
  Support/WhisperX` → `app/data/` (dev default). The C++ core resolves the path
  identically.

**Schema — three tables, exact DDL** (`store.py:36-64`)
- `sessions` (15 columns, in order): `id` (TEXT PK), `filename`, `audio_filename`,
  `status` (NOT NULL), `stage`, `error`, `options`, `language`, `diarized`,
  `model`, `num_segments`, `duration`, `translations`, `created_at` (NOT NULL),
  `updated_at` (NOT NULL).
- `settings` (`key` TEXT PK, `value`).
- `speaker_names` (`session_id`, `speaker_key`, `name`; **composite PK**
  `(session_id, speaker_key)`).

**Migrations — same idempotent pattern**
- Reproduce `_migrate`: `PRAGMA table_info(sessions)` → add `stage` / `translations`
  via `ALTER TABLE … ADD COLUMN` if absent (SQLite has no `ADD COLUMN IF NOT
  EXISTS`). Run the *same* check at open so a DB created by **either**
  implementation is forward- and backward-compatible.

**Value encodings that must match exactly**
- `status` ∈ {`queued`, `running`, `done`, `error`}.
- `stage` ∈ {`decoding`, `transcribing`, `loading_align`, `aligning`,
  `diarizing`} or NULL.
- `options`, `translations` — **JSON text** (parse on read). `translations`
  shape: `{lang: {status, service?, error?}}`.
- `diarized` — INTEGER 0/1 (map to bool on read).
- `created_at` / `updated_at` — **ISO-8601 UTC, seconds precision**
  (`isoformat(timespec="seconds")`).

**CRUD + lifecycle parity**
- `create / mark_running / mark_stage / mark_duration / mark_done / mark_error /
  get / list` (`ORDER BY created_at DESC, id DESC`) `/ delete` (cascades
  `speaker_names` + `rmtree`s the session dir) `/ reconcile_startup`
  (`running|queued → queued` on boot, to requeue work interrupted by a crash).

**On-disk artifacts** under `sessions/<id>/`
- `audio.*` (immutable; relative name in `audio_filename`), `transcript.json`
  (canonical result). Overlays written **atomically (tmp + rename)**:
  `transcript.edits.json`, `transcript.translation.<lang>.json`. Exports:
  `transcript.{srt,vtt,txt,json}`. **The writers must stay byte-identical** —
  enforced by a golden test (§3). *(Landed: the edits/undo overlay + translation
  files, and the `app/edits.py` turn algorithms behind them, are ported to the C++
  store under the `edits` token — see the Phase 1 brief; the rich translation v2
  payload shape stays Python. Exports/writers remain Phase 5.)*

**Backup / restore**
- Must keep working: **SQLite Online Backup API** for `snapshot_db` (never copy
  the live file), atomic `swap_db` + reopen, the manifest layout under
  `backup/`. The C++ DB layer exposes the same snapshot/swap primitives.

**Secrets stay out of the DB**
- OS keyring, service `manuscript-whisperx` (`hf_token`, Google creds/key), with
  env-var overrides. The C++ core reads the **same** keyring entries.

**Recommended library: SQLiteCpp.** A thin RAII wrapper over the genuine SQLite C
API — so the on-disk **file format is guaranteed identical** — with
transactions and prepared statements. Prefer it over `sqlite_orm`, whose
schema-synchronization would fight our hand-rolled `_migrate` and risk emitting a
subtly different schema. JSON columns via **nlohmann/json**.

## 3. Headless test + timing suite

Two concerns, one harness. It mirrors the existing pytest patterns (timestamp
validity, monotonic ordering, output completeness, `approx()` float compares,
synthetic `_make_emission`/`_seg`/`_word` fixtures) — but adds the thing the
repo lacks today: real measurement.

### 3a. Correctness — golden-parity tests

The repo has **no per-stage golden data files today**; the unit tests synthesize
inputs and mock the models. (It *does* now have a **Playwright end-to-end suite** —
`app/web/tests/e2e/` — that drives the real Svelte SPA against a seeded Flask
backend over the JSON/SSE API; that's an **integration oracle**, complementary to
the per-stage goldens below: once C++ stages swap in behind `WHISPERX_CORE_STAGES`,
the same e2e run validates the whole pipeline through the real UI, not just stage
diffs. See §6 / the Phase 5 brief.) The port still needs stage-level ground truth,
so we add a **golden-vector generator**: a
script/pytest pass that runs the *real* Python pipeline over a small fixed clip
set and dumps each stage's intermediates to JSON —

- merged VAD chunks (`vads/vad.py::merge_chunks`),
- CTC emissions + trellis + backtrack path + `merge_repeats`
  (`alignment.py:425-541`),
- word timings (`SingleWordSegment`: `word/start/end/score`),
- diarization turns + `assign_word_speakers` labels (`diarize.py:185`),
- the final writer outputs (`transcript.{srt,vtt,txt,json}`).

C++ tests assert each stage matches **within tolerance** — timings `≈` epsilon;
**trellis path, speaker labels, and writer bytes exact**. Because pybind11
exposes the C++ functions directly, the existing pytest oracle can call them and
diff in-process — no second harness.

Per-stage isolation tests (direct ports of what exists):
- `get_trellis` / `backtrack` / `merge_repeats` on synthetic emissions;
- `interpolate_nans` — `nearest` fills, `ignore` preserves NaNs;
- `IntervalTree` query correctness **and** O(log n) scaling vs a linear scan;
- `merge_chunks` boundary behavior;
- writer formatting (exact strings).
- End-to-end: `decode → … → write` on a golden clip equals Python's
  `transcript.json`.

### 3b. Timing — per-stage benchmarks

Today the only timing in the codebase is the **loose RTF constants**
(`pipeline.py:184-202`, `STAGE_RTF = {transcribing: 0.22, aligning: 0.19,
diarizing: 0.51}`) used for UI ETA hints — **no actual wall-clock measurement
anywhere**. The C++ suite measures it for real: wrap each stage in
`std::chrono::steady_clock` (or the bench framework's timer), report per-stage ms
and **RTF = stage_seconds / audio_seconds**, and gate regressions in CI. This
does double duty: it *validates the cpp-core streamlining claims* (decode-once,
batched alignment, models-resident) with numbers, and it *replaces the guessed RTF
constants with measured ones* that feed the real ETA. (Stage pipelining is
deferred — not measured.)

### 3c. Framework choice

- **Catch2 v3** for the unit/parity suite — one dependency, expressive
  `REQUIRE`/`CHECK`, and a **built-in `BENCHMARK(...)` block** so correctness and
  light timing live in the same file. (GoogleTest + GoogleMock is the heavier
  alternative if we want rich mocking; doctest if compile time is paramount.)
- **Google Benchmark** (or header-only **nanobench**) for the *serious* per-stage
  timing — warmup, calibration, statistics — in a separate `bench/` target that
  doesn't slow the test loop.
- Both wired into **CTest** and run in CI alongside the pybind oracle, next to
  the existing `python-compatibility.yml`.

## 4. C++ libraries — Python-counterpart replacements

What each Python dependency maps to. This is also the answer to "are there
libraries that make our life easier or replace the Python bits?" — most of the
heavy Python stack collapses into **ONNX Runtime + a handful of header-only
libs**.

| Python today | C++ replacement | Note |
|---|---|---|
| `faster-whisper` / `ctranslate2`, `pyannote`, `silero`, `torch` (four ML stacks) | **ONNX Runtime + sherpa-onnx** | one runtime; VAD + ASR + diarization off-the-shelf |
| `torch`/`transformers` wav2vec2 + numpy Viterbi | **ORT model + own Viterbi** (ref: torchaudio `forced_align`; the C++ kernel in `MahmoudAshraf97/ctc-forced-aligner`) | the one DIY model — no clean standalone lib in any language |
| `ffmpeg` **subprocess** (`audio.py:44`) | **ffmpeg *libraries*** (`libav*`), linked in-process (Option B) | one universal decode path (incl. M4A/AAC/MP4/video); decode to float in-memory; **no subprocess**; LGPL build |
| `numpy` arrays · `torch.stft` mel (`audio.py:112`) | **Eigen** (or **xtensor** for an n-d, numpy-feel API) + KissFFT/pocketfft | the mel can also be baked into the ORT graph |
| `pandas` DataFrames (`alignment.py:325,395`; `diarize.py:170`) | **plain structs + std loops** | DataFrames are incidental scaffolding, not needed |
| `nltk` punkt (`alignment.py:189`) | **ICU** sentence/word break (or a small rule-based splitter) | |
| `sqlite3` (`store.py`) | **SQLiteCpp** | same on-disk format, RAII (see §2) |
| `json` stdlib | **nlohmann/json** | header-only, STL-native |
| `pytest` | **Catch2 v3** (+ CTest) | parity + unit (see §3) |
| *(none — RTF guesses)* | **Google Benchmark / nanobench** | real per-stage timing (see §3) |
| Flask JSON API + SSE (`app/`, serving the Svelte SPA) | **Drogon / oat++ / cpp-httplib** | only **if/when** the host is swapped — *out of scope here*; the C++ server must then reproduce the **JSON `/api/*` + SSE contract** the SPA already depends on |
| `keyring` | OS keychain APIs (same service id) | secrets parity |

The net: the dependency graph shrinks from dozens of Python/transitive packages
to roughly **ORT + sherpa-onnx + an audio decoder + Eigen + nlohmann/json + your
code**.

## 5. Build tooling — standardize on CMake (+ Ninja + a package manager)

**Yes, ditch Make for the C++ core — adopt CMake.** Raw Makefiles can't express
what this project needs: a single build that targets the desktop OSes (Windows,
macOS, Linux), fetches native dependencies, and produces three different
artifacts (a shared lib, a pybind extension, a server binary) from one tree.
Reasoning:

- **CMake is the lingua franca** of C/C++ and — decisively here — it's what our
  whole dependency set already ships: **ONNX Runtime, sherpa-onnx, SQLiteCpp,
  Catch2, Google Benchmark, nlohmann/json, pybind11** are all CMake-native (most
  offer `find_package` / `FetchContent` / `add_subdirectory`). Consuming them
  with hand-written Makefiles means reverse-engineering each one's flags.
- **Cross-platform builds are a first-class concept** via **toolchain files** —
  the standard mechanism for per-OS desktop builds; Make has no equivalent and
  would need bespoke per-platform glue.
- **Package managers integrate cleanly** — **vcpkg** (manifest mode,
  `CMAKE_TOOLCHAIN_FILE`) or **Conan**. Pick *one* per the usual rule; **vcpkg**
  is the lighter default given our libs are all in its registry. This replaces
  the "manual dependency vendoring" pain the cpp-core doc flagged as the main
  ergonomic cost.
- **Ninja as the generator** for fast incremental builds; CMake still emits
  Visual Studio / Xcode projects when an IDE or platform signing flow needs them.
- **CTest** ties the test + benchmark targets together (§3) and into CI.

Alternatives considered and why not (now):
- **Meson** — cleaner syntax, genuinely nice, but **most of our dependencies and
  the ML runtimes publish CMake, not Meson**, so we'd be swimming upstream on the
  exact libraries that matter. Reasonable only if we owned the whole dep tree.
- **Bazel / Buck** — excellent hermetic, cached, multi-language builds, *but* the
  ML runtimes aren't Bazel-native, BUILD files for ORT/sherpa would be ours to
  maintain, and the setup cost only pays off at monorepo scale. Revisit only if
  this grows into a large polyglot monorepo.
- **Plain Make** — fine for a single-platform leaf binary; **wrong for a
  multi-OS, dependency-heavy, multi-artifact core**. Ditch it for the engine.

Recommendation: **CMake + Ninja + vcpkg**, per-OS toolchain files, CTest for
tests/benches. Keep the existing Python `uv` workflow and the Tauri
`cargo` build as-is — CMake governs only the new C++ core, and the pybind
artifact is what the Python side consumes.

## 6. Phased migration roadmap

Compat-aware refinement of the cpp-core roadmap. **Detailed per-phase execution
briefs** (context · goals · validation · unknowns) live in
[`cpp-core-migration-briefs.md`](./cpp-core-migration-briefs.md).

| Phase | Goal | Exit criteria |
|---|---|---|
| 0 | scaffold + golden generator + **decision gate** | `whisperx_core` importable; goldens dumped for the **EN/DE/RU** clip set (pinned lib + model revisions, transcript stored separately); ASan/LSan + CMake + Ninja + vcpkg + CTest + Catch2 green |
| 1 | full `store.py` in C++ (SQLiteCpp): the SQLite layer (`db`) **+** the file-backed edits/undo + translation sidecars & `app/edits.py` algorithms (`edits`) | full `store.py` API parity; round-trips a **real pre-existing** `sessions.db` + `transcript.edits.json`; migrations idempotent; backup snapshot/swap pass; the two stage tokens compose |
| 2 | decode-once (ffmpeg libs) + VAD / `merge_chunks` | **2A landed:** `merge_chunks` in C++ behind the `vad` token; chunks == golden on a *fixed* raw-segment input (decoupled — see briefs). **2B pending:** in-process `libav*` decode + ORT silero (`decode` token); PCM sample-parity + bench RTF |
| 3 | **alignment** — batched wav2vec2 (ORT) + Viterbi — *highest risk, early* | word-timing parity within tolerance; trellis path exact (fed Python ASR text) |
| 4a | **ASR backends** (sherpa-onnx ORT first; whisper.cpp/GGML follow-on) | each backend gated by **WER/CER** (not byte-equality); `device` selection wired |
| 4b | **diarize + `assign_word_speakers`** | assign glue exact on fixed turn sets; diarization quality A/B |
| 5 | writers + end-to-end | writers byte-identical **on a fixed transcript**; `run_job` per-stage parity |
| 6 | timing gates in CI | per-stage benchmarks tracked; regression budget enforced |

Phases 0–1 deliver the safety net (oracle + DB compat) before any model is
touched; 3 retires the biggest risk; 5 proves end-to-end parity; the strangler
flag (`WHISPERX_CORE_STAGES`) lets each phase ship behind a toggle. **4a/4b are
independent sub-tracks** and need not ship together. Validation follows the
**decoupled-golden** rule (briefs §"How to read", fact 3): downstream stages tested
on a fixed transcript, ASR judged by WER.

## 7. Risks & verification

| Risk | Mitigation |
|---|---|
| **DB drift** breaks in-place upgrade | §2 contract as a checklist **plus** a test that opens a real pre-existing `sessions.db` and asserts read/write/migrate parity |
| **wav2vec2 ONNX export drift** | validate in Phase 3; pin opset; golden emission tests |
| **Writer byte-drift** | exact-match golden test on all four formats |
| **Stage parity regressions** | the `WHISPERX_CORE_STAGES` flag keeps Python runnable for side-by-side diffing at every phase |
| **Build complexity (5-platform cross-compile)** | CMake toolchain files + vcpkg; Ninja; keep the core small and pure |

**Overall verification:** the golden-parity suite (Catch2 via pybind) and the
benchmark suite (Google Benchmark) both run in CI beside
`python-compatibility.yml`; a "real DB" fixture proves the in-place upgrade; the
pybind oracle keeps `app/` + `tests/` authoritative throughout.

## 8. References

- Architecture decision: [`cpp-core-spa-architecture.md`](./cpp-core-spa-architecture.md)
- Per-phase briefs: [`cpp-core-migration-briefs.md`](./cpp-core-migration-briefs.md)
- Stage-by-stage spec: [`pipeline-reference.md`](./pipeline-reference.md)
- Why C++ + ORT: [`single-language-runtime-options.md`](./single-language-runtime-options.md)
- SQLiteCpp: <https://github.com/SRombauts/SQLiteCpp> · nlohmann/json: <https://github.com/nlohmann/json>
- Catch2: <https://github.com/catchorg/Catch2> · Google Benchmark: <https://github.com/google/benchmark> · nanobench: <https://github.com/martinus/nanobench>
- CMake: <https://cmake.org/> · vcpkg: <https://vcpkg.io/> · ONNX Runtime: <https://onnxruntime.ai/docs/> · sherpa-onnx: <https://github.com/k2-fsa/sherpa-onnx>
- ctc-forced-aligner (C++ kernel reference): <https://github.com/MahmoudAshraf97/ctc-forced-aligner>
