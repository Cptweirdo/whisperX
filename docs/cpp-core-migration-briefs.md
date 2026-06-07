# C++ Core Migration — Per-Phase Briefs

> **Implementing this?** Start with the [`cpp-core-handoff.md`](./cpp-core-handoff.md)
> kickoff prompt, then read "How to read these briefs" below.
>
> Execution briefs for each phase of the [`cpp-core-migration-plan.md`](./cpp-core-migration-plan.md)
> roadmap (§6). One brief per phase, each with **Context** (why, what it depends
> on, the Python code it maps to), **Goals** (concrete deliverables),
> **Validation** (how we prove it done/correct), and **Unknowns / open questions**
> (the genuine risks and decisions to resolve before/while building it).
>
> Stage specs they reference live in [`pipeline-reference.md`](./pipeline-reference.md);
> the DB contract in [`cpp-core-migration-plan.md`](./cpp-core-migration-plan.md) §2;
> the streamlining rationale in [`cpp-core-spa-architecture.md`](./cpp-core-spa-architecture.md) §4.

## How to read these briefs

Four cross-cutting facts shape every phase:

1. **Strangler-fig, Python stays the host.** Until Phase 5, `app/` + the Python
   `whisperx` pipeline keep running. Each C++ stage is swapped in **behind the
   pybind11 `whisperx_core` module** and gated by `WHISPERX_CORE_STAGES`, so we
   always have a runnable reference to diff against. A phase is "done" when its
   stage matches Python on the golden set with the flag on.
2. **Alignment is sequenced before ASR (Phase 3 before 4) for risk, not for data
   flow.** Alignment *consumes* transcript text, which ASR produces — but in the
   strangler model **Python ASR feeds the C++ aligner** during Phase 3 (it also
   supplies the detected `language` that selects the wav2vec2 model), so we can
   retire the highest-risk stage (wav2vec2 export + Viterbi) early without C++ ASR
   existing yet. The dependency is satisfied by the oracle.
3. **Decoupled goldens (committed decision).** Whisper ASR output is **not
   byte-stable across engines** (CTranslate2 today vs sherpa-onnx vs whisper.cpp),
   so end-to-end byte-parity is impossible by design. Therefore: **test
   align/diarize/writers against a *fixed* transcript input** (the exact text Python
   used, stored as a golden artifact) — any difference is then a real bug — and
   **validate the ASR stage separately by WER/CER**, not identical text. Goldens
   store the intermediate transcript as its own file so downstream tests read it as
   input.
4. **The Python seam is already pinned by unit tests — use them as the porting
   oracle.** `tests/test_pipeline_contract.py` (added pre-port; runs torch-free in
   ~0.1 s) nails down the *exact* contract between the host (`app/`) and the engine
   (`whisperx/`) at the `app/pipeline.py` boundary the strangler swaps behind:
   `run_job`'s stage-progress sequence, the calls it makes into the engine and with
   which arguments, the result dict it returns, the artifacts it writes, and
   cancellation semantics — plus the **writer byte-goldens** (srt/vtt/txt verbatim,
   json round-trip) and the constants the API depends on (`WhisperModel` whitelist,
   `DEVICES`, `OUTPUT_FORMATS`, `WRITER_OPTIONS`, `eta_seconds`). Treat these as an
   **executable spec**: a C++ stage is wired correctly when, swapped in behind
   pybind with the flag on, it keeps this suite green. Per-phase Validation sections
   point at the specific cases. (The `app/store.py` API surface for Phase 1, and the
   JSON/SSE routes for the eventual host swap, are likewise pinned by
   `tests/test_api.py` / `tests/test_sse.py` and documented in
   [`api-reference.md`](./api-reference.md).)

**Tolerances** (used throughout "Validation"): structural shape and integer
indices **exact**; speaker labels and trellis/backtrack paths **exact**; writer
bytes **exact *given identical input*** (not end-to-end through a different ASR);
floating timings within **~1 frame (~20 ms)**; emission/score tensors within a
small **fp32 epsilon**. The exact epsilon/frame numbers were a **Phase-0
deliverable**, now **measured** (not guessed): **`emission_atol = 0.006`** (2× the
observed ORT-vs-torch wav2vec2 drift; see the Phase 0 brief + `tolerance_report.json`),
timings ±1 frame, scores ±0.01.

---

## Phase 0 — scaffold + golden generator + decision gate

### Context
Nothing can be validated until two things exist: a **C++ module Python can call**
(the migration vehicle *and* the parity oracle) and a **set of golden vectors**
to compare against. The repo has **no golden data today** — tests synthesize
inputs and mock models — so ground truth must be manufactured from the live
Python pipeline. This phase builds the harness, not any real stage — **but it is
also the decision gate**: several choices made here (datasets, tolerances, version
pinning, golden structure) bind every later phase, so under-scoping Phase 0 as
"just CMake setup" is a trap. Build tooling is settled in the plan (§5): **CMake +
Ninja + vcpkg**.

### Goals
- A CMake project producing a **`whisperx_core` pybind11 module** that imports
  cleanly from Python (`import whisperx_core`), even if its first functions are
  stubs.
- **CMake + Ninja + vcpkg** bootstrap with the dependency manifest seeded
  (ONNX Runtime, sherpa-onnx, SQLiteCpp, nlohmann/json, Catch2, Google Benchmark,
  Eigen) — pulling and linking at least one (e.g. nlohmann/json) end-to-end.
- A **golden-vector generator** — a Python script/pytest pass that runs the real
  `whisperx` pipeline over the fixed clip set (below) and dumps every stage's
  intermediate to JSON: merged VAD chunks (`vads/vad.py::merge_chunks`), CTC
  emissions + `get_trellis`/`backtrack`/`merge_repeats` (`alignment.py:425-541`),
  word timings (`SingleWordSegment`), diarization turns + `assign_word_speakers`
  labels (`diarize.py:185`), and the four writer outputs. **Per decoupled-goldens
  (fact 3): the intermediate transcript is dumped as its own artifact** so
  downstream-stage tests read it as fixed input.
  - **First cut landed: `tests/test_baseline_golden.py`** — an opt-in
    (`RUN_BASELINE=1`) pytest that runs **transcribe → align → diarize** over the
    golden clips and writes `golden/baseline.json` (per-clip hypothesis, WER/CER vs
    reference, segment/word/speaker counts, and **per-stage wall-clock + RTF**).
    Diarization uses the **vendored** community-1 checkpoint (token-free). This is
    the timing + end-to-end baseline; the per-stage **tensor** intermediates are
    dumped separately (next bullet). A first run on CPU
    (`tiny`/`int8`, 8 clips, 9 min) confirmed: **en+de align via the torchaudio
    loader, ru via the HF loader (Cyrillic path exercised)**; diarization dominates
    CPU RTF (~2.6 of ~2.9); tiny-model speaker counts drift (5/5/3 vs true 4) on the
    no-overlap synthetic dialogs — scored against the RTTM ground truth, not asserted
    exact.
  - **Per-stage tensor dump landed: `golden/dump_goldens.py`** → `golden/intermediates/`.
    Captures, per clip, the intermediates the C++ core is diffed against by **wrapping
    the real `merge_chunks` / `get_trellis` / `backtrack` / `merge_repeats`** (zero
    duplication, the exact pipeline values): merged **VAD chunks**, the **fixed
    transcript** as its own artifact (decoupled-goldens input), per align-segment **CTC
    emissions** (fp32 `.npz`) + tokens/blank_id + integer **backtrack path** +
    **char-segments**, and final **word timings** (+ speaker labels for dialogs). The
    `trellis` matrix is **not** stored (deterministically recomputable from
    emission+tokens; the parity gate downstream is the path/char-segments) — keeps the
    set at **~2.3 MB**, committable without LFS. `manifest.json` pins versions + sha256
    of every artifact + the **starting tolerance budget** (`emission_atol 1e-3`, ±1
    frame, `score 0.01`; tokens/path/char-segments/merged-chunks/speaker-labels EXACT).
    Guarded by **`tests/test_golden_intermediates.py`** — torch-free sha256 + shape
    consistency check (33 cases, ~0.1 s), skips when intermediates absent.
- **CTest + Catch2** wired; **ASan + LeakSanitizer on** from the first build (per
  the memory decision); one trivial parity test green to prove the oracle loop.
- A **CI job** skeleton beside `.github/workflows/python-compatibility.yml` that
  builds the module and runs CTest.
- **Adopt the existing seam-contract suite as the first oracle (fact 4).**
  `tests/test_pipeline_contract.py` already encodes the host↔engine contract and
  the writer byte-goldens; the writer goldens + constants port straight to Catch2,
  and the `run_job`-level cases become the pybind diff harness as stages land. No
  new ground truth to invent here — it exists and is green.

### Decisions to lock here (exit criteria, now resolved)
- **Dataset (resolved): English + German + Russian**, all from **Mozilla Common
  Voice *Spontaneous Speech*** (CC0, `sps-corpus-3.0`), plus **LibriSpeech** for an
  English single-speaker clip (CC BY 4.0). Generated by two committed scripts in
  `golden/`:
  - **`fetch_datasets.py`** — pulls/normalizes single-speaker clips (16 kHz mono)
    for **ASR/align**: `en_libri` (EN, torchaudio loader) and `ru_cv_*` (RU,
    **HF align loader + Cyrillic** non-Latin path). One clip is also re-encoded to
    **m4a** to exercise ffmpeg decode.
  - **`synthesize_dialog.py`** — CV-SPS clips are single-speaker, so multi-speaker
    **diarization** audio is **synthesized**: it concats clips from N **distinct
    speakers** round-robin (A,B,C,D,A…) into one track and emits a **ground-truth
    RTTM** + per-turn transcript. Produces `en_dialog` / `de_dialog` / `ru_dialog`
    (**4 speakers, ~60–67 s each**). Deterministic via `--seed` → byte-identical,
    so the output is committable as a pinned golden.
  All EN/DE hit the **torchaudio** align loader; RU hits the **HF** loader +
  **Cyrillic**.
  *Gaps accepted:* (1) **No real-overlap diarization** — the synthetic dialogs are
  clean **sequential** turns (no overlapping speech, cuts at clip boundaries), so
  only the no-overlap diarization + `assign_word_speakers` path is exercised; a real
  overlapping-speech corpus (e.g. AMI) can be added later if the overlap path needs
  coverage. (2) the `ja`/`zh` no-spaces path (`LANGUAGES_WITHOUT_SPACES`) is **not**
  covered by this set.
- **Golden determinism (resolved): pin everything.** Pin library versions and HF
  **model revision hashes** in the manifest; goldens are committed files,
  regenerated only via the documented script (a deliberate, reviewed act).
- **Diarization goldens (resolved): use the committed checkpoint.** The gated model
  is already **vendored** at `app/models/speaker-diarization-community-1.3533c8cf/`
  (`vendor_diarize_model.py`, `app/diarize_model.py`) → golden-gen runs offline,
  **no HF token, even in CI**. The per-dialog **RTTM** emitted by
  `synthesize_dialog.py` is the **ground-truth** speaker timeline (free, exact from
  concat offsets); the parity golden is whatever the vendored pyannote checkpoint
  produces on that audio — the RTTM lets us also sanity-check it against truth.
- **Tolerance budget (resolved as a deliverable): measure, don't guess.** Run the
  same model through torch vs ORT, set epsilon just above observed drift (start:
  emissions `atol ≈ 1e-3`, timings ±1 frame, scores ±0.01).
- **Goldens decoupled (resolved):** intermediate transcript stored separately;
  downstream stages tested on fixed text; ASR tested by WER/CER (fact 3).

### Validation
- `import whisperx_core` succeeds in the project venv. **Done:** the pybind module
  (`adapters/py/whisperx_core.cpp`) builds to `build/whisperx_core.*.so` and imports.
- `cmake --build` + `ctest` (with ASan/LSan) green locally and in CI. **Done:** root
  `CMakeLists.txt` (FetchContent for pybind11/Catch2/nlohmann-json; light deps pulled
  with no vcpkg bootstrap, `vcpkg.json` seeds the heavy ones), Catch2 suite green
  under **ASan+UBSan**, and CI in `.github/workflows/cpp-core.yml` (builds + CTest +
  parity, no heavy `whisperx` sync). *Note:* sanitizers run on the native test target,
  not the dlopened pybind module (would need `libasan` `LD_PRELOAD`).
- Goldens exist for the EN/DE/RU clip set, committed under `golden/` with a
  manifest (clip → pinned lib + model revisions → sha256), transcript stored as a
  separate artifact. **Done:** `golden/` holds the clips + `manifest.json`; the
  generator scripts (`fetch_datasets.py`, `synthesize_dialog.py`) and the
  end-to-end baseline (`tests/test_baseline_golden.py` → `golden/baseline.json`,
  with per-stage timings) are committed; the **per-stage tensor intermediates**
  (`golden/dump_goldens.py` → `golden/intermediates/`: merged VAD chunks, CTC
  emissions, backtrack path, char-segments, words, transcript-as-input — guarded by
  `tests/test_golden_intermediates.py`) are committed. The **tolerance budget is
  measured** (`golden/measure_ort_tolerance.py` → `tolerance_report.json`): the
  torchaudio wav2vec2 (`WAV2VEC2_ASR_BASE_960H`) exports to **ONNX opset 17**
  cleanly and runs under ORT; **ORT-vs-torch emission drift = max 2.8e-3 / mean
  2.3e-4**, and cross-process **torch-vs-torch run noise ≈ 2.9e-3** (float
  reduction-order), so exact emission byte-parity is impossible — only `atol`
  comparison is valid. `emission_atol` set to **0.006** (2× max observed drift) in
  `manifest.json`. This also **front-loaded the Phase-3 risk**: the wav2vec2 ONNX
  export works and drift is small, so the alignment ONNX path is de-risked early.
- The trivial parity test demonstrates a C++ function result matching Python.
  **Done:** `bindings/test/test_core_parity.py` — the first ported algorithm,
  `edit_distance` (WER/CER, lifted verbatim from the Python baseline), matches the
  Python reference bit-for-bit across 11 cases incl. Cyrillic tokens; this is the
  live strangler-fig oracle loop later stages plug into.

### Unknowns / open questions (remaining)
- **pybind in CI** — *largely answered by 2B.* The heavy native deps build via **CMake
  FetchContent + system ffmpeg with no vcpkg bootstrap**, in a **separate** `cpp-core.yml`
  job (`audio-stage`) that caches the slow sherpa-onnx build; the fast lane stays dep-free.
  Still open for prod packaging: the manylinux + vcpkg wheel build-time budget (Phase 5).
  Build gotcha for any ORT-using stage: pybind picks the right Python headers only with
  `-DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE=<interpreter-with-dev-headers>`.

---

## Phase 1 — session store in C++ (SQLiteCpp + file-backed sidecars)

### Context
The session DB is the **in-place-upgrade compatibility contract** (plan §2). It's
low-ML-risk and high-value to port early: it exercises the pybind boundary on
something deterministic and locks down the must-not-break surface before any model
work. Source of truth: `app/store.py` (schema `:37-72`, `_migrate` `:86-92`, CRUD,
`snapshot_db`/`swap_db`), `app/paths.py` (`data_dir()`), `app/backup/`.
**Scope reality (verified against the current code):** `SessionStore` is *not* just
CRUD — it exposes **~40 public methods** spanning four areas, two of which are
**file-backed** (JSON sidecars, not SQLite): (1) session CRUD/lifecycle, (2) the
**edits/undo** subsystem (`load_result`/`load_edits`/`current_segments`/
`save_turn_edit`/`save_turn_reassign`/`undo_turn_edit` over `edits.json`), (3) the
**translations** subsystem (`save_translation`/`load_translation`/`get_translations`/
`set_translation_status` over `translation/<lang>.json`, added after this brief was
first written — commit `8d9f7ef`), and (4) **settings**/`speaker_names`. So "port the
DB" really means "port the whole session-persistence layer, including its file I/O."

### Goals
- A C++ `SessionStore` on **SQLiteCpp** reproducing the **exact** schema (3 tables,
  `sessions` 15 columns in order), `PRAGMA journal_mode=WAL`, and the
  `PRAGMA table_info` + `ALTER TABLE ADD COLUMN` idempotent migration (`stage`,
  `translations`).
- Full CRUD + lifecycle parity: `create / mark_running / mark_stage /
  mark_duration / mark_done / mark_error / get / list` (`ORDER BY created_at DESC,
  id DESC`) `/ delete` (cascade `speaker_names` + remove session dir) `/
  reconcile_startup`.
- Value-encoding parity: `options`/`translations` as JSON text (**nlohmann/json**),
  `diarized` INTEGER 0/1, timestamps **ISO-8601 UTC seconds precision**.
- `snapshot_db` (SQLite **Online Backup API**) + atomic `swap_db` + reopen.
- **Edits/undo subsystem parity** (file-backed): `load_result` / `load_edits` /
  `current_segments` / `edit_history_len` / `save_turn_edit` / `save_turn_reassign` /
  `undo_turn_edit`, reading/writing `result.json` + `edits.json` with the same
  `edits.group_turns` semantics the SPA depends on (the turn-index contract — see
  CLAUDE.md). This is real logic (an undo stack + turn regrouping), not a CRUD wrapper.
- **Translations subsystem parity** (file-backed): `translation_path` /
  `save_translation` / `load_translation` / `get_translations` / `set_translation_status`
  over `translation/<lang>.json` + the `translations` JSON column.
- **Path/util helpers**: `session_dir`, `audio_path`, `artifact_path`, `result_path`,
  `edits_path`, `translation_path`, `rename`, `has_active_jobs`, `close`.
- A pybind binding that **fully replaces** `app/store.py` (decided — not a shadow):
  it must reproduce the **entire** API surface `app/` calls (every method + exact
  signatures), and all callers re-point to it. **Done-bar = full API parity** across
  all four areas above (CRUD + edits/undo + translations + settings/speaker_names) —
  **enumerate every `store.py` public method first** (it is ~40, not the ~10 CRUD
  methods, and several do JSON-sidecar file I/O, not just SQLite).
  - *Cleanup first:* `get_setting`/`set_setting` are currently **defined twice** in
    `SessionStore` (`:411/:418` dead, `:442/:449` live — a merge artifact); dedupe in
    Python before porting so the C++ surface isn't ambiguous.

### Status — landed (full session-persistence layer)
Phase 1 is **done** — the DB layer **and** the file-backed sidecars + `app/edits.py`
algorithms are ported and parity-tested. The decisions below were taken before
building; the DB layer landed first (DB-only), then the completion slice finished
the phase. The two slices share one strangler pattern with two composable flags.

**Scope: the whole of `app/store.py`.** The C++ `SessionStore` owns **SQLite**
(CRUD/lifecycle, `settings`, `speaker_names`, the `translations` *column*, WAL-safe
snapshot/swap) **and** the file-backed subsystems — the **edits/undo overlay**
(`transcript.edits.json`; the algorithm IP in `app/edits.py`: `group_turns` /
`coalesce_segments` / `apply_turn_edit` / `apply_turn_reassign` / `undo_last` /
`realign_words` / `_interpolate_gaps`, a difflib token-diff + gap interpolation,
ported to `core/edits/edits.{hpp,cpp}`) and the **per-language translation files**.
Only the pure **path helpers** stay on the Python facade.

**Swap mechanic: Python facade, two composable flags.** `app.store.SessionStore` is
a facade: its DB-method group forwards to `self._db` and its file-method group to
`self._edits`, each chosen at construction — `_PyStore` (today's Python, the default
+ parity oracle) or `whisperx_core.SessionStore` (C++), per **`db`** / **`edits`** in
`WHISPERX_CORE_STAGES` (either alone, or `db,edits`; one shared C++ store when both).
`app/edits.py` is **also** a facade so `group_turns`/`distinct_speakers`/
`next_speaker_key` (imported directly by `server.py`/`render.py`) route through the
same C++ implementation. Callers are unchanged; both backends stay runnable for
side-by-side diffing — the strangler seam. These are the first stages to use the
`WHISPERX_CORE_STAGES` flag (none existed before Phase 1).

**JSON parity decided: semantic (parse-on-read).** §2 says `options`/`translations`
are JSON text parsed on read, so nlohmann emits valid (compact) JSON and parity is
compared on **parsed values**, not byte layout. (Python `json.dumps` keeps spaces +
insertion order; nlohmann is compact + sorted keys — both round-trip identically.)

**What exists now**
- `core/db/session_store.{hpp,cpp}` — the C++ store on **SQLiteCpp** (bundled
  SQLite amalgamation via FetchContent → WAL + Online Backup API guaranteed, no
  vcpkg/system-SQLite needed). Verbatim schema + idempotent `stage`/`translations`
  migration; ISO-8601 UTC-seconds timestamps (`...+00:00`); a `std::mutex`
  serializing the DB + a separate `files_lock_` for the sidecar writes (mirrors the
  Python locks). **Plus** the file methods (`load_result`/`load_edits`/
  `current_segments`/`save_turn_edit`/`save_turn_reassign`/`undo_turn_edit` +
  translation I/O), delegating to `whisperx::edits`.
- `core/text/sequence_matcher.{hpp,cpp}` — the verbatim CPython difflib port
  (`find_longest_match`/`get_matching_blocks`, autojunk-off case only).
- `core/edits/edits.{hpp,cpp}` — the `app/edits.py` algorithm port (compiled
  `-ffp-contract=off`). `core/time_iso.hpp` — the shared `now_iso()` helper.
- `adapters/py/whisperx_core.cpp` — binds `SessionStore` (31 methods) + the 8 edits
  free functions, the 3 constants, `matching_blocks`, and a registered `NoChange`
  exception, all via the `nlohmann::json ↔ py::object` casters.
- `core/tests/{test_session_store,test_sequence_matcher,test_edits}.cpp` — 34 Catch2
  cases under ASan/UBSan (DB CRUD/lifecycle/ordering/settings/speakers/translations/
  snapshot-swap/migration; difflib block structure incl. the >200 no-autojunk case;
  edits exact-double borrow / untimed-word-has-neither-key / no-mutation / throws).
- `bindings/test/{test_store_parity,test_edits_parity,test_store_edits_parity}.py` —
  behavioural parity (Python vs C++, field-for-field minus volatile `ts`, **exact
  `==`** on interpolated floats, difflib-adversarial) **and** on-disk round-trip in
  **both** directions for `sessions.db` and `transcript.edits.json`.

### Validation — met
- Open a **real pre-existing `sessions.db`** and round-trip it through the other
  implementation — **no schema drift, no data loss**:
  `test_store_parity.py::test_{python_db_read_by_cpp,cpp_db_read_by_python}`.
- Migrations **idempotent** (legacy DB → open twice → no error/dup columns):
  Catch2 `migration adds stage/translations …`.
- **Field-for-field parity** (semantic JSON; timestamps checked for format, not
  equality since they're "now"): the 8 behavioural `test_store_parity.py` cases.
- `snapshot_db` → `swap_db` round-trip preserves data; **`tests/test_backup*.py`
  stays green against the C++-backed store** — verified by running the store suite
  with `WHISPERX_CORE_STAGES=db` (123 passed, same as flag-off).

### Unknowns — resolved
- **Threading** → the C++ store serializes every method with its own `std::mutex`
  (matching the Python `threading.Lock`); the pybind calls do **not** release the
  GIL this phase, so behaviour equals today's GIL-bound Python store. Composite
  reads (`get_translations` → `get`) take the lock only in the leaf to avoid
  re-entrancy. GIL-release for long ops (`snapshot_db`) is a later optimization.
- **JSON key-order / whitespace** → **semantic parity** chosen (see above); no
  consumer diffs the raw column text.
- **SQLite build** → **SQLiteCpp's bundled amalgamation** (`SQLITECPP_INTERNAL_SQLITE`),
  fetched via FetchContent — WAL + Online Backup API present, file-format identical
  to Python's `sqlite3`.
- **Secrets + settings** → the `settings` table is **ported to C++**;
  `secret_store.py` (keyring) **stays in Python** (out of the DB; §2 keeps secrets
  out of the DB anyway).
- **File-artifact ownership** → resolved: the file-backed sidecars + `app/edits.py`
  are **ported to C++** under the composable `edits` token (the completion slice
  below), so the **whole** session-persistence layer runs natively. Done, not open.

### Completion slice — file-backed sidecars + `app/edits.py` (landed)
This slice finished Phase 1 by porting the deferred file-backed subsystems and the
edit/undo algorithm IP to C++, so the **whole** session-persistence layer runs
natively behind the strangler flag with Python kept as the live oracle. The design
below is as built.

**Two decisions (locked):**
- **Route the server-side glue too.** `app/edits.py` becomes a facade (same pattern
  as `store.py`) so `group_turns` / `distinct_speakers` / `next_speaker_key` —
  imported **directly** by `app/server.py` (`_build_turns`, the speaker
  list/reassign/enroll endpoints) and `app/render.py` (markdown export), *not* via
  the store — delegate to C++ when on. This guarantees the store **and** the server
  compute turn-grouping with the *same* implementation, so the turn index the UI
  shows can't desync from the one an edit targets (the SPA turn-index contract).
- **New composable `edits` token**, independent of `db`. `WHISPERX_CORE_STAGES=db,edits`
  enables both; either can be validated alone (keeps Phase 1's independent-rollout
  property and lets the risky difflib port be flipped on by itself).

**Scope.** (A) the pure algorithms in `app/edits.py` (~250 lines): `group_turns`,
`distinct_speakers`, `next_speaker_key`, `coalesce_segments` (+ `_coalesce_run` /
`_merge_segments` / `_seg_dur` / `_seg_key`), `apply_turn_edit`,
`apply_turn_reassign`, `undo_last`, `realign_words`, `_interpolate_gaps`, the `Turn`
shape, `NoChange`, and `HISTORY_LIMIT=100` / `MIN_WORD_WIDTH=0.1` /
`SEGMENT_MIN_DURATION=0.2`. (B) the store's file methods: `load_result` /
`load_edits` / `current_segments` / `edit_history_len` / `save_turn_edit` /
`save_turn_reassign` / `undo_turn_edit` (+ `_baseline_segments` / `_original_segments`
/ `_write_edits`) and `load_translation` / `save_translation` (**opaque** atomic JSON
I/O — the rich v2 `{version,target_language,service,created_at,entries:{start_key:{src,tr}}}`
payload shape lives in `app/translate_job.py` / `app/translation_overlay.py` and
**stays Python**). Pure **path helpers** stay on the facade (Phase 1 convention).

**The difflib port is the highest risk — front-load it.** `realign_words` uses
`difflib.SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False).get_opcodes()`,
but downstream it only distinguishes `equal` (keep the old word + its `start`/`end`)
from `replace`/`insert` (untimed placeholder — handled identically) from `delete`
(drop). So the port needs only the **matched index-pairs** from difflib's *matching
blocks* — difflib's **greedy longest-contiguous match with a specific tie-break, not
a generic LCS**. Plan: a new `core/text/sequence_matcher.{hpp,cpp}` that ports
CPython `Lib/difflib.py`'s `find_longest_match` (the `j2len` rolling-map DP +
`besti/bestj/bestsize` tie-break: earliest `i`, then `j`, then longest) and
`get_matching_blocks` **verbatim for the `autojunk=False`, no-`isjunk` case only** —
**skip the autojunk popularity-pruning branch entirely** (its absence is observable
through `realign_words`). `get_opcodes` need not be ported; reconstruct
`realign_words` from blocks. Tokenization: `new_text.split()` (Python whitespace
split), `old_tokens = [(w.get("word") or "").strip() …]`.

**Implementation outline.**
- `core/text/sequence_matcher.{hpp,cpp}` (new) — the difflib port.
- `core/time_iso.hpp` (new) — extract `now_iso()` from `session_store.cpp`'s anon
  namespace for reuse by `edits.cpp` (delta `ts`).
- `core/edits/edits.{hpp,cpp}` (new) — `whisperx::edits` namespace; everything as
  `nlohmann::json` objects (preserves arbitrary keys like `score`/`speaker`/`avg_logprob`;
  value-copy == `deepcopy`). **Four correctness rules:** (1) **key presence** — use
  `.contains()`/`.find()`, never `operator[]` on reads; an untimed word carries
  **neither** `start` nor `end` key (not null); the equal-branch copies timing only
  when both are present + non-null. (2) **Float parity** — transcribe
  `_interpolate_gaps` statement-for-statement (the two-stage `take_l`/`take_r`
  min-chain is evaluation-order-sensitive) and compile this TU with **`-ffp-contract=off`**
  (no FMA) so the borrowing arithmetic matches Python's never-fused IEEE754 doubles
  bit-for-bit. (3) no input mutation. (4) negative `turn_index` → throw (→ Python
  `IndexError`). `group_turns` returns dicts `{index,speaker,start,end,seg_indices,text}`.
- `core/db/session_store.{hpp,cpp}` (extend) — the scope-B file methods; atomic write
  = dump to `<path>.tmp` + `std::filesystem::rename`; a **separate `files_lock_`**
  mutex distinct from the SQLite `lock_`; `current_segments`/`_baseline_segments` call
  `whisperx::edits::coalesce_segments`.
- `adapters/py/whisperx_core.cpp` — bind the 8 free functions, the 3 constant attrs, a
  registered `NoChange` exception, and the 9 new store methods (reusing the existing
  `json_to_py`/`py_to_json` casters).
- `CMakeLists.txt` — add the two `.cpp` to `whisperx_core_lib`, set `-ffp-contract=off`
  on `core/edits/edits.cpp` only (guard non-GCC/Clang), add the two new Catch2 files.
- **Python facades.** `app/store.py`: add `_core_edits_enabled()` (token `edits`);
  rename `_PyDbStore`→`_PyStore` (keep a `_PyDbStore = _PyStore` alias for the existing
  parity test), move the file/sidecar methods **into** it so it's a full-surface
  oracle; the facade builds a `_db` impl and an `_edits` impl and routes each
  method-group independently — **one shared C++ store when both tokens are on** (no
  duplicate SQLite connection), one `_PyStore` when both off, two objects in mixed mode
  (acceptable in the opt-in validation mode). `app/edits.py`: keep every function as
  `_py_*` plus `Turn`/`NoChange`/constants **public + unchanged**; each routed public
  function becomes a thin wrapper that calls C++ when `edits` is on and re-wraps
  `group_turns` dicts as `Turn(**d)`; `app/server.py`/`app/render.py` imports untouched.

**Validation — met.** The pure-stdlib **`tests/test_turn_edits.py`** oracle
(group/collapse/splice, the borrow-arithmetic incl. exact-float asserts, undo,
coalesce, reassign + `NoChange`, the store overlay, end-to-end) passes identically
with `WHISPERX_CORE_STAGES` unset, `edits`, and `db,edits` — a **CI matrix** proves
the tokens compose. *(Note: `tests/test_word_timestamp_interpolation.py` is an
`alignment.py` test of the same `"start" in word` key-presence contract, not an
`app.edits` test, so it's unaffected by this slice.)* New `bindings/test/`:
`test_edits_parity.py` (Python `_py_*` vs C++ per function, `_strip_ts` on deltas,
**exact `==`** on interpolated floats, **plus** a difflib section diffing C++
`matching_blocks` against `difflib.SequenceMatcher(autojunk=False).get_matching_blocks()`
on adversarial token lists — repeats, tie-breaks, a `>200`-element list confirming no
autojunk drift, unicode) and `test_store_edits_parity.py` (same edit sequence through
both stores; segments + `edits.json` + undo round-trip match; cross-read of
`edits.json` between implementations). Native Catch2 under ASan/UBSan:
`test_sequence_matcher.cpp`, `test_edits.cpp`. Results: **34/34 CTest**, the new
parity files green, the full `uv run pytest tests/` green flag-off (226 passed),
`import whisperx` clean.

**Risks (handled):** (1) **float bit-parity** of `_interpolate_gaps` → `-ffp-contract=off`
+ statement-order transcription + native exact-value gate (`test_edits.cpp`). (2)
**difflib fidelity** → verbatim port + adversarial parity test; the >200 no-autojunk
case passes. (3) **key-presence vs null** → `.contains()`/`.find()` only; native test
asserts untimed words carry neither key. (4) **two SQLite connections** when
`db`+`edits` both on → one shared C++ store instance (mixed-flag mode opens two, an
accepted validation-only inefficiency). (5) **`NoChange` identity across pybind** →
registered C++ exception; the `app/edits.py` facade translates it to `app.edits.NoChange`.

---

## Phase 2 — decode-once + VAD / merge_chunks

### Context
The first real pipeline stage and the first **streamlining win**: today audio is
decoded up to 3× (ASR, alignment, diarization) via an **ffmpeg subprocess**
(`audio.py:44`); we decode **once** to a shared float32 16 kHz mono buffer. Then
the VAD segments it and `merge_chunks` packs voiced spans into ≤30 s windows.
Source: `audio.py:25` (`load_audio`, `SAMPLE_RATE=16000`), `vads/vad.py:19`
(`Vad.merge_chunks`), `vads/silero.py`.

### Status — landed (2A `merge_chunks`/`vad`; 2B in-process decode + ORT silero VAD/`decode`)
Phase 2 is **split into two composable slices** (the Phase 1 pattern), **both landed**.
**2A** is the pure, dep-free algorithm IP; **2B** is the heavy in-process `libav*` decode +
ORT silero VAD.

**Slice 2A (landed).** `Vad.merge_chunks` is ported verbatim to
`core/vad/merge_chunks.{hpp,cpp}` (`whisperx::vad`, everything `nlohmann::json`), with
the audio hyperparameters in `core/audio/audio_constants.hpp`. It's wired into the
always-built `whisperx_core_lib` (no new deps) and exposed via pybind
(`merge_chunks(segments, chunk_size, onset, offset)` + the constants as module attrs).
`whisperx/vads/vad.py::Vad.merge_chunks` is now a **facade** (same shape as the
`app/edits.py` one): the original body is `_py_merge_chunks` (the oracle); the public
static routes to C++ when **`vad`** ∈ `WHISPERX_CORE_STAGES`, restoring the inner
`(start,end)` tuples so the result is byte-identical. Routing the one static covers
**both** Silero and Pyannote (they delegate to it).

**Decoupled VAD golden (the decision).** silero ≠ pyannote, and torch.hub JIT silero ≠
ORT/sherpa silero, so the live VAD segments are **not** byte-comparable across engines.
Per the decoupled-golden rule (fact 3) the **raw pre-merge segments** are now dumped as
a *fixed input* golden — `golden/dump_goldens.py`'s `_wrap_vad` spy records them, and a
new `--vad-only` mode surgically (re)writes just the `segments` field into each
`golden/intermediates/*.vad.json` (cheap: silero only, no whisper/align/diarize),
asserting the merge of the captured segments still reproduces the committed
`merged_chunks` before overwriting (so the pinned tensor goldens + measured tolerance
budget are untouched). `merge_chunks` is then gated **exactly** on that fixed input; the
model's own segments are judged only by a loose boundary tolerance (deferred to 2B).

**Validation — met.** `core/tests/test_merge_chunks.cpp` (6 Catch2 cases under
ASan/UBSan: the flush condition, the `curr_end-curr_start>0` guard, final append,
onset/offset-ignored, speaker-irrelevance). `bindings/test/test_vad_parity.py`:
per-function parity vs `_py_merge_chunks` on adversarial lists **plus** golden-replay
(the 8 `*.vad.json` raw segments → C++ → committed `merged_chunks`, exact). 40/40 CTest,
73 bindings green, full `uv run pytest tests/` green (226 passed) with
`WHISPERX_CORE_STAGES` unset, `vad`, and `db,edits,vad` (tokens compose); `import
whisperx` clean.

**Slice 2B (landed — `decode` token).** In-process `libav*` decode
(`core/audio/decode.{hpp,cpp}` + the shared `AudioBuffer`) replaces the ffmpeg subprocess;
`whisperx/audio.py::load_audio` facades to `whisperx_core.load_audio` under **`decode`**
(`_py_load_audio` is the oracle). The ORT **silero VAD** (`core/audio/vad_silero.cpp`,
**sherpa-onnx**) emits the raw segments 2A's `merge_chunks` consumes; `whisperx/vads/silero.py`
facades to it under `decode` (torch.hub stays default/oracle — decoupled, smoke-only). All
behind a **`WHISPERX_CORE_AUDIO`** CMake option (default **OFF** so the dep-free Phase-0/1/2A
build + fast CI are untouched): ffmpeg via `pkg-config` (system dev libs for the dev/CI build —
exact-version match to the CLI → bit-exact PCM; vcpkg ffmpeg for prod), sherpa-onnx vendored via
FetchContent (it brings its own ONNX Runtime). Pinned silero ONNX in `models/silero_vad.onnx`.

**2B validation — met.** `bindings/test/test_decode_parity.py`: C++ `load_audio` vs the
subprocess on **every** golden clip + a committed **m4a/AAC** — wavs **bit-exact**, m4a ≤2 LSB
(`atol=2/32768`). `bindings/test/test_vad_smoke.py`: the ORT silero path runs end-to-end and its
17 segments feed `merge_chunks` (decoupled — boundaries loose, not torch-parity).
`tests/test_pipeline_contract.py` decode-once contract green with `decode` on. 40/40 CTest; full
`uv run pytest tests/` green (226 passed, 15 skipped) across `WHISPERX_CORE_STAGES` ∈ {unset,
`decode`, `vad,decode`, `db,edits,vad,decode`}; `import whisperx` clean (facade `hasattr`-guarded).
`bench/bench_audio` RTF: decode ≈ 0.0015, VAD ≈ 0.003 (≪ 1) on the 60 s EN dialog.

**Two non-obvious 2B build gotchas (settled):** (1) sherpa-onnx bundles its own `nlohmann_json`
and unconditionally `add_subdirectory()`s it (target-name collision) → `CMakeLists.txt` guards
that one line so sherpa reuses our `nlohmann_json`. (2) **Link ORT shared, not static** — the
prebuilt static `libonnxruntime.a` (glibc2_17 toolchain) corrupts the heap (`free(): invalid
pointer` inside ORT `DeviceDiscovery`'s `std::regex`) when linked into a system-libstdc++ 13
binary; `BUILD_SHARED_LIBS ON` in sherpa's scope selects sherpa's shared/C-API ORT path
(internals stay behind `libonnxruntime.so`'s C boundary).

### Goals
- **In-memory decode** to float32/16 k/mono via the **ffmpeg libraries**
  (`libavformat` + `libavcodec` + `libswresample`) **linked in-process — no
  subprocess** (replacing `load_audio`/`audio.py:44`). One decode path for **all**
  formats incl. WAV/MP3/FLAC/Ogg **and** the hard ones (M4A/AAC, MP4/MOV, MKV/WebM,
  video). **Decision: Option B** — universal ffmpeg-libs over a tiered
  dr_libs+ffmpeg split, for one consistent code path and decode/resample parity
  with the current (ffmpeg-based) pipeline.
- **silero VAD** through sherpa-onnx (ONNX Runtime).
- A faithful **`merge_chunks` port** (chunk_size, onset/offset, the voiced-span
  packing) producing the same chunk boundaries.
- A **shared, zero-copy buffer** abstraction the later stages slice into.
- First **bench RTF** numbers for decode + VAD.

### Validation
- Decoded PCM matches `whisperx.load_audio` output **sample-for-sample within a
  small tolerance** — and parity should be *tight*, since both use ffmpeg's own
  `swresample` (same decoder + resampler as today, just linked vs spawned).
- **Merged chunk boundaries == golden** (`vads/vad.py::merge_chunks`), exact.
- Decode-once verified: one decode call services all downstream consumers.
- Bench RTF for decode + VAD recorded for Phase 6 baselining.
- *Contract oracle:* `test_pipeline_contract.py` already pins the host-visible
  decode contract — `run_job` decodes **once** (the buffer is reused by every stage),
  reports `duration == len(audio)/SAMPLE_RATE`, and fires `on_duration` exactly once
  right after decode. Keep these green when the C++ decode swaps in.

### Unknowns / open questions
- ~~**VAD model mismatch**~~ **(resolved — decoupled golden, slice 2A).** Rather than
  golden a specific live VAD, the **raw pre-merge segments** are dumped as a *fixed
  input* (silero, via `dump_goldens.py --vad-only`) and `merge_chunks` is tested
  **exactly** on them; the VAD model's own segments are judged only by a loose boundary
  tolerance. So the silero-vs-pyannote / torch-vs-ORT segmentation difference no longer
  needs to be byte-reconciled — `merge_chunks` parity is engine-independent by
  construction.
- **ffmpeg build/license/size (Option B residuals)** — build a **default LGPL**
  ffmpeg (no `--enable-gpl`/`--enable-nonfree`; native AAC decoder, no `fdk-aac`;
  MP3 patents expired); dynamic-link or LGPL-compliant static. Open: **codec/format
  trimming** (strip muxers/encoders/video filters we never use to cut binary size),
  the final size budget, and a legal sign-off for distribution.
- **Decode determinism** — ffmpeg version pinned so decoded buffers (hence
  goldens) are reproducible across builds.
- **silero version/thresholds** — `vad_onset=0.500`, `vad_offset=0.363`,
  min_duration on/off — exact parity with the sherpa-onnx silero build?

### Slice 2B — in-process `libav*` decode + ORT silero VAD (`decode` token) — execution plan
The remaining Phase-2 work, and the first slice to pull a **heavy native toolchain**
(ffmpeg `libav*`, ONNX Runtime, sherpa-onnx) into the build. It replaces the ffmpeg
**subprocess** with linked libraries, decoding **once** to a shared float32/16 kHz/mono
buffer the later stages slice, and produces the raw VAD segments slice 2A's
`merge_chunks` already consumes. Gated behind a new composable **`decode`** token
(independent of `db`/`edits`/`vad`).

**Scope.**
- **(A) Decode** — `core/audio/decode.{hpp,cpp}` (`whisperx::audio`):
  `AudioBuffer load_audio(const std::string& path, int sr = kSampleRate)`. Open/demux
  with `libavformat` (`avformat_open_input` → `avformat_find_stream_info` →
  `av_find_best_stream(AVMEDIA_TYPE_AUDIO)`), decode with `libavcodec`
  (`avcodec_send_packet`/`receive_frame`), and resample/downmix/reformat with
  `libswresample` to **mono / 16 kHz / `AV_SAMPLE_FMT_S16`**, then `sample / 32768.0f`
  → float32. This reproduces the current subprocess (`-ac 1 -ar 16000 -f s16le -acodec
  pcm_s16le`, `audio.py:44-65`) so the PCM is **sample-for-sample** identical (same
  decoder + same `swresample`). One universal path for every format (WAV/MP3/FLAC/Ogg +
  M4A/AAC, MP4/MOV, MKV/WebM, video — demux the audio stream, ignore video).
- **(B) Shared buffer** — `core/audio/audio_buffer.hpp`: `struct AudioBuffer {
  std::vector<float> samples; int sample_rate = kSampleRate;
  std::span<const float> slice(size_t f0, size_t f1) const; }`. The "decode once" buffer
  every later C++ stage slices zero-copy (ASR chunks, align `audio[:, f1:f2]`). In the
  pybind seam `load_audio` returns a numpy float32 array (today's contract); the buffer
  matters end-to-end once Phases 3–5 are native.
- **(C) ORT silero VAD** — wrap sherpa-onnx's silero `VoiceActivityDetector` (ORT) to
  emit raw `{start, end, "UNKNOWN"}` segments, fed straight into
  `whisperx::vad::merge_chunks` (slice 2A). Thresholds mirror `default_vad_options`
  (`vad_onset=0.5`, window/hop per the sherpa silero config). Per the **decoupled**
  decision its segments are **not** chased to byte-parity with torch silero — only a
  loose boundary tolerance.
- **(D) pybind + facade** — bind `load_audio` (→ numpy) and the VAD entry; make
  `whisperx/audio.py::load_audio` a **facade** (the 2A pattern): route to
  `whisperx_core.load_audio` when **`decode`** ∈ `WHISPERX_CORE_STAGES`, keep the
  subprocess body as `_py_load_audio` (the oracle). `app/pipeline.py` is unchanged — it
  already calls `load_audio` once.

**Build / deps (the heavy part).**
- `vcpkg.json` — add **`ffmpeg`** with a trimmed feature set (LGPL default: no
  `--enable-gpl`/`--enable-nonfree`, native AAC decoder, no `fdk-aac`; keep only the
  demuxers/decoders we need — `mov`/`matroska`/`ogg`/`wav`/`mp3`/`flac`/`aac` etc. — and
  drop muxers/encoders/video filters to cut size). `onnxruntime` is already declared.
- **sherpa-onnx** is **not in the vcpkg registry** (see the `vcpkg.json` `$notes`) →
  vendor it / build from source via `FetchContent` or an `ExternalProject`, pinned to a
  release tag, linking its static VAD lib + ORT.
- `CMakeLists.txt` — a `WHISPERX_CORE_AUDIO` option (**default OFF** so the dep-free
  Phase-0/1/2A build and the existing fast CI are unchanged; **ON** under the vcpkg
  toolchain). When ON: `find_package(FFMPEG)` (`libavformat`/`libavcodec`/
  `libswresample`/`libavutil`) + ORT + sherpa-onnx, compile `decode.cpp` + the VAD TU
  into `whisperx_core`, and add a **`bench/`** target (Google Benchmark) for decode + VAD
  RTF.
- **CI** (`.github/workflows/cpp-core.yml`) — a **new vcpkg job** (manifest mode +
  binary caching to keep build time bounded) that configures with
  `WHISPERX_CORE_AUDIO=ON`, runs the decode-parity + a VAD smoke + the bench, and
  extends the stage-token matrix to include `vad` and `decode`. The existing
  FetchContent-only job stays as the fast lane.

**Validation (definition of done).**
- **PCM parity** — `bindings/test/test_decode_parity.py`: C++ `load_audio` vs
  `whisperx.load_audio` (subprocess) on **every** golden clip incl. the **m4a/AAC** one.
  Already-16 kHz-mono clips exercise pure decode + s16 conversion (expected
  bit-exact); the m4a exercises the AAC decoder + (no-op) resample. Tolerance tight
  (`atol` ≈ a couple LSB) to absorb only legitimate resampler/dither differences.
- **Decode-once** — assert one decode call services all downstream consumers; the
  host-visible contract (`run_job` decodes once, `duration == len(audio)/SAMPLE_RATE`,
  `on_duration` fires once) stays green with `decode` on
  (`tests/test_pipeline_contract.py`).
- **VAD smoke** — the ORT silero path runs end-to-end on a golden clip and its segments
  feed `merge_chunks` without error (quality is the loose-boundary check, not parity).
- **Bench RTF** for decode + VAD recorded (seeds Phase 6).
- Full `uv run pytest tests/` green with `WHISPERX_CORE_STAGES` ∈ {unset, `decode`,
  `vad,decode`, `db,edits,vad,decode`}; `import whisperx` clean.

**Suggested build order.** (1) `decode.cpp` + `AudioBuffer` + the `WHISPERX_CORE_AUDIO`
CMake/vcpkg plumbing + PCM-parity test — this is the deterministic, high-value half and
de-risks the toolchain. (2) the vcpkg CI job. (3) the ORT silero VAD + bench (depends on
sherpa-onnx vendoring, the looser half).

**Risks / open decisions — outcomes (2B landed).**
- **Resampler/dither parity** — *resolved.* Linking the **same `libav*`/`swresample`** the
  CLI wraps (default `swr`, no dither override) gives **bit-exact** PCM on the 16 kHz-mono
  wav goldens and **≤2 LSB** on a 44.1 kHz-stereo m4a (AAC decode + downmix + resample) —
  `bindings/test/test_decode_parity.py`. The dev/CI build uses the **system** ffmpeg-dev,
  whose version matches the CLI, so the engines are identical by construction.
- **Downmix coefficients** — *resolved* (covered by the m4a parity above: ffmpeg's default
  stereo→mono matrix on both sides).
- **silero model asset** — *resolved.* Pinned `models/silero_vad.onnx` (snakers4 silero v5,
  sha in `models/README.md`); loaded by sherpa's VAD. Decoupled, so the revision only needs
  to load (not byte-match torch silero).
- **sherpa-onnx vs hand-rolled VAD** — *resolved:* **vendored sherpa-onnx** via FetchContent
  (it brings its own ONNX Runtime). It builds heavy but cached; the hand-rolled fallback was
  not needed.
- **ffmpeg version pin + license/size** — partially open: the **system** ffmpeg pins the dev
  build; the **vcpkg ffmpeg** entry (LGPL feature-trim) + a distribution license sign-off are
  still the prod-packaging task (carry to Phase 5/packaging).
- **Heavy-deps build path** — *resolved & proven without vcpkg:* `WHISPERX_CORE_AUDIO=ON`
  builds ffmpeg (pkg-config) + sherpa/ORT (FetchContent) with **no vcpkg bootstrap**; CI caches
  the sherpa build. **Settled cross-cutting fact for Phases 3–4 (which also use ORT):** link
  ORT **shared, not the static archive** — the prebuilt static `libonnxruntime.a` (glibc2_17
  libstdc++) corrupts the heap (`free(): invalid pointer` in ORT `DeviceDiscovery`'s
  `std::regex`) inside a system-libstdc++ binary; `BUILD_SHARED_LIBS ON` (sherpa scope) selects
  the shared/C-API path. Also: sherpa bundles its own `nlohmann_json` — guard its
  `add_subdirectory` to reuse ours.

---

## Phase 3 — alignment: batched wav2vec2 + Viterbi (highest risk)

### Context
The **one stage with no off-the-shelf binding in any language** and the project's
core IP. A wav2vec2-CTC model produces frame emissions; a Viterbi trellis +
backtrack maps transcript characters to frames for word timings. Today it's torch
+ pandas + nltk and **explicitly un-batched** (`TODO` at `alignment.py:245`).
Source: `alignment.py:80` (`load_align_model`), `:117` (`align`), `:425`
(`get_trellis`), `:455` (`backtrack`), `:508` (`merge_repeats`), `:298-411`
(char→word→sentence + `interpolate_nans`), `:189` (nltk punkt), `:272-283`
(wildcard/OOV), `:32-77` (`DEFAULT_ALIGN_MODELS_TORCH/HF` — 5 + 38 languages).
**Do this early; it's where the migration most likely fails.**

### Status — slice 3A landed (Viterbi + assembly + native splitter, `align` token)
Phase 3 is **split into two composable slices** (the Phase 1/2 pattern). **3A** —
the pure, dep-free algorithm IP — is **landed**; **3B** (the heavy ONNX wav2vec2
forward) is the remaining slice. The settled facts:

- **The Viterbi + assembly are native (`align` token).** `core/align/trellis.{hpp,cpp}`
  ports `get_trellis`/`backtrack`/`merge_repeats`/`merge_words` verbatim over a row-major
  fp32 emission; `core/align/align.{hpp,cpp}` ports the char→word→sentence assembly
  (`alignment.py:298-411`) as struct loops (no pandas), with `core/align/interpolate.hpp`
  porting `interpolate_nans` (nearest/ignore) and the `groupby(["start","end"])` sentence
  merge. All in the always-built `whisperx_core_lib` — **no new deps**. Compiled
  `-ffp-contract=off`; `round3` uses `nearbyint` (banker's, matching Python `round()`).
- **The seam keeps the model forward in Python (3A scope).** `whisperx/alignment.py::align`
  is a facade: the torch forward + log_softmax + char-cleaning + **wildcard extension** +
  tokenization stay Python (unchanged), then the **fixed extended emission + tokens** cross
  to `whisperx_core.align_assemble` under `align` (the Python body is the live oracle).
  `align()`'s signature is unchanged → `tests/test_pipeline_contract.py` needs no edit.
- **nltk punkt → native splitter (the resolved unknown).** `core/text/sentence_split.{hpp,cpp}`
  is rule-based + vendored Moses `nonbreaking_prefixes` (`core/text/data/`, embedded at build
  time via a CMake-generated header; `core/text/utf8.hpp` codepoint cursor). **Evaluation
  recorded:** FreeLing rejected (GPLv3 + 100s-MB data), ICU rejected (heavy dep, UAX#29 not
  abbrev-aware → as divergent from punkt as a rule), a punkt port rejected (no C++ one;
  linking forces a Go/Rust toolchain; its only edge — byte-fidelity — is moot once divergence
  is accepted, and its case/adaptive machinery degrades on short ASR segments). The splitter
  **reproduces punkt exactly on every golden transcript**, so `words.json` needed **no
  re-baseline**; on an adversarial **en+ru+de+fr** corpus it agrees 80% (divergences
  documented + pinned). **German ordinals** ("1." = 1st, e.g. "Am 1. Januar") get a
  de-gated suppression rule (`sentence_split.cpp`): a ≤2-digit token before a starter is
  treated as non-breaking, so ordinals converge with punkt while 3-/4-digit year-final
  sentences ("…1990. Danach…") and bare decimals ("1.23") still split correctly. de/fr
  abbreviations + accented-capital starts are covered in `test_sentence_split.cpp`.
- **Downstream blast-radius of any splitter divergence (analysed):** only segment/paragraph/
  cue structure + interpolated edges of *untimed* words on multi-sentence segments — never
  aligned-word timing, never speaker labels, never transcript text. The abbreviation
  over-split (the one case with teeth) is neutralised by the Moses lists.
- **Gates met.** Viterbi **path + char_segments exact** vs the committed emissions on all 8
  clips (torch-free golden replay); `align_assemble` words match `words.json` within ±1
  frame / ±0.01; the splitter is pinned against a committed **punkt baseline**
  (`bindings/test/sentence_split_baseline.json` ← `golden/sentence_split_corpus.py`) with
  contract invariants asserted on every input; **62/62 CTest** under ASan/UBSan
  (`test_trellis.cpp`, `test_sentence_split.cpp`); `tests/test_word_timestamp_interpolation.py`
  green under `align`; full `uv run pytest tests/` green (226) across `WHISPERX_CORE_STAGES`
  ∈ {unset, `align`, `vad,align`, `db,edits,vad,align`}; dep-free build unaffected. The align
  goldens gained per-segment `dictionary` + `clean_cdx` (`dump_goldens.py --align-io`) for the
  torch-free replay.

**3B — landed (native ONNX wav2vec2 forward, `align_onnx` token).** The torch forward
(`alignment.py:278-285`) runs in C++ under a raw `Ort::Session`:
`core/align/wav2vec2_onnx.{hpp,cpp}` (pImpl; bucket-by-length + padded **attention_mask**
batched forward; trimmed by the graph's `frame_lengths` output) + pure
`core/align/emission_post.{hpp,cpp}` (log_softmax + OOV wildcard + tokenize — "path 2",
so the mirror-lang path is torch-free forward→post→assemble). `alignment.py` facades under
`align_onnx`: `load_align_model` pulls the mirror `.onnx`+`meta.json` → `Wav2Vec2Onnx`;
`align()` runs one batched forward over all segments, then per-segment `align_emission_post`
→ `align_assemble` (3A). Inherits the 2B build facts (ORT **shared**; sherpa-`nlohmann_json`
guard; decode-once `AudioBuffer::slice` = the per-segment read).

**The masking finding (the load-bearing 3B result).** Batched padded inference is parity-
safe **only for `layer_norm` feature extractors** (HF xls-r). The torchaudio base bundles
(en/de) use **`group_norm`, which normalizes each channel over time** → right-padding shifts
every valid frame's stats (measured eager drift ≈ **6**, ≫ `emission_atol`); *no* attention
mask recovers it (the norm already mixed padding in, and torchaudio's own `lengths` masking
makes it *worse*). So batching is **data-gated on `meta["batchable"]`**: layer_norm models
batch (mask + `frame_lengths`-trim); group_norm models run **per-segment** (batch 1, no
padding → exact). The expensive models (xls-r large) are the batchable ones, so the perf win
still lands. The mirror was **re-exported to contract v2** for this: dynamo exporter (the
legacy TorchScript tracer can't trace the mask/lengths op), opset **18**, inputs
`waveform`+`attention_mask`, outputs `emissions`+`frame_lengths`, `meta.batchable`.

**Gates met.** C++ forward+post reproduces every golden `seg{i}_emission` within
`emission_atol = 0.006` **per-segment and batched** + exact tokens, all 8 clips / both
loaders (`bindings/test/test_align_onnx_forward_parity.py`, `RUN_MIRROR=1`); `align()` under
`align_onnx` matches `words.json` **exactly** (en per-segment + ru 112-word batched dialog);
Catch2 `test_emission_post` (7 cases, incl. degenerate-input safety) under ASan/UBSan; 66/66
CTest; full `uv run pytest tests/` green (226) token-off; `import whisperx` + dep-free build
unaffected. *Known gap:* `<400-sample` segments are padded to the conv min + masked (correct
for batchable models now); none in the goldens.

**Model-asset mirror (landed, pre-3B).** Resolves the "per-language model proliferation /
which to bundle vs download" unknown for the *asset* side. **Run vs. produce are separate:**
ONNX is portable → C++/ORT runs wav2vec2 with no Python; only the one-time **conversion**
needs PyTorch, and that's **offline tooling**, never the runtime. We publish our **own**
parity-pinned export (public mirrors cover ~1 of our 3 and aren't `emission_atol`-validated
or hash-pinned) to **`KonstantK/wav2vec2-align-onnx`** (HF public repo, free + CDN, reusing
the `huggingface_hub` cache/sha/revision infra the original weights already use; R2 is the
only-if-we-drop-HF alternative). `golden/export_align_onnx.py` is **extensible by data not
code** — it imports `whisperx.alignment`'s `load_align_model` + `DEFAULT_ALIGN_MODELS_*`
tables and dispatches on `loader-type → wrapper` (torchaudio `model(x)[0]` / HF
`model(x).logits`), so `--lang <code>` exports any of the 43; `model.onnx` = **raw logits**
(opset 17, dynamic `{batch,time}` axes; consumer does log_softmax + the OOV wildcard column,
matching `alignment.py:285,295-302`); `meta.json` carries `dictionary` + `blank_id` for a
torch-free tokenizer. Idempotent (skips already-published unless `--force`), self-checked
vs the golden emissions pre-upload. The C++ 3B seam only consumes *a path to a parity-valid
`.onnx`* — producer swappable (torch-export now → CI-host later) without C++ change. Round
trip pinned by the opt-in `bindings/test/test_align_onnx_mirror.py` (`RUN_MIRROR=1`): every
golden `seg{i}_emission` reproduced under ORT from the **downloaded** onnx within
`emission_atol` across all 8 clips / both loader families. *Known gap:* the export omits the
torchaudio `lengths` input, so a **<400-sample** segment (the `alignment.py:270` masked
re-pad) would diverge — none in the goldens; revisit when 3B adds bucket+mask batching.

### Goals
- **Export wav2vec2-CTC to ONNX** (start `WAV2VEC2_ASR_BASE_960H`, English), pin
  opset, run on ORT.
- **Batched forward passes** across segments (pad to common length, one ORT batch)
  and/or **emissions computed over larger spans and sliced per segment** —
  addressing the `:245` TODO and the `<400-sample` re-pad (`:248`).
- Port **`get_trellis` / `backtrack` / `merge_repeats` / `merge_words`** to C++.
- Port **char→word→sentence assembly** as struct loops (no pandas) +
  **`interpolate_nans`** (`nearest` / `ignore`).
- Replace **nltk punkt** sentence splitting with **ICU** or a rule-based splitter.
- Port **wildcard/OOV char handling**.

### Parallelization (how alignment goes fast)
**Structural fact:** Phase 2 of `align()` (`alignment.py:207-411`) is **independent
per segment** — each segment reads a *disjoint, read-only* audio slice
(`audio[:, f1:f2]`, `:246`), runs its own wav2vec2 forward pass, builds its own
trellis, and writes its own output entry. There is **no cross-segment state**
(`segment_data` is precomputed in Phase 1; weights/dictionary/audio are read-only).
That independence is exactly why the `:245` TODO exists, and it gives three
parallelization axes, in payoff order:

1. **Batch the forward passes (the big win, the `:245` TODO).** Stack N segments
   into one ORT inference instead of one call per segment — the forward pass
   dominates cost, and per-call overhead + the `<400`-sample re-pad (`:248`) amortize
   away. Data-parallel on the GPU/NPU. Two **non-negotiable correctness rules**:
   - **Bucket by length.** Segments vary 1 s↔25 s; naive padding to a common length
     wastes most of the compute, so group similar lengths per batch.
   - **Mask padded frames.** Padding must not leak into emissions. The torchaudio
     path passes `lengths` (`:258`); the **HF path currently does not** (`:260`), so
     a batched port *must* add the attention mask or padded segments corrupt their
     batch-mates. This is both a perf detail **and a golden-parity gate**: batched
     inference is numerically ≈ per-segment **only if** masks/lengths are correct.
   - *Stronger variant:* compute emissions over **larger contiguous spans once and
     slice per segment** (adjacent segments re-run overlapping audio today). Care:
     transformer attention is global over its input window, so big-window emissions
     ≠ small-slice emissions at boundaries — use overlap.
2. **Thread-pool the per-segment CPU glue.** Trellis DP, `backtrack`, and char→word
   assembly are independent per segment → `std::for_each(std::execution::par, …)`
   over segments. (Python can't: GIL-bound. **This is a concrete C++-core win.**)
3. **The per-trellis time-scan stays sequential — and that's fine.** `get_trellis`
   (`:438-444`) is an inherent Viterbi recurrence (row `t+1` depends on row `t`);
   it's already **vectorized across the token dimension**, and a ≤30 s segment is
   ~1500 frames × tens of tokens — trivial. (Associative-scan Viterbi could
   parallelize the time axis in O(log n) depth, but it's unnecessary; the trellis is
   cheap vs the model.)

**Scope note:** this is **within-stage data parallelism** — distinct from, and far
safer than, the cross-stage **stage pipelining** deferred in
[`cpp-core-spa-architecture.md`](./cpp-core-spa-architecture.md) §4A. Batching the
aligner adds no mid-job complexity and is fully compatible with "one job at a time,
run to completion." It is the recommended Phase 3 approach: **batch inference
(bucket + mask) + thread-pool the Viterbi/assembly; keep the time-scan sequential.**

### Validation
- **CTC emissions** (post log-softmax) match golden within fp32 epsilon.
- **Trellis matrix and backtrack path exact**; `merge_repeats`/`merge_words`
  segments exact.
- **Word start/end within ~1 frame** of Python; `score` within epsilon.
- `interpolate_nans` matches: `nearest` fills, `ignore` preserves NaNs.
- The existing `tests/test_word_timestamp_interpolation.py` cases ported to Catch2
  and green (known-char timestamps present/ordered; unknown words still get spans).
- Per-language dictionary mapping correct for at least English + one HF language.
- *Contract oracle:* `test_pipeline_contract.py` pins the seam-level wiring around
  the aligner — it receives the **ASR-detected `language`** (not the requested hint)
  for model selection, the ASR's segments + the loaded align model/meta, and is
  forced to **`_align_device(device)`** (CPU on the Apple-Silicon backends, not the
  compute device). The C++ aligner must satisfy the same wiring; numerical parity is
  the per-stage golden above.

### Unknowns / open questions
- ~~**ONNX export fidelity**~~ **(resolved — 3B).** Exports cleanly via the **dynamo**
  exporter (legacy TorchScript can't trace the lengths/mask op; dynamo tracks the conv
  frame dim symbolically), opset **18**, single file (`external_data=False`). torch→ORT
  fp32 drift is well within `emission_atol = 0.006` on every golden (en 1.8e-3, de ≤9.8e-4,
  ru 8.3e-4).
- ~~**Dynamic shapes vs buckets**~~ **(resolved — 3B).** Dynamic `{batch, samples, time}`
  axes; bucket-by-length only to bound pad waste. Boundary frames are exact for batchable
  models (mask + `frame_lengths`-trim) — word edges match `words.json` exactly.
- ~~**Batching correctness**~~ **(resolved — 3B, with a twist).** Padded batch elements are
  numerically identical to single runs **only for `layer_norm` extractors**; `group_norm`
  (torchaudio base) normalizes over time so padding corrupts valid frames (drift ≈ 6) and
  no mask fixes it → those run **per-segment** (`meta["batchable"]` gates it). See the 3B
  status block's masking finding.
- ~~**nltk → ICU parity**~~ **(resolved — slice 3A).** Replaced punkt with a rule-based
  native splitter + vendored Moses `nonbreaking_prefixes` (not ICU/FreeLing/a punkt port —
  see the 3A status block for the evaluation). It reproduces punkt exactly on the golden
  transcripts (no `words.json` re-baseline) and the accepted adversarial-corpus divergences
  are pinned against a committed punkt baseline. The divergence only touches multi-sentence
  segment grouping (blast-radius analysed) — small and accepted.
- ~~**torchaudio vs HF dicts**~~ **(resolved — 3B).** Both families' dictionaries ship in
  `meta.json` (char→id, lower-cased) so tokenization (`align_emission_post`) needs no torch
  model; both are parity-tested (en/de torchaudio + ru HF) — exact tokens + emissions within
  atol on every golden.
- **Per-language model proliferation** — 5 + 38 models. Export pipeline for all?
  Which to bundle vs download? (Ties to the model-asset / download strategy.)

### Build/runtime inheritance from Phase 2B (don't rediscover)
The aligner's ONNX Runtime + audio access are already wired by 2B's
`WHISPERX_CORE_AUDIO` path — reuse it rather than standing ORT up again:
- **ORT comes through the vendored sherpa-onnx** (FetchContent, its own ORT) and **must be
  linked shared** — the prebuilt static `libonnxruntime.a` (glibc2_17 libstdc++) corrupts the
  heap (`free(): invalid pointer` in ORT `DeviceDiscovery`'s `std::regex`) inside a
  system-libstdc++ binary. `BUILD_SHARED_LIBS ON` in sherpa's scope selects the shared/C-API
  ORT. (sherpa exposes ORT C++ headers too if the aligner wants a raw `Ort::Session`.)
- **Decode-once is available:** `core/audio/audio_buffer.hpp`'s `AudioBuffer` + zero-copy
  `slice(f0,f1)` is exactly the `audio[:, f1:f2]` per-segment read (`alignment.py:246`) — the
  batched forward passes slice the one decoded buffer, no re-decode.
- **emission tolerance is measured, not guessed** — `emission_atol = 0.006` (Phase 0); ORT is
  fp32-`atol`-compared, never byte-equal.

---

## Phase 4 — ASR backends (4a) + diarize & assign (4b)

> **Split into two independent sub-tracks** — they share no code and can ship
> separately. 4a is two ASR engines; 4b is diarization + the assignment glue.

> **Slice `assign` LANDED** (the pure 4b glue, the 3A analog). Native
> `IntervalTree` + `assign_word_speakers` (`core/diarize/{interval_tree,assign_speakers}.cpp`
> in `whisperx_core_lib`, no deps) behind the **`assign`** token;
> `whisperx/diarize.py::assign_word_speakers` facades to it (pandas body is the
> oracle). The pyannote **DataFrame stays Python** → extracted to `(start,end,speaker)`
> turns at the facade; `speaker_embeddings` passthrough + the **mutate-and-return**
> contract (callers rely on in-place *and* the return) live in the facade. Exact-parity
> tie-breaks (`max` first-wins, `argmin` first-min, `searchsorted side='left'`),
> `-ffp-contract=off`. **Golden:** a new model-independent `*.assign.json`
> (`dump_goldens.py --assign`, align-only) from the **CC0 RTTM ground truth** + the
> Python assign oracle over pre-diarize segments — so the C++ replay
> (`bindings/test/test_assign_parity.py`) is torch/model-free. Gates: exact segment +
> word labels on all 3 dialogs; Catch2 `test_assign_speakers` (9 cases, ASan/UBSan);
> 78/78 CTest; pytest 226 green across `{unset, assign, db,edits,vad,align,assign}`.

> **Slice `asr` LANDED** (4a, first backend). Native **sherpa-onnx Whisper**
> (`core/asr/whisper_sherpa.{hpp,cpp}` in `whisperx_core_audio`, pImpl) wraps sherpa's
> high-level **`OfflineRecognizer`** (mel+encode+decode+detokenize+lang-ID built in,
> greedy) behind the **`asr`** token — reuses the vendored sherpa-onnx/ORT (2B), so it's
> a thin batched front-end over the VAD spans (no raw `Ort::Session`). faster-whisper
> stays the **default + WER/CER oracle**. `whisperx/asr_sherpa.py` (`SherpaWhisperPipeline`,
> sibling of `asr_mlx`/`asr_whispercpp`) reuses the VAD loop + `merge_chunks`, same
> `transcribe()->{segments,language}` contract; `asr.py::load_model` facades under
> `_core_asr_enabled()` (signature preserved → `test_pipeline_contract.py` untouched).
> **Assets:** `golden/mirror_whisper_onnx.py` re-hosts + sha-pins sherpa's pre-exported
> Whisper ONNX (already ONNX — no torch export) to a public repo we control
> (`KonstantK/whisper-onnx-sherpa`, **tiny/base/small published**), pulled via the 3B
> `huggingface_hub` cache; resolver order = local dir (`WHISPERX_SHERPA_WHISPER_DIR`) →
> mirror → **sherpa-onnx official release** (cached) for unmirrored models, so any model
> still loads. **Decoupled gate (fact 3):** `bindings/test/test_asr_sherpa_parity.py` runs
> the backend over the committed VAD spans (model-free) and asserts WER ≤ baseline + 0.30
> / CER ≤ baseline + 0.20, exact language detection, + facade glue shape — 15/15 on
> en/de/ru. `avg_logprob` best-effort. Gates: pytest 226 green across `{unset, asr,
> db,edits,vad,align,assign,asr}`; seam green under `asr`; dep-free build unaffected; CI
> torch-free `WhisperSherpa` smoke in `audio-stage`.

> **Slice `diarize` LANDED** (4b, the diarization model). Native **sherpa-onnx
> OfflineSpeakerDiarization** (`core/diarize/diarize_sherpa.{hpp,cpp}` in
> `whisperx_core_audio`, pImpl) = pyannote-segmentation-3.0 + **wespeaker_en_voxceleb
> CAM++** embedding (512-dim) + cosine **FastClustering**, behind the **`diarize`** token;
> pyannote `community-1` stays the **default oracle**. `whisperx/diarize_sherpa.py`
> (`SherpaDiarizationPipeline`) emits the **same `[segment,label,speaker,start,end]`
> DataFrame** (cluster int → `SPEAKER_xx`) so `assign_word_speakers` + callers are
> untouched; `return_embeddings` via a **second `SpeakerEmbeddingExtractor` pass** (the
> diarization result has none). `diarize.py::DiarizationPipeline` facades under
> `_core_diarize_enabled()` (`__init__` builds `self._impl`, `__call__` delegates).
> **Speaker count → FastClustering `num_clusters`** precedence `num→max→min→auto` (no
> native min/max range; `num_clusters` is a clustering *target*, final count may be fewer).
> **Assets:** `golden/mirror_diarize_onnx.py` re-hosts + sha-pins the two ONNX (already
> ONNX — no torch export; MIT + Apache-2.0) to `KonstantK/diarize-onnx-sherpa`; resolver
> order = local dir (`WHISPERX_DIARIZE_ONNX_DIR`) → mirror → sherpa official releases
> (cached). **A/B gate (not parity):** the synthetic dialogs are hard — community-1 itself
> scores DER ≈ 32/32/64 % with the wrong count on them; sherpa is comparable (≈ 43/32/45 %).
> `bindings/test/test_diarize_sherpa.py` gates **DER ≤ 0.70** vs CC0 RTTM (regression
> bound), ≥ 2 speakers, the assign-glue contract, embedding dim — 5/5 on en/de/ru. Gates:
> pytest 226 green across `{unset, diarize, db,edits,vad,align,assign,diarize}`; seam green
> under `diarize`; dep-free build unaffected; CI torch-free `SherpaDiarizer` smoke.
> **Still open:** only the **whisper.cpp/GGML** ASR backend (4a follow-on).

### Context
Swap the two remaining models. **Whisper** moves to a **pluggable ASR backend**
(default sherpa-onnx Whisper on ORT; **whisper.cpp/GGML for Apple-Silicon Metal** —
see §"Runtime & acceleration"); **pyannote diarization → sherpa-onnx**
(`pyannote-segmentation-3.0` + CAM++). Then port the speaker-assignment glue.
Source: `asr.py:31` (`WhisperModel`), `:106` (`FasterWhisperPipeline`), `:325`
(`load_model`), `:417-445` (`default_asr_options`); `diarize.py:14`
(`IntervalTree`), `:91` (`DiarizationPipeline`, default gated `community-1`),
`:185` (`assign_word_speakers`), `:170` (pandas DataFrame).
**Verified-against-code drift:** `DiarizationPipeline.__call__` now takes
`return_embeddings: bool = False` and its return type is
`Union[tuple[DataFrame, embeddings_dict], DataFrame]` (`:113`) — it can emit
**per-speaker embeddings** alongside the turns, not just the bare DataFrame the
brief originally assumed. The C++ port must mirror this optional second return
(and the `num/min/max_speakers` controls).

**Build inheritance from Phase 2B:** **sherpa-onnx is already vendored and building**
(FetchContent under `WHISPERX_CORE_AUDIO`, bringing its own ONNX Runtime) — 4a (sherpa
Whisper) and 4b (sherpa pyannote-seg + CAM++) reuse that integration, not a fresh ORT
setup. Heed the 2B settled facts: link **ORT shared, not static** (the glibc2_17
`libonnxruntime.a` → `std::regex`/heap crash), and the sherpa-`nlohmann_json` collision
guard. The diarization parity goldens still come from the **vendored**
`app/models/speaker-diarization-community-1.3533c8cf/` checkpoint (no HF token).

**Model assets — extend the 3B mirror pattern (decide-during).** 4a/4b need *runtime*
ONNX: sherpa **Whisper** (base ~150 MB … large ~3 GB), **pyannote-seg-3.0**, **CAM++**.
Small fixed models can be **committed** like 2B's `models/silero_vad.onnx`; the large
Whisper models can't, so reuse the **3B mirror mechanism** already in the tree —
`huggingface_hub` pull + cache, a pinned `meta.json`/`contract_version`, offline export
tooling (the "we host our own parity-pinned export" decision). sherpa publishes
pre-exported Whisper/pyannote ONNX in its model zoo; if we pin those, mirror+sha-pin them
the same way (don't hot-link an upstream URL). Whether to bundle the default model or
lazy-download on first run is the **Phase-5 packaging policy** (the *mechanism* —
`load_align_model`-style pull+cache — is already built for 3B; ASR/diarize reuse it).

### Goals — 4a (ASR backends)
- An **ASR backend interface** producing `TranscriptionResult` (`segments` text +
  `language` + `avg_logprob`) from the Phase-2 VAD chunks. Ship **sherpa-onnx
  Whisper (ORT) first** (default, cross-platform); add **whisper.cpp/GGML**
  (Metal/CUDA/CPU) as a follow-on (it drags in a Metal-shader/CMake build). The
  backend produces **text only** — word timestamps still come from the ORT wav2vec2
  aligner (Phase 3), never whisper.cpp's DTW. Likely **supersedes the current `mlx`
  backend** as the Apple path.
- Backend selection wired to the existing `device` setting
  (`cpu`/`cuda`/`mlx`/`whispercpp` → ORT vs GGML), with language detection.

### Goals — 4b (diarize & assign)
- **sherpa-onnx diarization** → turns (`start`/`end`/`speaker`), replacing the
  pyannote DataFrame with structs. **Preserve the optional `return_embeddings` path**
  (`diarize.py:113`): when requested, return per-speaker embedding vectors alongside
  the turns (the current Python API does), so callers relying on it don't break.
- Port **`IntervalTree`** (sorted arrays + binary search, O(log n)) and
  **`assign_word_speakers`** (dominant speaker by overlap duration, `fill_nearest`).

### Validation
- **Speaker labels == golden** (IntervalTree + assignment exact) — this is the
  reliably-exact part.
- `assign_word_speakers` matches on synthetic turn sets (isolation test,
  independent of which diarization model produced the turns).
- Language detection matches Python.
- Segment text compared to golden — **with a defined strategy** (see unknowns).
- Bench RTF for ASR + diarization recorded.
- *Contract oracle:* `test_pipeline_contract.py` pins the seam — ASR is called with
  `batch_size` + the requested `language` and returns `{segments, language}`; the
  diarizer receives `min/max_speakers` and its turns feed `assign_word_speakers`;
  with no diarizer the `diarizing` stage and assignment are skipped and
  `diarized=False`. A backend is wired right when these hold (text quality is the
  separate WER/CER gate, per fact 3).

### Unknowns / open questions
- *(Resolved by fact 3 — decoupled goldens.)* ASR text won't match byte-for-byte
  across the three decoders; each ASR backend is gated by **WER/CER**, and
  downstream stages are tested on a fixed transcript. No "segment text == golden".
- **`default_asr_options` parity** — beam size, temperature fallback, patience,
  suppress tokens, `chunk_size=30`, `batch_size`. Which map onto sherpa-onnx, and
  which don't exist there?
- **GGML build cost (4a follow-on)** — Metal-shader compilation + one more CMake
  dependency; confirm acceptable before adding the whisper.cpp backend.
- **Diarization model differs (A/B, not parity)** — Python default is `community-1`
  (**vendored locally** at `app/models/speaker-diarization-community-1.3533c8cf/`,
  so golden-gen needs **no HF token**); sherpa uses **pyannote-seg-3.0 + CAM++**.
  Different models → different turns, so **`assign_word_speakers` is golden-tested in
  isolation** on fixed turn sets (realistic turns from the vendored model, and/or
  synthetic), while end-to-end speaker quality is an **A/B**, not a parity check.
- **`avg_logprob` semantics** — used downstream/UI; does sherpa expose an
  equivalent, and is the scale comparable?
- **num/min/max speakers** params — parity of the speaker-count controls.
- **Batched inference parity (inherit the 3B gate)** — if 4a batches VAD chunks through
  Whisper (or 4b windows pyannote-seg), apply the 3B finding: **a model that normalizes
  over the time axis corrupts right-padded batch-mates, and no attention mask recovers
  it** (the torchaudio `group_norm` result — eager drift ≈ 6 ≫ atol). *Prove* batched ==
  single-segment within tolerance before trusting a batched path; gate batching on that,
  per model. Whisper's fixed 30 s mel padding likely makes it uniform-length (low risk),
  but verify rather than assume — it's a golden-parity gate, not just perf.

---

## Phase 5 — writers + end-to-end

### Context
Port the output writers and assemble the **full `run_job` parity** path, so a C++
end-to-end run reproduces Python's `transcript.json` and exports byte-for-byte.
This is where the strangler flag can flip fully on. Source: `utils.py:443`
(`get_writer`), writers for srt/vtt/txt/tsv/json/aud; `SubtitlesProcessor.py`
(line splitting); `conjunctions.py` (per-language); `app/pipeline.py:506-584`
(`run_job`, `OUTPUT_FORMATS`, `WRITER_OPTIONS`, the `_stage(...)` progress calls).

### Goals
- Port the **writers** (srt/vtt/txt/json at minimum; tsv/aud as needed) as C++
  string formatting.
- Port **`SubtitlesProcessor`** line splitting (`highlight_words`,
  max line width/count, conjunction-aware splits via `conjunctions.py`).
- Assemble the full **`AlignedTranscriptionResult`** through a C++ orchestrator
  mirroring `run_job`.
- Bridge **stage progress events** (`decoding`/`transcribing`/`loading_align`/
  `aligning`/`diarizing`) to `mark_stage` + the SSE broker.
- Write `transcript.json` + exports via the **atomic tmp+rename** convention.
- **No mid-job cancellation** (per the memory decision): the C++ orchestrator
  mirrors `run_job` but **ignores `cancel_event` mid-stage** — cancellation, if
  honored at all, is checked only at stage boundaries.

### Validation
- **Writer outputs byte-identical to golden** when fed the **fixed golden
  transcript** (fact 3) — *not* through a live C++ ASR whose text differs. The
  verbatim srt/vtt/txt strings (+ json round-trip) already live in
  `test_pipeline_contract.py::test_writer_output_is_byte_identical_to_golden`; the
  C++ writers must reproduce them exactly (and the Catch2 port carries the same
  constants).
- `run_job` from a **fixed transcript** reproduces Python's `transcript.json`
  (structure + labels exact; timings within ~1 frame). Full-pipeline runs (live
  ASR) are validated per-stage, not by end-to-end `transcript.json` byte-equality.
- Progress events fire in the **correct order** and update the session row's
  `stage` exactly as Python does — pinned end-to-end by
  `test_pipeline_contract.py` (the `decoding→transcribing→loading_align→aligning→
  diarizing` sequence, the result dict `run_session`/`mark_done` consume, the
  deterministic artifact names, and **stage-boundary-only cancellation** — the C++
  orchestrator must keep this whole suite green with the flag on).
- The `app/` UI shows a C++-produced session indistinguishably from a Python one —
  **assert this with the existing Playwright e2e suite** (`app/web/tests/e2e/`, which
  drives the real Svelte SPA against the JSON/SSE API): run it with
  `WHISPERX_CORE_STAGES` off (Python) and on (C++) against the same seeded clip; the
  SPA-rendered turns/timings/exports must match. This is the integration gate on top
  of the per-stage goldens, and the point where the strangler flag can flip fully on.
  *(Note: the SPA's **turn data** — `server.py::_build_turns` → `group_turns` — is
  already served by C++ when the Phase-1 `edits` token is on, so this e2e run also
  exercises that seam; add `edits` to the on-flag set when validating the full host.)*

### Unknowns / open questions
- **Float formatting / byte-identical JSON** — `transcript.json` byte-equality
  requires matching Python's float `repr`/rounding for timestamps and scores.
  Achievable, but needs an explicit number-formatting decision (and the SRT/VTT
  timecode rounding must match exactly too). *(Where the arithmetic itself must
  match bit-for-bit — not just the printed form — reuse the Phase-1 finding:
  compile that TU with `-ffp-contract=off` to defeat FMA, so doubles match
  CPython's never-fused IEEE-754. See `core/edits/edits.cpp` in the Phase 1 brief.)*
- **`WRITER_OPTIONS` exact values** — `highlight_words`, `max_line_width`,
  `max_line_count`, segment/sentence splitting, and the **per-language conjunction
  lists** all affect output bytes; every option must be mirrored.
- **`SubtitlesProcessor` edge cases** — long words, missing timings (NaN words),
  CJK/no-space languages, overlapping segments.
- **Progress bridging across pybind** — calling a Python SSE/`mark_stage` callback
  from a C++ worker thread: GIL acquisition, ordering vs the durable DB write
  (CLAUDE.md notes the DB row is the source of truth; broker carries deltas).
- **Which formats are in-scope for v1** — json/srt/vtt/txt certainly; tsv/aud?

---

## Phase 6 — timing gates in CI

### Context
Replace the **guessed RTF constants** (`pipeline.py:184-202`,
`STAGE_RTF = {transcribing: 0.22, aligning: 0.19, diarizing: 0.51}`,
`eta_seconds`) with **measured** per-stage wall-clock numbers, gate regressions in
CI, and use the data to *prove the streamlining claims* (decode-once, batched
alignment, models-resident) rather than assert them. *(Stage pipelining is
deferred — not a claim to measure here.)*

### Goals
- A **`bench/` target** (Google Benchmark or nanobench) timing each stage with
  warmup/calibration/statistics, separate from the correctness suite.
- Per-stage **ms + RTF** (`stage_seconds / audio_seconds`) emitted in a
  machine-readable form.
- A **CI job** that records the numbers and **gates regressions** against a budget.
- Feed **measured RTF** back into `eta_seconds` (replace the hardcoded constants).
- Quantified before/after for the streamlining wins (decode 3×→1×; batched vs
  per-segment alignment; resident vs reload).

### Validation
- Benchmarks run green in CI and emit per-stage RTF across the clip set.
- A regression budget is enforced (build fails on >X% slowdown) **without
  flakiness**.
- Documented decode-once and batched-alignment improvements vs the Python baseline.
- `eta_seconds` uses measured constants; UI ETAs track reality.

### Unknowns / open questions
- **CI timing noise** — shared runners are noisy; how do we gate without flaky
  failures? Relative budgets, multi-run medians, a dedicated/self-hosted runner,
  or trend-tracking instead of hard gates?
- **Baseline** — compare against the **Python pipeline on the same runner**, or an
  absolute RTF target? The former needs both stacks runnable in CI.
- **Device matrix** — CI realistically covers desktop CPU (maybe one GPU). Mobile
  NNAPI/ANE and desktop CUDA/DirectML can't be CI-gated — do those get manual
  benchmark runs, and how are their RTFs captured for ETA?
- **Apple-Silicon acceleration bench** — the decisive measurement from
  §"Runtime & acceleration": **diarization CoreML-EP vs CPU** (heaviest stage,
  loses MPS under ORT), plus whisper.cpp-Metal ASR and CPU alignment. Manual
  Apple-Silicon bench run; decides whether to ship the CoreML-EP ORT build.
- **Cold vs warm** — first-run model load vs steady state; which RTF feeds the ETA,
  and how is model-load time accounted separately (it's the `loading_align` stage)?
- **Surfacing measured RTF** — bake constants at build time, ship a config table,
  or measure-and-adapt at runtime per device?

---

## Runtime & acceleration (committed decision + findings)

> **Decision.** ORT/sherpa-onnx is the runtime for **VAD, alignment, diarization**;
> **ASR is a pluggable backend** — sherpa-onnx Whisper (ORT) as the cross-platform
> default, **whisper.cpp/GGML (Metal)** as the Apple-Silicon path. "One runtime"
> means *one runtime for the non-ASR stages plus a pluggable ASR engine*, and
> **one runtime ≠ one accelerator.**

### Why ASR stays pluggable (not folded into ORT)
ORT has **no Metal EP**; on Apple GPU its only lever is the **CoreML EP**, and
Whisper is awkward under CoreML (autoregressive decode, dynamic shapes), so
sherpa-onnx Whisper on Mac tends to fall to **CPU**. **whisper.cpp/GGML has
hand-tuned Metal kernels** for Whisper and is **already in production** in the app
(the `device` setting already carries `whispercpp`/`mlx`). Dropping it to satisfy
"one runtime" would be a real Mac regression. GGML is Whisper-specific, so it
**cannot** run the other models — they stay on ORT. The ASR backend produces
**text only**; word timestamps always come from the ORT wav2vec2 aligner. In the
C++ core, whisper.cpp/GGML likely **supersedes the `mlx` backend**, so Apple-Silicon
ASR consolidates onto one engine (net: `{ORT, whisper.cpp}`, *simpler* than today's
`{cpu, cuda, mlx, whispercpp}`).

### Apple-Silicon acceleration map (the finding)
Default reality: sherpa-onnx usually ships **CPU-only ORT**, so the non-ASR stages
are **CPU unless** ORT is built with the CoreML EP *and* it's enabled (even then
coverage is partial — unsupported ops fall back to CPU). Stage by stage, against
what the current Python app does on Apple Silicon (`_torch_device` → MPS for
VAD/diarize; `_align_device` **forces alignment to CPU** via the MPS conv1d limit):

| Stage | RTF | Today (Apple Silicon) | New core (ORT) | Verdict |
|---|---|---|---|---|
| **VAD** (silero) | tiny | torch→MPS | CoreML-EP / CPU | **Non-issue** — trivial model |
| **ASR** (Whisper) | 0.22 | whisper.cpp Metal / mlx | **whisper.cpp Metal (kept)** | **Best path preserved** |
| **Alignment** (wav2vec2) | 0.19 | **already CPU** (MPS conv workaround) | ORT **CPU** | **Parity — no regression**; CoreML-EP an optional upside |
| **Diarization** (pyannote-seg + CAM++) | **0.51** | torch→**MPS** | CoreML-EP / **CPU** | **The one to watch** (heaviest stage) |

**The one open item: diarization.** It's the heaviest stage and the only one that
loses today's GPU (MPS) under ORT. Whether the **CoreML EP** accelerates
pyannote-seg/CAM++ or it runs CPU is **empirical** → a Phase 4/Phase 6 benchmark.
Levers if CPU is too slow: (1) build/enable the **CoreML EP** for the non-ASR
models; (2) **native Core ML conversion** of the diarization models (ANE, the
WhisperKit/SpeakerKit route — heavier); (3) **accept CPU** (one job at a time on a
desktop may be fine).

### Open questions
- **Diarization on Apple Silicon** — CoreML EP vs CPU for pyannote-seg + CAM++:
  benchmark (Phase 6), decide whether to ship the CoreML-EP ORT build.
- **CoreML-EP ORT build** — does the sherpa-onnx/ORT we bundle include the CoreML
  EP, or do we build it? Affects VAD/align/diarize acceleration headroom.
- **wav2vec2 via CoreML EP** — optional alignment speedup on Mac (transformer
  encoder is more CoreML-friendly than Whisper decode); benchmark, no-regression
  fallback to CPU.
- **GGML build cost** — Metal-shader compilation + one more CMake dependency.

## Memory management (committed decision)

> **Decision.** Jobs run **to completion** — no mid-job cancellation, no streaming
> stop. Memory is managed by **(1) memory-mapping read-only model weights**,
> **(2) a per-job arena for working buffers that is reset at job end**, **(3) a
> fragmentation-resistant allocator**, and **(4) leak sanitizers in CI**. No exotic
> memory library, no manual peak-budgeting. Primary target: **desktop standalone
> and (eventually) a server, ≥ 8 GB RAM, one job at a time** (serial `JobQueue`,
> as today).

### Why this is enough (the rationale)

The pipeline's memory splits cleanly into two kinds, and only one kind is ours to
manage:

| Kind | What | Strategy | Cost to us |
|---|---|---|---|
| **Read-only model weights** | Whisper (int8 large-v3 ≈ 1.5 GB; small ≈ 0.5 GB), wav2vec2 align (≈ 360 MB/lang), diarization (≈ 30–40 MB), silero (≈ 2 MB) | **`mmap`** the model files (ORT/ggml do this by default) | ~nil — file-backed, OS-shared, evictable under pressure; **not dirty heap we own** |
| **Working buffers (dirty)** | decoded audio (**~64 KB/s** → 1 hr ≈ 230 MB, 3 hr ≈ 700 MB — the one big variable), mel, ORT activation arena, CTC emissions, Viterbi trellis, result/output structs | **`std::pmr` arena, reset per job** | the only real management, and it's mechanical |

Consequences that justify "simple":

- **Peak footprint is a non-issue at our scale.** Resident models (~2 GB, mostly
  mmap) + one audio buffer (≤ ~700 MB) + ORT scratch (a few hundred MB) ≈ **< 3.5 GB
  peak** with a serial queue. At ≥ 8 GB we have wide headroom, so we do **not**
  budget peak memory or load/unload models. (That tension only appears on mobile,
  which is out of scope here — see below.)
- **The real risk is *time*, not peak** — the standalone app has **high uptime**
  (users sleep the lid rather than quit), so a slow leak or allocator fragmentation
  compounds over days/weeks. The countermeasures are aimed squarely at that.
- **Sleep/resume is free.** A closed lid is a freeze/thaw, not a restart; mmap'd
  model pages survive in RAM, so **resume is instant** *because* models stay
  resident. "Resume quickly" is therefore an argument **for** the resident-models
  design, not against it.
- **Run-to-completion is what lets the arena be trivial.** Because no job stops
  mid-flight, the per-job arena needs no sub-job checkpoints or partial rollback:
  **allocate scratch from one arena, reset it wholesale when the job finishes.**
  Reset reclaims everything at once with zero per-allocation `free` — which is also
  exactly what prevents leak-creep and fragmentation across a week of uptime.

### The toolset (and nothing more)

- **`mmap` model weights** — keep the big bytes file-backed and OS-reclaimable, not
  on our heap. (Default in ORT/ggml; just don't fight it.)
- **`std::pmr::monotonic_buffer_resource` per job, reset at job end** — one reusable
  arena sized to the largest expected job; all working buffers allocate from it; the
  orchestrator resets it after writing outputs. No accumulation across jobs.
- **mimalloc or jemalloc** (link-time swap) — for everything outside the arena, to
  resist the RSS-creep glibc `malloc` shows over long uptime.
- **AddressSanitizer + LeakSanitizer in CI** (Phase 0 onward) — over high uptime,
  even a tiny per-job leak is eventually fatal, so it must be impossible to merge
  one. (TSan later for the server's concurrency.)
- **RAII for everything in-process** — `unique_ptr` by default; custom-deleter
  `unique_ptr` wrapping every C handle (`OrtValue*`, `sherpa_*`, `sqlite3*`,
  ffmpeg `AVFrame*`); `std::span`/`string_view` for the zero-copy spans.

### FFI ownership rule (the one place that needs discipline)

Across the pybind/JNI/Swift seam, **memory is always freed by the same side (same
allocator) that created it.** Concretely: prefer **caller-allocates / callee-fills**
for big buffers; within pybind, return owning C++ objects with the correct
`return_value_policy` and expose large buffers zero-copy via the **buffer protocol**
+ `py::keep_alive` (so the C++ owner outlives the Python view). **Never `malloc`
on one side and `free` on the other.** Raw paired `create_X`/`free_X` only appears
at the C-ABI layer (mobile), which is out of scope here.

### Per-phase implications

- **Phase 0** — turn on **ASan/LSan** in the CTest/CI job from the start; pick
  mimalloc-or-jemalloc here so every later phase builds on it.
- **Phase 2** — the **decode-once buffer** is the largest working allocation; it is
  the arena's headline tenant and the thing to size the arena around.
- **Phase 3** — preallocate the **Viterbi trellis** to a max size and reuse it;
  batched-alignment emission buffers come from the arena; use **ORT IO binding** so
  emissions land in our buffer (zero-copy), not an ORT-internal copy we then copy
  again.
- **Phase 5** — the orchestrator **resets the arena at job end**, after writers
  flush. Progress callbacks crossing pybind follow the FFI ownership rule.
- **Phase 6** — benchmarks should also watch **RSS over many sequential jobs** (a
  leak/fragmentation guard), not just per-stage latency.

### Explicitly out of scope (deliberate simplifications)

- **Mid-job cancellation / streaming stop** — not supported; jobs run to completion.
  Cancellation, if ever needed, is **coarse** (between jobs in the queue, or at most
  at stage boundaries), never mid-stage. This is what keeps the arena reset-only.
- **Peak-memory budgeting / model load-unload** — unnecessary at desktop/server
  scale; not implemented.
- **Mobile memory policy** — a phone's per-app ceiling *would* force load/unload and
  a different per-platform model-residency policy. Out of scope until/unless mobile
  becomes a first-class decode target; it does not change the desktop/server design
  above.

## Phase dependency map

```
0 scaffold+goldens ─┬─► 1 store.py (DB + edits) ──► (host-side, parallelizable)
                    │
                    ├─► 2 decode+VAD ─► 3 alignment ─► 4 ASR+diarize ─► 5 writers+e2e
                    │        (Python ASR feeds 3 until 4 lands)              │
                    └────────────────────────────────────────────► 6 timing gates
                                            (benchmarks accrue from phase 2 on)
```

- **0 gates everything** (no oracle, no goldens → nothing to validate against).
- **1 is independent** of the model phases and can run in parallel once 0 is done.
- **2 → 3 → 4 → 5 is the critical path**; 3 is sequenced before 4 deliberately
  (risk), with Python ASR supplying transcripts to the aligner in the interim.
- **6 accrues** bench numbers from Phase 2 onward; the *gate* lands once stages
  are stable.

## Cross-cutting decisions

**Resolved (locked in Phase 0 / by committed decision):**
1. ✅ **Golden reference for ASR** — *decoupled goldens* (fact 3): downstream
   stages tested on a **fixed transcript**; ASR judged by WER/CER. Not regenerated
   per ASR engine.
2. ✅ **Model + library version pinning** — pin lib versions + HF model revisions in
   the golden manifest; regenerate only via the documented script.
3. ✅ **Tolerance budget** — a **Phase-0 deliverable**, measured from torch-vs-ORT
   drift (starting points in the tolerances note above).
4. ✅ **Diarization golden source** — the **vendored** `community-1` checkpoint
   (`app/models/…`), no HF token, even in CI.
5. ✅ **`store.py` shadow vs replace** — **replace** (Phase 1 done-bar = full API
   parity).
6. ✅ **VAD default** — **silero** (architecture §4D); *open sub-point:* whether
   pyannote VAD stays selectable, and goldens are generated with silero.

**Still open:**
- **Host swap timing** — `app/` stays Python through Phase 5; *when* (if ever) the
  C++ HTTP/SSE server replaces Flask is out of scope here but shapes Phase 5's
  progress-bridging design.
