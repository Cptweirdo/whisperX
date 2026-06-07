# C++ Core Migration — Per-Phase Briefs

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

Two cross-cutting facts shape every phase:

1. **Strangler-fig, Python stays the host.** Until Phase 5, `app/` + the Python
   `whisperx` pipeline keep running. Each C++ stage is swapped in **behind the
   pybind11 `whisperx_core` module** and gated by `WHISPERX_CORE_STAGES`, so we
   always have a runnable reference to diff against. A phase is "done" when its
   stage matches Python on the golden set with the flag on.
2. **Alignment is sequenced before ASR (Phase 3 before 4) for risk, not for data
   flow.** Alignment *consumes* transcript text, which ASR produces — but in the
   strangler model **Python ASR feeds the C++ aligner** during Phase 3, so we can
   retire the highest-risk stage (wav2vec2 export + Viterbi) early without C++ ASR
   existing yet. The dependency is satisfied by the oracle.

**Tolerances** (used throughout "Validation"): structural shape and integer
indices **exact**; speaker labels and trellis/backtrack paths **exact**; writer
bytes **exact**; floating timings within **~1 frame (~20 ms)**; emission/score
tensors within a small **fp32 epsilon** (to be pinned in Phase 0).

---

## Phase 0 — pybind scaffold + golden generator

### Context
Nothing can be validated until two things exist: a **C++ module Python can call**
(the migration vehicle *and* the parity oracle) and a **set of golden vectors**
to compare against. The repo has **no golden data today** — tests synthesize
inputs and mock models — so ground truth must be manufactured from the live
Python pipeline. This phase builds the harness, not any real stage. Build tooling
is settled in the plan (§5): **CMake + Ninja + vcpkg**.

### Goals
- A CMake project producing a **`whisperx_core` pybind11 module** that imports
  cleanly from Python (`import whisperx_core`), even if its first functions are
  stubs.
- **CMake + Ninja + vcpkg** bootstrap with the dependency manifest seeded
  (ONNX Runtime, sherpa-onnx, SQLiteCpp, nlohmann/json, Catch2, Google Benchmark,
  Eigen) — pulling and linking at least one (e.g. nlohmann/json) end-to-end.
- A **golden-vector generator** — a Python script/pytest pass that runs the real
  `whisperx` pipeline over a small fixed clip set and dumps every stage's
  intermediate to JSON: merged VAD chunks (`vads/vad.py::merge_chunks`), CTC
  emissions + `get_trellis`/`backtrack`/`merge_repeats` (`alignment.py:425-541`),
  word timings (`SingleWordSegment`), diarization turns + `assign_word_speakers`
  labels (`diarize.py:185`), and the four writer outputs.
- **CTest + Catch2** wired; one trivial parity test green (Python value ==
  `whisperx_core` value) to prove the oracle loop.
- A **CI job** skeleton beside `.github/workflows/python-compatibility.yml` that
  builds the module and runs CTest.

### Validation
- `import whisperx_core` succeeds in the project venv.
- `cmake --build` + `ctest` green locally and in CI.
- Goldens exist for **N clips** (covering: short/long, 1-speaker/2-speaker,
  English + at least one non-Latin-script language) and are committed under a
  `golden/` dir with a manifest (clip → model versions → sha256).
- The trivial parity test demonstrates a C++ function result matching Python.

### Unknowns / open questions
- **Clip set licensing + composition** — which audio can we commit? How many,
  which languages/speaker counts/lengths? (Drives coverage of every later phase.)
- **Determinism of goldens** — torch/CTranslate2/pyannote outputs can shift across
  library/model versions. Do we **pin exact model + lib versions** in the manifest
  and regenerate deliberately, or store goldens as immutable artifacts?
- **The gated diarization model** (`pyannote/speaker-diarization-community-1`,
  `diarize.py:91`) needs an HF token to generate its goldens — who runs that, and
  is it reproducible in CI?
- **Epsilon values** — what fp32 tolerance is "matching" for emissions/scores?
  Needs to be measured, not guessed.
- **pybind in CI** — manylinux build, vcpkg binary caching, build time budget. Does
  the compat CI runner have the headroom, or do we need a separate workflow?

---

## Phase 1 — DB layer in C++ (SQLiteCpp)

### Context
The session DB is the **in-place-upgrade compatibility contract** (plan §2). It's
low-ML-risk and high-value to port early: it exercises the pybind boundary on
something deterministic and locks down the must-not-break surface before any model
work. Source of truth: `app/store.py` (schema `:36-64`, `_migrate` `:86-92`, CRUD,
`snapshot_db`/`swap_db`), `app/paths.py` (`data_dir()`), `app/backup/`.

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
- A pybind binding so `app/store.py` can **delegate to** (or be shadowed by) the
  C++ store without changing callers.

### Validation
- Open a **real pre-existing `sessions.db`** (created by the Python app), read all
  rows, write new ones, and reopen with the Python store — **no schema drift, no
  data loss**.
- Migrations are **idempotent** (run twice → no error, no duplicate columns).
- A parity test: a row written via the Python store and via the C++ store is
  **byte-equal** field-for-field (including timestamp format and JSON serialization
  key order, if order matters to consumers).
- `snapshot_db` → `swap_db` round-trip preserves data; the existing
  `tests/test_backup*.py` suite stays green against the C++-backed store.

### Unknowns / open questions
- **Shadow vs replace** — does `app/store.py` delegate to `whisperx_core` (thin
  Python wrapper) or get fully replaced? Affects how much of `app/` changes now.
- **Threading** — Python holds a `threading.Lock`; the C++ store needs its own
  mutex, and WAL allows concurrent readers. How do the GIL, the pybind call, and
  the C++ mutex compose under the job queue + SSE readers?
- **JSON key-order / whitespace** — is any consumer sensitive to the exact
  serialization of `options`/`translations`? nlohmann/json must match Python
  `json.dumps` formatting if so.
- **SQLite build** — SQLiteCpp's bundled SQLite vs system SQLite: confirm WAL +
  Online Backup API availability and version alignment with what Python's `sqlite3`
  produced (file-format compatibility is guaranteed, but PRAGMAs/features should
  match).
- **Secrets + settings** — `secret_store.py` (keyring) and the `settings` table
  (`active_model`, `device`) — port to C++ now or leave in Python for this phase?
- **File-artifact ownership** — `transcript.json` and the atomic tmp+rename
  overlays are written by `app/pipeline.py`, not the store. Who owns those writes
  while the host is still Python?

---

## Phase 2 — decode-once + VAD / merge_chunks

### Context
The first real pipeline stage and the first **streamlining win**: today audio is
decoded up to 3× (ASR, alignment, diarization) via an **ffmpeg subprocess**
(`audio.py:44`); we decode **once** to a shared float32 16 kHz mono buffer. Then
the VAD segments it and `merge_chunks` packs voiced spans into ≤30 s windows.
Source: `audio.py:25` (`load_audio`, `SAMPLE_RATE=16000`), `vads/vad.py:19`
(`Vad.merge_chunks`), `vads/silero.py`.

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

### Unknowns / open questions
- **VAD model mismatch** — WhisperX's *default* VAD is **pyannote**
  (`vads/pyannote.py`), but the plan defaults the C++ core to **silero**. Goldens
  for `merge_chunks` must be generated with **silero** to be comparable (silero ≠
  pyannote segmentation). Do we standardize golden generation on silero, or keep
  pyannote VAD as an option and golden both?
- **ffmpeg build/license/size (Option B residuals)** — build a **default LGPL**
  ffmpeg (no `--enable-gpl`/`--enable-nonfree`; native AAC decoder, no `fdk-aac`;
  MP3 patents expired); dynamic-link or LGPL-compliant static. Open: **codec/format
  trimming** (strip muxers/encoders/video filters we never use to cut binary size),
  the final size budget, and a legal sign-off for distribution.
- **Decode determinism** — ffmpeg version pinned so decoded buffers (hence
  goldens) are reproducible across builds.
- **silero version/thresholds** — `vad_onset=0.500`, `vad_offset=0.363`,
  min_duration on/off — exact parity with the sherpa-onnx silero build?

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

### Unknowns / open questions
- **ONNX export fidelity** — wav2vec2's conv feature extractor, layer norm, GELU,
  attention: do they export cleanly at a chosen opset, and what's the torch→ORT
  fp32 drift on emissions? (The load-bearing risk.)
- **Dynamic shapes vs buckets** — variable audio length: dynamic axes vs fixed
  length buckets with padding. Does padding + attention mask change boundary-frame
  emissions enough to shift word edges?
- **Batching correctness** — do padded batch elements stay numerically identical to
  single-segment runs (mask handling)?
- **nltk → ICU parity** — sentence spans drive `interpolate_nans` *grouping*, so a
  different split shifts interpolated timings. How close must the splitter be, and
  do we accept small divergence?
- **torchaudio vs HF dicts** — `DEFAULT_ALIGN_MODELS_TORCH` (torchaudio bundles)
  vs `_HF` (Wav2Vec2ForCTC) use different tokenizers/dictionaries; the port must
  reproduce each. `torchaudio.functional.forced_align` is a reference but is the
  same algorithm — do we lean on it for cross-checking?
- **Per-language model proliferation** — 5 + 38 models. Export pipeline for all?
  Which to bundle vs download? (Ties to the model-asset / download strategy.)

---

## Phase 4 — ASR (sherpa/ORT) + diarize + assign

### Context
Swap the two remaining models. **Whisper** moves to a **pluggable ASR backend**
(default sherpa-onnx Whisper on ORT; **whisper.cpp/GGML for Apple-Silicon Metal** —
see §"Runtime & acceleration"); **pyannote diarization → sherpa-onnx**
(`pyannote-segmentation-3.0` + CAM++). Then port the speaker-assignment glue.
Source: `asr.py:31` (`WhisperModel`), `:106` (`FasterWhisperPipeline`), `:325`
(`load_model`), `:417-445` (`default_asr_options`); `diarize.py:14`
(`IntervalTree`), `:91` (`DiarizationPipeline`, default gated `community-1`),
`:185` (`assign_word_speakers`), `:170` (pandas DataFrame).

### Goals
- An **ASR backend interface** producing `TranscriptionResult` (`segments` text +
  `language` + `avg_logprob`) from the Phase-2 VAD chunks, with two
  implementations: **sherpa-onnx Whisper (ORT)** default, **whisper.cpp/GGML**
  (Metal/CUDA/CPU). The backend produces **text only** — word timestamps still come
  from the ORT wav2vec2 aligner (Phase 3), never whisper.cpp's DTW. Likely
  **supersedes the current `mlx` backend** as the Apple path.
- Backend selection wired to the existing `device` setting
  (`cpu`/`cuda`/`mlx`/`whispercpp` → ORT vs GGML), with language detection.
- **sherpa-onnx diarization** → turns (`start`/`end`/`speaker`), replacing the
  pyannote DataFrame with structs.
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

### Unknowns / open questions
- **Whisper decode parity is NOT byte-exact — across *three* decoders now.**
  CTranslate2 (today), sherpa-onnx ORT, and whisper.cpp/GGML each produce different
  tokens/segmentation/`avg_logprob`. "Segment text == golden" is **too strict**.
  Strategy (reinforced by the pluggable backend): **golden-test align/diarize/writers
  against a *fixed* transcript input** (decoupled from the ASR engine) and validate
  each ASR backend by **WER/CER, not bytes**. **Decide before building the
  validation.**
- **ASR backend selection + parity between backends** — sherpa-onnx vs whisper.cpp
  won't agree segment-for-segment either; the orchestrator picks per `device`, and
  each is quality-gated independently. Confirm the GGML CMake/Metal-shader build cost
  is acceptable.
- **`default_asr_options` parity** — beam size, temperature fallback, patience,
  suppress tokens, `chunk_size=30`, `batch_size`. Which map onto sherpa-onnx, and
  which don't exist there?
- **Diarization model differs** — Python default is the **gated `community-1`**;
  sherpa uses **pyannote-seg-3.0 + CAM++**. Different models → different turns, so
  **`assign_word_speakers` must be golden-tested in isolation** (synthetic turns),
  while end-to-end speaker quality is an **A/B**, not a parity check.
- **`avg_logprob` semantics** — used downstream/UI; does sherpa expose an
  equivalent, and is the scale comparable?
- **num/min/max speakers** params — parity of the speaker-count controls.

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

### Validation
- **Writer outputs byte-identical to golden** for every supported format.
- Full `run_job` on a golden clip == Python's `transcript.json` (structure +
  labels exact; timings within ~1 frame).
- Progress events fire in the **correct order** and update the session row's
  `stage` exactly as Python does.
- The `app/` UI shows a C++-produced session indistinguishably from a Python one.

### Unknowns / open questions
- **Float formatting / byte-identical JSON** — `transcript.json` byte-equality
  requires matching Python's float `repr`/rounding for timestamps and scores.
  Achievable, but needs an explicit number-formatting decision (and the SRT/VTT
  timecode rounding must match exactly too).
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
alignment, models-resident + pipelining) rather than assert them.

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
0 scaffold+goldens ─┬─► 1 DB layer ───────────────► (host-side, parallelizable)
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

## Open questions that span phases (resolve early)

1. **Golden reference for ASR** — once sherpa-onnx Whisper differs from CTranslate2
   (Phase 4), do downstream goldens (alignment input text, writer output) get
   **regenerated from sherpa**, making it the new reference? This decision affects
   Phases 3 and 5 validation strategy.
2. **VAD default (silero vs pyannote)** — fixes how Phase 2 goldens are generated
   and whether pyannote VAD stays selectable.
3. **Host swap timing** — `app/` stays Python through Phase 5; *when* (if ever) the
   C++ HTTP/SSE server replaces Flask is out of scope here but shapes Phase 5's
   progress-bridging design.
4. **Model-version pinning** — the whole golden strategy depends on pinning model +
   library versions and a deliberate regeneration process.
5. **Tolerance budget** — the single epsilon/frame budget (set in Phase 0) is
   referenced by every later phase; getting it wrong invalidates comparisons.
