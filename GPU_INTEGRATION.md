# GPU / CUDA Integration Brief — Native C++ Pipeline (Stages 1–3)

## General overview

The native compute chain runs as three sequential, independently-loaded inference
stages, each one wrapping a neural-network forward pass. Today **every stage runs on
CPU** — the provider is hardcoded to `"cpu"` and intra-op threads to `1` at every
construction site. There are no custom GPU kernels to write: all three stages already
execute their models under **ONNX Runtime** (two of them via the sherpa-onnx C-API,
one via a direct `Ort::Session`). The single lever that moves this pipeline onto the
GPU is **ONNX Runtime's CUDA execution provider** — turned on at build time (so the
runtime ships the GPU build of `libonnxruntime.so`) and selected at runtime (a device
knob threaded down to each engine constructor).

Pipeline (driven by `core/orchestrate/orchestrate.cpp` → server wiring in
`adapters/server/jobs/runner.cpp` → engines built in
`adapters/server/models/model_manager.cpp`):

```
decode (ffmpeg, CPU)
  └─ Stage 1  ASR        Whisper enc/decoder   sherpa-onnx   RTF ~0.10   ← VAD (silero) feeds it
  └─ Stage 2  Align      wav2vec2-CTC          direct ORT    RTF ~0.15   ← already batched
  └─ Stage 3  Diarize    segmentation+embed    sherpa-onnx   RTF ~0.15   ← provider already plumbed
```

RTF figures are the measured CPU medians in `runner.cpp::stage_rtf()`. Together stages
1–3 dominate wall-clock; decode and the pure-C++ post-processing (trellis Viterbi,
log-softmax, speaker assignment, writers) are cheap by comparison and are **out of
scope** for GPU — they would add kernel-authoring cost for negligible return.

### Cross-cutting enablers (apply to all three stages)

These are shared prerequisites; the per-stage briefs assume they land first.

1. **Build flag.** `CMakeLists.txt:165` sets `SHERPA_ONNX_ENABLE_GPU OFF`. sherpa-onnx
   downloads its own ONNX Runtime; OFF means the **CPU** ORT. Flip to ON so the GPU
   build is fetched. Because the server links **one shared** `libonnxruntime.so`
   (`CMakeLists.txt:215`, via sherpa's `location_onnxruntime` cache var) and Stage 2's
   direct `Ort::Session` uses that same library, this one flag fixes the runtime for
   all three stages at once.
2. **Keep ORT shared.** `CMakeLists.txt:171-178` documents that ORT is forced `SHARED`
   to dodge a static-archive `std::regex`/heap-corruption bug in ORT's device
   discovery (`free(): invalid pointer in DeviceDiscovery`). The GPU build **must stay
   shared** — do not switch to the static archive when enabling CUDA.
3. **A device knob.** No device/provider configuration exists today. Add a
   `WHISPERX_DEVICE` env var (e.g. `cpu` | `cuda`) to `adapters/server/config.{hpp,cpp}`
   (alongside the existing `WHISPERX_*` knobs and `batch_size`), carry it on `Config`,
   and thread it through `ModelManager` to each engine constructor.
4. **Honest status.** `model_manager.cpp:66-67` hardcodes `{"device","cpu"}` and
   `{"cuda_available",false}` in the `/models` status payload (the SPA reads this).
   Wire real detection so the UI reflects the active device.
5. **CPU fallback.** Selecting `cpu` should also be the moment to raise
   `num_threads` above the current hardcoded `1` — single-thread CPU is leaving
   performance on the table even without a GPU.

---

## Stage 1 — ASR (Whisper)

**Files:** `core/asr/whisper_sherpa.{hpp,cpp}`, constructed at
`model_manager.cpp:105`.

### Purpose
Transcribe each voiced span (VAD-segmented, ~30 s windows) into text. Wraps
sherpa-onnx's offline Whisper recognizer (encoder + greedy decoder + tokenizer +
language ID, all inside sherpa). Also performs language identification over the first
30 s (`detect_language`). This is the heaviest single model in the pipeline.

> Note: VAD (silero, `core/audio/vad_silero.cpp`) feeds this stage. It is also an ORT
> model but is a tiny network streamed in 512-sample windows through a serialized
> drain loop — poor GPU occupancy, marginal payoff. Treat VAD as **CPU-stays** and
> fold any change into this stage only if trivial.

### Why GPU benefits
Whisper's encoder/decoder transformer forward is the largest matrix-multiply workload
in the chain (~0.10 RTF on CPU, and that is per-span across the whole clip). GPU
acceleration of attention/FFN is exactly the regime CUDA ORT is built for; this is the
highest-value stage to move.

### How it would be implemented
- `whisper_sherpa.cpp:66` hardcodes `config.model_config.provider = "cpu"`. The
  constructor (`whisper_sherpa.hpp:48`) **takes no provider argument at all** — add a
  `const std::string& provider = "cpu"` parameter, store it on `Impl`, and assign
  `config.model_config.provider = provider.c_str()` (the string must outlive the
  config, as the existing owned-string members already ensure).
- Pass the device from `ModelManager::load_asr` (`model_manager.cpp:105`), which today
  passes `num_threads=1` and no provider.
- sherpa selects the CUDA EP internally from the provider string once the GPU ORT is
  present (enabler #1), so no direct ORT calls are needed here.

### Risks
- **Per-span serial loop.** `transcribe()` runs one offline stream per VAD span in a
  serial loop (`whisper_sherpa.cpp:180`); each span pads to a fixed 30 s mel. On GPU
  the per-call launch/copy overhead is amortized poorly versus a true batch — expect
  good but not linear speedup. Batching multiple spans into one GPU forward is a
  larger change sherpa's offline API may not expose.
- **VRAM.** Large-v3 weights plus the 30 s activation tensors per concurrent job.
  Jobs are already serialized one-at-a-time (`jobs.hpp`), which bounds this.
- **Numeric drift.** GPU kernels are not bit-identical to CPU; transcription text is
  robust to this, but any downstream parity assumptions should be re-checked.

### Unknowns / open questions
- Does the sherpa-onnx version pinned at `v1.13.2` (`CMakeLists.txt:183`) build cleanly
  against the CUDA ORT, and does its offline Whisper path actually dispatch to the CUDA
  EP (some sherpa graphs silently fall back to CPU nodes)?
- Cold-start cost of CUDA context + cuDNN autotune on first decode — does it blow the
  `/healthz` warm path or the first job's ETA?
- Is per-span batching worth pursuing, or is single-stream-on-GPU enough?

---

## Stage 2 — Align (wav2vec2-CTC)

**Files:** `core/align/wav2vec2_onnx.{hpp,cpp}`, constructed at
`model_manager.cpp:214`; driven via `core/align/align_driver.cpp`.

### Purpose
Force-align the transcript to audio with a per-language wav2vec2-CTC model to produce
word-level timestamps. This is the **only stage that calls ONNX Runtime directly**
(`Ort::Session`), not through sherpa. It already implements length-bucketed,
attention-masked **batching** (`kMaxBatch=8`, `kMaxBatchSamples=30 s`,
`wav2vec2_onnx.cpp:14-20,78-105`).

### Why GPU benefits
Highest GPU upside of the three: the forward is a deep conv+transformer over long
waveforms (~0.15 RTF), and the code is **already structured for batched execution** —
the exact shape GPUs reward. The padded-batch machinery that exists to bound CPU memory
becomes a throughput win on GPU.

### How it would be implemented
- `make_options()` (`wav2vec2_onnx.cpp:22-27`) builds `Ort::SessionOptions` with only
  intra-op threads + graph optimization. Append the CUDA execution provider here:
  `OrtSessionOptionsAppendExecutionProvider_CUDA(opts, /*device_id*/0)` (or the C++
  `AppendExecutionProvider_CUDA(cuda_options)`), gated on the device knob.
- The constructor (`wav2vec2_onnx.hpp:23`) takes only `(onnx_path, num_threads)` — add
  a provider/device parameter and pass it from `model_manager.cpp:214`.
- The input tensors are built on a CPU `MemoryInfo` (`wav2vec2_onnx.cpp:108`). With the
  CUDA EP, ORT will copy host→device automatically; an optional later optimization is
  binding device memory via `IoBinding` to cut the per-batch H2D/D2H copies.

### Risks
- **Direct-ORT coupling.** Unlike stages 1 & 3, this path constructs the EP itself, so
  it depends on the CUDA ORT exposing `OrtSessionOptionsAppendExecutionProvider_CUDA`
  in the headers reached transitively via sherpa's interface includes
  (`onnxruntime_cxx_api.h`, per the `CMakeLists.txt:210` comment). If those headers are
  the CPU build's, the symbol may be missing at compile time.
- **Two graph contracts.** The mirror ships a 2-in/2-out (layer_norm, batchable) and a
  1-in/1-out (group_norm, per-segment) graph; the code introspects and binds
  accordingly (`wav2vec2_onnx.cpp:36-54`). Verify both run on CUDA — group_norm models
  force `b==1`, losing the batch advantage.
- **Batch budget tuned for CPU RAM.** `kMaxBatchSamples = 30 s` was chosen to bound
  CPU attention memory; on GPU the optimal batch/VRAM tradeoff differs and may want
  retuning (possibly via the device knob).

### Unknowns / open questions
- Are the CUDA-EP headers/symbols actually present once enabler #1 swaps the ORT build,
  given the headers arrive transitively through sherpa rather than a direct ORT package?
- Does `frame_lengths`-based trimming behave identically under CUDA (it is read back to
  host at `wav2vec2_onnx.cpp:153`)?
- Is `IoBinding` worth the added complexity, or is auto-copy fast enough at these batch
  sizes?

---

## Stage 3 — Diarize (segmentation + embedding)

**Files:** `core/diarize/diarize_sherpa.{hpp,cpp}`, constructed at
`model_manager.cpp:175`.

### Purpose
Label who-spoke-when. Runs **two** sherpa-onnx models: a pyannote-segmentation-3.0
graph and a speaker-embedding extractor (wespeaker CAM++), followed by clustering, then
`assign_word_speakers` (pure CPU) maps speaker turns onto aligned words. Optional —
disabled when no diarization assets are present.

### Why GPU benefits
Two NN forwards over the full clip (~0.15 RTF combined). The embedding extractor in
particular runs per-cluster passes (`diarize_sherpa.cpp:116-165`) and benefits from GPU
throughput. Same ORT-CUDA mechanism as the other stages.

### How it would be implemented
- **Least work of the three** — the provider is already a first-class constructor
  parameter: `SherpaDiarizer(..., const std::string& provider = "cpu", ...)`
  (`diarize_sherpa.hpp:41`), applied to both `config.segmentation.provider` and
  `config.embedding.provider` (`diarize_sherpa.cpp:29,33`) and to the embedding
  extractor (`:70`).
- The only change is to **pass a non-default provider** from `model_manager.cpp:175`
  (which currently constructs `SherpaDiarizer(seg, embed)` with all defaults) and to
  thread the device knob there.
- sherpa selects the CUDA EP from the provider string once the GPU ORT is present.

### Risks
- **Clustering stays on CPU.** sherpa's FastClustering / PLDA step is not a GPU op;
  GPU only accelerates the two forward passes, so the stage's speedup is partial.
- **Embedding stream granularity.** Embeddings are computed by feeding per-cluster
  audio through online streams in a loop (`diarize_sherpa.cpp:134-160`); like Stage 1's
  per-span loop, many small GPU calls may underutilize the device.
- **Parity gate.** `assign_speakers.cpp` is compiled `-ffp-contract=off` for exact
  parity with Python (`CMakeLists.txt:143`); that is downstream CPU code and unaffected,
  but GPU drift in the *upstream* embeddings could nudge cluster assignments. Re-check
  the diarization parity tests.

### Unknowns / open questions
- Does the pinned sherpa build route **both** the segmentation and embedding graphs to
  CUDA, or does one fall back to CPU?
- Is the per-cluster embedding loop a bottleneck that negates the GPU win at this
  stage's smaller compute share?
- Should diarize even use the GPU when stages 1–2 are mid-flight (it runs after them,
  so contention is low given serialized jobs — confirm).

---

## Summary of required changes

| Concern | Location | Stage(s) |
|---|---|---|
| `SHERPA_ONNX_ENABLE_GPU ON` | `CMakeLists.txt:165` | all |
| Keep ORT `SHARED` | `CMakeLists.txt:171-178` | all |
| Add `WHISPERX_DEVICE` knob | `adapters/server/config.{hpp,cpp}` | all |
| Real device/cuda status | `model_manager.cpp:66-67` | all |
| Add `provider` ctor param + assign | `whisper_sherpa.{hpp,cpp}:48,66` | 1 |
| Append CUDA EP in `make_options` + ctor param | `wav2vec2_onnx.{hpp,cpp}:22,23` | 2 |
| Pass provider from construction site | `model_manager.cpp:175` (already-supported param) | 3 |
| Pass provider/device from construction sites | `model_manager.cpp:105,214` | 1, 2 |

**Out of scope (CPU stays):** silero VAD (tiny, streamed), ffmpeg decode, trellis
Viterbi, emission log-softmax (FMA-off for torch parity), speaker assignment, writers.

**Biggest single unknown across all stages:** whether the sherpa-onnx `v1.13.2` GPU ORT
build dispatches each graph to CUDA cleanly (no silent CPU-node fallback) and exposes
the CUDA-EP symbols Stage 2's direct `Ort::Session` needs — this should be validated
with a spike before committing to the full plumbing.

---

## Runtime CPU ⇄ GPU switching

The briefs above add a **load-time** device knob (`WHISPERX_DEVICE`). Switching device
**while the server is running** — without a restart — is a strictly larger problem,
because an ORT/sherpa session bakes its execution provider in at construction and cannot
be re-pointed. A device switch therefore means **tearing down and rebuilding every
resident engine**, and the current `ModelManager` is built on the opposite assumption.

### What already exists
- A `POST /api/device` endpoint (`api_controller.hpp:467`) — today it accepts only
  `"cpu"` (no-op), rejects everything else, returns **409 `"busy"`** when
  `store.has_active_jobs()`, and persists the choice via `store.set_setting("device", …)`.
  The request/response shape and the busy-gate are already in place.
- A persisted `device` setting and an onboarding `device` field
  (`onboarding_finish`, `api_controller.hpp:582`) — also hardcoded to reject non-cpu.
- `set_active(model)` already demonstrates the load/warm/`notify_change` pattern a
  device switch would mirror (`model_manager.cpp:139`).

### What is missing

**1. Device state + a rebuild path on `ModelManager`.** There is no `device_` member and
no way to change it. The class comment (`model_manager.hpp:6-8`) explicitly says *"device
is fixed to cpu"*. Need a `set_device(const std::string&)` that swaps the active provider
and forces every subsequent load onto it.

**2. Cache eviction.** The header advertises *"cached, no eviction — switching back is
instant"* (`model_manager.hpp:3`). Every resident engine — `asr_` (per checkpoint),
`align_` (per language), `diarize_` — is bound to the *old* provider, so a device switch
must **drop and rebuild all of them**. "No eviction" is exactly the invariant that breaks.

**3. Lifetime safety — the load-bearing risk.** The job runner holds **raw references /
pointers** into the manager-owned caches for the duration of a job:
- `load_asr()` returns `WhisperSherpa&` (`runner.cpp:110`),
- `align_for()` returns `AlignHandle{ Wav2Vec2Onnx*, dictionary*, … }` (`model_manager.hpp:38`),
- `ensure_diarize()` returns `SherpaDiarizer*`.

Evicting/rebuilding those maps while a job is mid-flight **frees memory the running job is
still using → use-after-free**. The existing `has_active_jobs()` 409 gate is the intended
guard, but it must be made *authoritative*: no new job may start between the gate check and
the eviction, and a job already running must complete (or be cancelled and joined) first.
Options, in increasing robustness: (a) rely on serialized single-job execution + the busy
gate under a lock that also blocks job start; (b) hand engines out as `shared_ptr` so an
in-flight job keeps its instance alive past eviction; (c) a generation/refcount so eviction
waits for outstanding borrows.

**4. Capability detection.** The switch must reject `cuda` when the binary wasn't built
with the GPU ORT or no device is present — today `status()` hardcodes
`cuda_available:false` (`model_manager.cpp:67`). Real detection (enabler #4 above) is a
prerequisite, and the endpoint should 400 with a clear message on an unavailable device
rather than constructing a session that silently falls back to CPU.

**5. VAD path.** silero VAD is **not** cached in the manager — `runner.cpp:151` calls
`wa::silero_segments(buf, silero_path, …)` which builds a VAD per job with a hardcoded
`"cpu"` provider (`vad_silero.cpp:36`). To honour a device switch (if VAD ever moves to
GPU) the device must also be threaded into that call; otherwise document VAD as
deliberately CPU-pinned regardless of the switch.

**6. Re-warm + status push.** After a successful switch, re-warm the active model on the
new device in the background (as `set_active` does) and `notify_change()` so
`/models/events` and the SPA device toggle reflect the new state. The first job after a
switch otherwise pays the full reload cost.

**7. Persistence + boot precedence.** The `device` setting is persisted but unused at
boot. Define precedence between `WHISPERX_DEVICE` (env) and the stored `device` setting,
and construct the boot engines accordingly so a restart preserves the user's choice.

**8. Failure atomicity.** A GPU init failure mid-switch (OOM, missing cuDNN, EP load
error) must leave the manager in a **consistent** state — ideally roll back to the
previous device rather than a half-evicted, unusable manager — and surface the error via
the endpoint + status, not crash the server (recall the server has no signal handler;
an uncaught exception on the request thread is fatal).

### How it would be implemented (sketch)
- Add `std::string device_` to `ModelManager`, guarded by `lock_`; route every engine
  constructor through it (the same provider plumbing the per-stage briefs add).
- Add `json set_device(const std::string& dev)`: validate against capability detection →
  under `lock_` + `load_lock_`, confirm idle (no active jobs/borrows) → clear `asr_`,
  `align_`, reset `diarize_`/`diarize_loaded_` → set `device_` → `warm(active_)` →
  `notify_change()` → return `status()`.
- Replace the cpu-only branch in `switch_device` (`api_controller.hpp:472`) with: parse
  device, 400 on unknown/unavailable, **409 if busy** (keep the existing gate), else
  `manager.set_device(dev)` + persist + return status. Mirror the same acceptance in
  `onboarding_finish`.
- Decide the lifetime strategy (#3) up front — it dictates whether `load_asr` /
  `align_for` keep returning raw refs or move to `shared_ptr`.

### Risks
- **UAF on eviction** (#3) is the dominant correctness risk; the whole feature hinges on
  getting engine lifetime right relative to in-flight jobs.
- **Double memory / VRAM pressure** during the swap if old and new engines briefly
  co-exist (mitigated by evicting *before* loading, at the cost of a reload stall).
- **Switch latency** — rebuilding large models + CUDA context/cuDNN autotune can take
  seconds; the SPA needs a loading state, and the request should not block the event loop.
- **Partial diarize/align caches** — a switch must invalidate *all* language align
  entries and the diarizer, not just the active ASR checkpoint.

### Unknowns / open questions
- Is the serialized single-job model (`jobs.hpp`) a strong enough guarantee to make the
  busy-gate sufficient, or is explicit borrow-tracking (`shared_ptr`/refcount) required?
- Should an in-flight job **block** a device switch (current 409 design) or should the
  switch **cancel + drain** it (a `CancelFlag` already exists in the runner)?
- Can ORT/CUDA contexts be created and destroyed repeatedly in one process without leaks
  or driver-state corruption across many switches?
- Does switching to CPU need to free VRAM eagerly (destroy the CUDA EP/context), and does
  sherpa expose that, or does it linger until process exit?
