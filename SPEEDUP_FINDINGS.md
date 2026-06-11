# Pipeline Speedup Findings — CUDA & Apple Silicon

Assessment of the native C++ pipeline (sherpa-onnx + ONNX Runtime) against
three external sources: the research report on forced alignment / diarization
architectures ("Advanced Methodologies in Speaker-Attributed Transcription",
2026-06); a full read of
[insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)
(a 183-line CLI over the HF `transformers` ASR pipeline — no new engine, just
four switches: fp16 + Flash-Attention-2 + 30 s chunking + batch 24); and a
full read of
[lightning-whisper-mlx](https://github.com/mustafaaljadery/lightning-whisper-mlx)
(Apple's mlx-whisper example + batched KV-cache decode + 4/8-bit weights —
dedicated analysis in `LIGHTNING_WHISPER_MLX_ANALYSIS.md`). Goal: major
speedup on CUDA or Apple (MLX / CoreML / Metal) while preserving quality.

Companion briefs: `GPU_INTEGRATION.md` (CUDA, landed), `METAL_INTEGRATION.md`
(Metal routes A/B), `MLX_PORT.md` (streaming, future), `MAC_PERFORMANCE.md`
(Python-side Mac benchmarks).

## What we already have (the report recommends things that are landed)

- **CUDA EP + runtime device switch** — `Device{Cpu,Cuda,CoreML}`, `set_device()`
  with engine teardown, `server-cuda` preset, provider threaded to every engine
  (`adapters/server/models/model_manager.cpp`).
- **whisper.cpp Metal, proven Python-side** — RTF **0.070** on `large-v3-turbo`
  (≈14× realtime), 1.57–2× faster than MLX at equal Russian quality
  (`MAC_PERFORMANCE.md`). The native-core port plan is `METAL_INTEGRATION.md`
  Route B.
- **sherpa-onnx as the deployment runtime** — exactly the report's recommended
  execution layer; the foundation is right.

## Apple Silicon — most likely speedups after Route B (2026-06-11)

Route B (whisper.cpp Metal) landed and **flattened the profile**. The Amdahl picture
is now completely different from when the items below were first ranked, and that
re-ranks everything.

**New warm profile (M4, 300 s Russian clip, `large-v3-turbo` fp16 ggml, E2E RTF 0.199):**

| Stage | RTF | share | accelerator |
|---|---|---|---|
| Stage 1 ASR (whisper.cpp/Metal) | 0.073 | 37% | Metal GPU |
| Stage 2 align (wav2vec2, ORT) | 0.069 | 35% | **CPU** |
| Stage 3 diarize (sherpa, ORT) | 0.056 | 28% | **CPU** (pinned) |

No stage dominates. Stage 1 — the old 65% bottleneck — is now fast *and* small.
**Combined align+diarize = 63%, both CPU.** So the next gains come from removing or
accelerating the CPU stages, not more Stage-1 work. Every lever is judged against this.

### Most likely, ranked by ROI ÷ cost

**1. Quantized ggml + flash-attention on Metal — MEASURED 2026-06-11 (M4): flash is the
lever, q8_0 a small bonus, q5_0 rejected.** Both knobs already ship
(`WHISPERX_GGML_QUANT=q5_0|q8_0`, `WHISPERX_WHISPERCPP_FLASH_ATTN=1`) — this was a
*measurement*, a boot flag and a one-file download, not an implementation. Swept all six
quant×flash combos on the 300 s Russian clip (`scripts/bench_whispercpp_metal.sh`),
WER as drift vs the fp16/flash-off transcript (`bench_whispercpp_wer.py`):

| config | Stage-1 RTF | ×fp16 | WER drift | verdict |
|---|---|---|---|---|
| fp16, flash off | 0.063 | 1.00× | 0.000 | baseline |
| fp16, **flash on** | 0.055 | 1.15× | 0.013 | flash carries it |
| q8_0, flash off | 0.062 | 1.02× | 0.007 | quant alone ≈ nothing |
| **q8_0, flash on** | **0.050** | **1.26×** | **0.011** | **winner (under gate)** |
| q5_0, flash off | 0.057 | 1.11× | 0.046 | over WER gate |
| q5_0, flash on | 0.050 | 1.26× | 0.062 | over WER gate |

**The prediction inverted.** The lightning-whisper-mlx thesis was *quantization is the
decode-bandwidth lever* (`LIGHTNING_WHISPER_MLX_ANALYSIS.md` §2); measured on
`large-v3-turbo`, **q8_0 alone barely moves Stage 1 (1.02×)** — flash-attention is the
real win (the insanely-fast-whisper FA2 lesson, item 5). Turbo's decoder is pruned to 4
layers, so autoregressive decode is a small slice and the bandwidth argument has little
leverage; the compute-bound encoder, which flash-attn accelerates, dominates. q8_0 only
helps *stacked on* flash (1.15× → 1.26×) and is near-lossless (1.1% drift). **q5_0 breaks
the gate** — 4.6–6.2% Russian WER drift (it garbles words, e.g. drops "мало"), reject.
Net: **q8_0 + flash-attn = 1.26× Stage 1** (≈4× over the original sherpa CPU baseline),
but only **~5% E2E** — exactly the doc's modest ceiling, because Stage 1 is already 37%.
**Now the Apple default** (`config.cpp::load_config`, 2026-06-11): on `__APPLE__`,
`WHISPERX_ASR_BACKEND`/`WHISPERX_GGML_QUANT`/`WHISPERX_WHISPERCPP_FLASH_ATTN` default to
`whispercpp`/`q8_0`/`1` (degrades to sherpa if the build lacks `WHISPERX_WHISPERCPP_BUILD`;
explicit env/persisted choice still wins). Done — the cheap win is banked and shipped by
default; the structural levers (2–3) are where the real E2E time still is.

**2. Skip align — "fast timestamps" (item 6), now high-value.** Stage 2 is 35% of
wall-clock and CPU-bound with **no Mac GPU lever** (CoreML EP is dead; MLAS has no fp16
win). whisper.cpp already emits token-level DTW timestamps — the Metal backend produces
the input for free. An opt-in mode that takes word timestamps from whisper.cpp and drops
the wav2vec2 stage removes ~35% E2E for **all** languages. The quality trade is real
(attention-DTW boundaries < wav2vec2 forced alignment — the founding premise of
whisperX), so opt-in, clearly labeled. This was an "optional product idea" when Stage 1
dominated; post-Route-B it is the single biggest *general-purpose* E2E lever.

**3. GigaAM CTC for Russian (item 2) — biggest structural win for the actual corpus.**
Compounding: a Conformer-CTC Russian forward is cheaper than Whisper enc/dec, gives ~8%
WER vs ~25% for large-v3, **and** its CTC logits do forced alignment in the same pass →
the wav2vec2 align stage (35%) disappears for Russian. Our benchmark corpus *is* Russian,
so this hits both CPU co-bottlenecks at once. Cost: a per-language ASR-backend routing
dimension — much cheaper now that the `AsrBackend` plumbing exists (it's just another
backend; language-detect already routes). Russian-only.

**4. INT8 embedding extractor on CPU (Stage 3, item 4).** sherpa ships int8 embedding
models — ~4× smaller, big CPU speedup, near-lossless for speaker embeddings. Diarize is
28% and CPU-pinned by design (the tiny-forward GPU-starvation finding, item 1 / §1,
applies to Metal too). INT8 is the realistic Mac diarize lever. Gate: DER unchanged on
the golden dialogs. Size it first by profiling the embedding-vs-segmentation-vs-clustering
split within the 17 s diarize stage.

### Demoted by the re-ranking

**Batched MLX decode (lightning's headline) — now poor ROI.** The "batched MLX beats
serial whisper.cpp" argument (`LIGHTNING_WHISPER_MLX_ANALYSIS.md` §1) was compelling when
Stage 1 was 65% of wall-clock. It no longer is: Stage 1 is 37% and already RTF 0.073.
Even a 2× from batched decode is ~18% E2E, for a large cost (hand-rolled MLX C++ decode
loop, a third model-asset family, Apple-only) — while lever 1 captures most of the
remaining Stage-1 headroom for near-zero cost and levers 2–3 attack the bottleneck MLX
doesn't touch. **Verdict: watch-list only; re-open solely for a future *streaming*
feature (`MLX_PORT.md`).**

**WHISPER_COREML ANE encoder (Phase 3) — marginal.** The encoder is a small fraction of
Stage 1 (autoregressive decode dominates); 3× on a small slice of an already-fast 37%
stage is noise, for an extra per-model asset. Skip unless profiling shows the encoder is
unexpectedly large.

**Diarize on ANE/CoreML — uncertain, low priority.** The conv-heavy embedding extractor
*might* load under the CoreML EP where the Whisper export didn't (different graph: no
fp16-cast fusion, no external-weights path loss), but the per-call overhead that pins
diarize to CPU (hundreds of tiny forwards) persists on Metal. Cheap to test, marginal
expected; INT8-on-CPU (lever 4) is the better-understood diarize lever.

### The one-line answer

Cheapest real win: **`WHISPERX_WHISPERCPP_FLASH_ATTN=1` (+ `WHISPERX_GGML_QUANT=q8_0`)** —
measured 1.26× Stage 1 / ~5% E2E (lever 1, done; flash is the lever, q8_0 a bonus, q5_0
rejected). Biggest structural win: **delete the align stage** — fast-timestamps for
everyone (lever 2) or GigaAM for Russian (lever 3); that's where the E2E time now is.
Stop ranking Stage-1 batching as a headline — Route B already took that bottleneck off
the table.

---

## Open items, ranked

### 1. CUDA — batch VAD chunks through ASR (biggest CUDA lever)

**STATUS: implemented + measured 2026-06-11 — batching turned out not to be
the CUDA lever; the real one was the model variant (item 1a, resolved — see
below and `CUDA_DECODE_FINDINGS.md`).**

`core/asr/whisper_sherpa.cpp` decodes **one stream per chunk, serially**
(`SherpaOnnxDecodeOfflineStream` per call; also inside the `decode_capped`
sub-window loop). The serial loop is precisely what WhisperX's famous "70×
realtime" fix removes — a GPU starves on single 30 s streams. The sherpa C API
exposes `SherpaOnnxDecodeMultipleOfflineStreams` — **but for Whisper it is a
serial loop inside sherpa** ("batch decoding is not implemented yet",
`offline-recognizer-whisper-impl.h`; still true upstream as of 2026-06-11). We
implemented true batching as a sherpa patch
(`third_party/sherpa-onnx-patches/`): one (N,T,C) encoder pass + lockstep
batched greedy decoder, opt-in via `WHISPERX_ASR_BATCH_SIZE` (default 1 = off).
The ONNX exports are batch-ready (dynamic `n_audio` axis everywhere).

Caveat: one-chunk-per-span is load-bearing — orchestrate/align map
spans↔chunks **by index** (`whisper_sherpa.cpp:120`). Batch decode preserves
stream order, so the contract holds, and the parity tests stay green
(`test_asr_sherpa_parity.py::test_batched_decode_matches_serial`).

**Measured (3080 Ti, large-v3-turbo, 10 min audio, scratch server):**
batch 8 transcribe RTF **0.526** vs serial **0.443** — 19% *slower*, VRAM
4.1 → 7.2 GB; serial GPU utilization was a ~20% sawtooth.

**Root cause found (item 1a, 2026-06-11 — see `CUDA_DECODE_FINDINGS.md`):
the model, not the decode loop.** The mirrored turbo assets are the **int8**
export (the only variant sherpa's release tarball ships); ORT's CUDA EP has
no `MatMulInteger`/`DynamicQuantizeLinear` kernels, so every heavy matmul ran
on the **CPU EP** — single-threaded (`threads_for(Cuda)=1`) — with Memcpy
ping-pong around each. "CUDA" was slower than plain CPU mode. The fp32 export
(csukuangfj/sherpa-onnx-whisper-turbo) gives transcribe RTF **0.036 E2E**
(12.3×) with zero code changes — `assets.cpp::pick()` already prefers
non-int8 files. Quality: equal-or-better WER on 5/7 goldens. The asset
plumbing has since **shipped** (item 4): int8/fp32/fp16 turbo variants on
the mirror + `WHISPERX_ASR_PRECISION` (default fp16 on GPU; Cpu stays
int8). With fp32/fp16, batch 4 ≈ serial and batch 8 regresses
(VRAM pressure), so `WHISPERX_ASR_BATCH_SIZE=1` stays the default and this
item's batching patch is now a dormant nice-to-have, not a lever. The
earlier suspects (IOBinding/PCIe, syncs, arena churn) were each profiled or
code-read and excluded — ledger in `CUDA_DECODE_FINDINGS.md`.

**Corroboration — insanely-fast-whisper's A100 benchmark (150 min of audio):**

| Config | Wall time | RTF |
|---|---|---|
| faster-whisper large-v2, fp16, beam 1 (serial) | 9 min 23 s | 0.063 |
| large-v3 HF, fp16 + batch 24 + BetterTransformer | 5 min 2 s | 0.034 |
| **large-v3 HF, fp16 + batch 24 + Flash Attention 2** | **1 min 38 s** | **0.011** |
| distil-large-v2, fp16 + batch 24 + FA2 | 1 min 18 s | 0.009 |

The first row is the lesson: serial CTranslate2 — a *faster* engine per call —
loses **5.7×** to batched vanilla PyTorch. Batching dominates engine choice,
and it also gates item 5's fused-attention multiplier (kernels only pay once
sequences are long and batched). lightning-whisper-mlx shows the same lesson
holds on Apple GPUs: its claimed "10× over whisper.cpp" is batched MLX vs
serial whisper.cpp — see item 3 for the consequence.

### 2. Russian path — GigaAM via sherpa-onnx (speed **and** quality)

The sherpa-onnx model zoo ships GigaAM CTC/RNN-T Russian models in ONNX
(NeMo-format recipes). Published SberDevices numbers (echoed by the report):
≈8% WER average on hard Russian domains vs ≈25% for Whisper `large-v3` — and a
Conformer-CTC forward is much cheaper than a Whisper encoder/decoder loop.

The compounding win: sherpa-onnx has **native CTC forced alignment**, so word
timestamps come from the *same* CTC logits — for Russian audio the separate
wav2vec2 alignment stage (RTF ~0.15, roughly a third of pipeline wall-clock)
can be dropped entirely. The language-detect stage already exists to route
`ru` to a different backend. MIT license.

Scope cost: a per-language ASR-backend dimension (model assets, mirror
entries, routing), not just a model swap. Quality gate: WER/CER vs the Whisper
baseline on the Russian goldens; timestamp tolerances from the existing align
parity suite. Watch CTC peak-lag (systematic late boundaries) — validate
word-onset bias against the wav2vec2 goldens before dropping the align stage.

### 3. Apple — land Route B: whisper.cpp Metal as a native ASR backend

**STATUS: LANDED + measured 2026-06-11 (Apple M4).** whisper.cpp v1.8.6 + GGML
Metal is a second ASR backend (`AsrBackend::WhisperCpp` /
`WHISPERX_ASR_BACKEND=whispercpp`, `WHISPERX_ENABLE_WHISPERCPP` build; ggml `.bin`
from the official `ggerganov/whisper.cpp` repo). Measured on the 300 s Russian clip,
`large-v3-turbo` fp16: Stage 1 transcribe RTF **0.073** vs the sherpa CPU int8
baseline **0.236** — **3.2×**; end-to-end RTF **0.199** vs **0.364** — **1.83×**.
Matches the prediction below; align is now the largest stage. Full write-up in
`METAL_INTEGRATION.md` Route B.

Own benchmarks already proved the ROI (`MAC_PERFORMANCE.md`); since ASR
dominates Mac wall-clock, expect ~2× end-to-end. whisper.cpp is C++ and embeds
cleanly — but it is a **new ASR backend dimension** (ggml `.bin` assets, not
ONNX), not a `Device`. Align/diarize stay on ORT CPU.

**Pre-step (was: benchmark batched MLX before committing) — now moot.** Route B
shipped with whisper.cpp and measured 3.2× on Stage 1 / 1.83× E2E. The batched-MLX
question is **demoted**, not pending: with Stage 1 down to 37% and already RTF 0.073,
a batched-MLX rewrite is poor ROI (full reasoning in §"Apple Silicon — most likely
speedups after Route B"). MLX stays watch-list, justified only by a future *streaming*
feature (`MLX_PORT.md`).

The CoreML-EP spike (Route A, landed speculatively in 650d084) stays
timeboxed: evidence says CoreML EP is a wash or regression for transformer ASR
graphs. The one place it might genuinely win is the **diarize embedding
extractor** (conv-heavy ResNet/TDNN maps to ANE far better than transformers)
— worth the cheap measurement before discarding.

Optional Phase 3 on top: whisper.cpp's `WHISPER_COREML` ANE encoder.

### 4. Reduced-precision model variants — fp16 (CUDA) + INT8 (CPU)

**fp16 on `Device::Cuda` — LANDED for Whisper turbo (2026-06-11):**
`golden/convert_whisper_fp16.py` converts the fp32 export offline
(`keep_io_types=True` — sherpa's float32 I/O contract is preserved); the
mirror now carries int8/fp32/fp16 turbo variants (meta.json contract v2,
`golden/mirror_whisper_onnx.py --variant`), selected at runtime by
`WHISPERX_ASR_PRECISION` (default **fp16** on GPU devices; Cpu always loads
int8). Measured: WER identical to fp32 on all 7 golden clips
(`scripts/wer_golden.py` gate), RTF 0.029 vs fp32's 0.030 (overhead-bound at
this point, not FLOPs), half the download/VRAM (1.55 GB vs 3.2 GB).
Still open here: fp16 for the **wav2vec2 align** models (same converter
should apply; align is RTF 0.002 on CUDA so the win is VRAM, not speed).
CPU stays int8/fp32 (MLAS has no fp16 win).

**INT8 on CPU/edge:** sherpa ships `model.int8.onnx` for the embedding
extractors: ~4× smaller, big CPU speedup, near-lossless for speaker
embeddings. For the wav2vec2 aligner, gate on the existing parity tolerances
(timings ±1 frame, scores ±0.01).

**Quantized ggml on Metal (from lightning-whisper-mlx) — MEASURED, thesis didn't
hold for turbo.** The prediction was that decode is memory-bandwidth-bound on Apple
unified memory, making weight quantization a *decode-throughput* lever. Measured (lever 1
above): on `large-v3-turbo`, q8_0 alone is **1.02×** — flash-attention, not quant, drives
Stage 1, because turbo's 4-layer decoder makes decode a small slice. q8_0 stacks a small
near-lossless bonus on flash (→1.26×); q5_0 is over the Russian WER gate. So q8_0's value
is mostly the smaller download + a marginal speed bump, not the bandwidth win predicted.

### 5. CUDA — fused attention in the ONNX graphs

insanely-fast-whisper's single biggest same-model jump (5:02 → 1:38, 3.1×)
came purely from a better attention kernel (BetterTransformer → FA2). We don't
author kernels; the ORT equivalent is fused attention contrib ops
(`MultiHeadAttention`/`Attention` nodes) baked into the graph by the
onnxruntime transformers optimizer. Check whether sherpa's stock Whisper
exports already carry them before assuming the CUDA EP leaves 3× on the
table; if absent, run the optimizer over our mirrored models. Multiplier on
item 1 — worthless while decode is serial.

### 6. Optional product idea — "fast timestamps" mode

Whisper-native word timestamps (cross-attention DTW, as in HF
`return_timestamps="word"`; whisper.cpp exposes token-level timestamps too)
would skip the align stage (RTF ~0.15) entirely, for **all** languages. The
quality trade is the founding premise of whisperX — attention-DTW boundaries
are noticeably worse than wav2vec2 forced alignment — so never the default;
an opt-in speed mode, clearly labeled lower precision. The Metal backend
(item 3) gets the input for free.

## Rejected ideas (for this repo)

| Idea | Verdict |
|---|---|
| Blind 30 s chunk+stride instead of VAD (insanely-fast-whisper) | Transcribes silence (hallucination risk) and cuts words at arbitrary points — VAD-gated chunking exists for a reason and our VAD is cheap, CPU-pinned. Batch the VAD output instead (item 1). Only wins on near-100%-speech audio. |
| distil-whisper | English-only — our own benchmark caught it mis-transcribing Russian as English (`MAC_PERFORMANCE.md`). `large-v3-turbo` (pruned decoder, multilingual) is the correct analogue and already our default mirror. |
| PyTorch MPS path on Mac | insanely-fast-whisper's own caveats (batch capped at 4, "way more memory hungry", no benchmark published) confirm it; whisper.cpp Metal (RTF 0.070) stays our Apple route. |
| Qwen3-ASR / LLM-ForcedAligner | SLLM, VRAM-heavy, no ONNX/sherpa path — wrong fit for a local-first single-binary server. Watch only. The report's RTF/accuracy claims here are unverifiable; treat as marketing. |
| MLX (serial or batched) | whisper.cpp Metal beats *serial* MLX 1.57–2× (`MAC_PERFORMANCE.md`). **Batched** MLX was the open question — now **demoted** post-Route-B: Stage 1 is 37% and RTF 0.073, so a batched-MLX rewrite is poor ROI (see §"most likely speedups after Route B"). Watch-list; `MLX_PORT.md` streaming only. |
| lightning-whisper-mlx wholesale | Python, stale since 2024-05, blind chunking, seek-rewind dropped, no align/diarize. A source of two ideas (batched MLX decode, quantized weights) and one reference file (`decoding.py`), not a dependency. Speculative decoding advertised there never landed — ignore. |
| NeMo MSDD / Sortformer | A second CUDA-only stack duplicating sherpa diarization for marginal DER. |
| Montreal Forced Aligner | Ground-truth/golden tooling only; CPU-bound Viterbi, fails on imperfect transcripts. Never production. |
| gigastt | A competing Rust server — we have our own host. Steal its INT8 idea (item 4), nothing else. |
| pyannote community-1 | Quality (DER) item, not speed; CC-BY-4.0 weights. Revisit only on diarization-quality complaints. |

## Suggested order

This order is **CUDA-centric and partly historical**. For Apple Silicon, item 3
(Route B) is **done** and re-ranked everything — see §"Apple Silicon — most likely
speedups after Route B" for the current answer (quant ggml + flash-attn → fast-timestamps
/ GigaAM → INT8 diarize; MLX demoted).

**CUDA / cross-platform order — 1 → 5 → 3 → 2 → 4 → 6:** batched CUDA decode first
(cheapest big win, contained in one file), then the fused-attention graph
check (its multiplier — do it while measuring item 1), then the
Metal backend (now landed), GigaAM (biggest combined win but adds a model-routing
dimension), reduced precision (incremental: fp16 rides the CUDA work,
quantized ggml rides Metal, INT8 is CPU/edge), fast-timestamps mode last
(product call, not pure engineering).

Note on sourcing: the research report mixes verifiable claims (pyannote DER
tables, GigaAM WER, MFA tolerance benchmarks) with unverifiable ones
(LLM-ForcedAligner AAS reductions, Qwen3 throughput) — treat the latter as
directional. insanely-fast-whisper and lightning-whisper-mlx were read in
full (both are tiny); their benchmark claims are reproduced as published, not
re-run. Items above were cross-checked against this tree's own briefs and
benchmarks where possible.
