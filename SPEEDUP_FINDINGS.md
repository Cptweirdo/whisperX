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

Own benchmarks already prove the ROI (`MAC_PERFORMANCE.md`); since ASR
dominates Mac wall-clock, expect ~2× end-to-end. whisper.cpp is C++ and embeds
cleanly — but it is a **new ASR backend dimension** (ggml `.bin` assets, not
ONNX), not a `Device`. Align/diarize stay on ORT CPU.

**Pre-step (cheap, from `LIGHTNING_WHISPER_MLX_ANALYSIS.md`):** our whisper.cpp
> MLX ranking compared *serial vs serial*; whisper.cpp has no batched decode,
MLX does, and lightning's claimed 10× is exactly batched-vs-serial. Before
committing native work, benchmark Python `lightning-whisper-mlx` (batch 12)
against whisper.cpp Metal on the 300 s Russian clip from `MAC_PERFORMANCE.md`.
If batched MLX wins end-to-end at equal quality, Route B's engine choice
should be revisited — MLX is usable from C++ (`MLX_PORT.md`), and lightning's
`decoding.py` is a complete reference for a batched KV-cache decode loop.

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

**Quantized ggml on Metal (from lightning-whisper-mlx):** autoregressive
decode streams the full weight matrix per token — memory-bandwidth-bound on
Apple unified memory, so weight quantization is a *decode-throughput* lever
there, not just a size lever. whisper.cpp ships quantized ggml variants
(`Q5_0`, `Q8_0`) natively; rides item 3, gated on Russian WER vs the fp16
ggml baseline.

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
| MLX serial (batch pipeline) | Own benchmarks show whisper.cpp Metal beats *serial* MLX 1.57–2×. `MLX_PORT.md` scopes MLX to a future *streaming* feature — but **batched** MLX is back on the table pending the item 3 pre-step benchmark. |
| lightning-whisper-mlx wholesale | Python, stale since 2024-05, blind chunking, seek-rewind dropped, no align/diarize. A source of two ideas (batched MLX decode, quantized weights) and one reference file (`decoding.py`), not a dependency. Speculative decoding advertised there never landed — ignore. |
| NeMo MSDD / Sortformer | A second CUDA-only stack duplicating sherpa diarization for marginal DER. |
| Montreal Forced Aligner | Ground-truth/golden tooling only; CPU-bound Viterbi, fails on imperfect transcripts. Never production. |
| gigastt | A competing Rust server — we have our own host. Steal its INT8 idea (item 4), nothing else. |
| pyannote community-1 | Quality (DER) item, not speed; CC-BY-4.0 weights. Revisit only on diarization-quality complaints. |

## Suggested order

**1 → 5 → 3 (pre-step first) → 2 → 4 → 6**: batched CUDA decode first
(cheapest big win, contained in one file), then the fused-attention graph
check (its multiplier — do it while measuring item 1), the batched-MLX vs
whisper.cpp benchmark next (one afternoon, decides item 3's engine), then the
Metal backend, GigaAM (biggest combined win but adds a model-routing
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
