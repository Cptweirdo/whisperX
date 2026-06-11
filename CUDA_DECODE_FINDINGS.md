# CUDA Decode Findings — root cause isolated (item 1a investigation)

Companion to `CUDA_DECODE_HANDOFF.md` (the mission brief — now **superseded by
this file**). Investigation run 2026-06-11 on branch `cpp-core/host-swap-server`,
3080 Ti 12 GB, Windows. Reference workload: 607 s `build\e2e_long.wav`
(golden `en_dialog.wav` ×10), `large-v3-turbo`, fixed `language=en`.
Harness: `scripts/bench_cuda_decode.py` (pybind module directly, synthetic
29.5 s spans — reproduces the server's transcribe-stage RTF without
upload/VAD/align/diarize noise).

## Verdict

**The model assets were the problem, not the decode loop.** The mirrored
`large-v3-turbo` is the **int8 dynamic-quantized** export
(`turbo-{encoder,decoder}.int8.onnx` — the only variant sherpa's official
release tarball ships). ORT's CUDA EP has no kernels for
`DynamicQuantizeLinear`/`MatMulInteger`, so **every heavy matmul runs on the
CPU EP** with Memcpy ping-pong at each partition boundary — and
`model_manager.cpp::threads_for(Cuda) == 1` runs that CPU work
**single-threaded**. "CUDA" mode was a single CPU core doing int8 GEMMs with a
GPU doing elementwise glue: slower than plain CPU mode on the same machine.

Swapping in the **fp32 export** (csukuangfj/sherpa-onnx-whisper-turbo) fixes
everything with **zero code changes** — `assets.cpp::pick()` already prefers
non-int8 files in a model dir.

## Measurements

Bench (60 s screen unless noted; full-file numbers marked ★):

| Model | Provider | Threads | Batch | RTF | Note |
|---|---|---|---|---|---|
| int8 | cuda | 1 | 1 | 0.542 / **0.501 ★** | production config; server measured 0.443 with real VAD spans |
| int8 | cuda (no IOBinding) | 8 | 1 | 0.244 | accidental A/B: threads dominate, IOBinding secondary |
| int8 | cuda | 8 | 1 | 0.156 | 3.5× from threads alone — CPU-bound confirmed |
| int8 | cpu | 8 | 1 | 0.269 | plain CPU **beats** production "cuda" |
| **fp32** | **cuda** | **1** | **1** | **0.030 ★** | **16.8× vs int8-cuda production** (18.15 s vs 304.5 s) |
| fp32 | cuda | 1 | 4 | 0.029 ★ | batching is a wash at fp32 |
| fp32 | cuda | 1 | 8 | 0.109 ★ | regresses — VRAM/arena pressure at 8× fp32 activations on 12 GB |

**E2E (scratch server :8011, `WHISPERX_SHERPA_MODELS_ROOT` → fp32 dir):**
`stage=transcribing elapsed=21.81s rtf=0.036 device=cuda` vs the 2026-06-11
baseline `269.25s rtf=0.443` — **12.3× faster through the full server path**,
language detection correct, 123 segments. Transcribe is no longer the
bottleneck (diarize, 37.7 s on CPU, now dominates the job).

**Quality (WER vs golden references, CUDA, both turbo variants):** fp32 is
equal or better on 5/7 clips (ru_dialog 0.163→0.122, de_dialog 0.024→0.008,
ru_cv_71085 0.091→0.000, en_dialog 0.133→0.124), noise-level worse on two
short CV clips (ru_cv_71379 0.050→0.100, ru_cv_71606 0.111→0.167). Mean WER
0.082 → 0.074. No quality cost.

## Hypothesis ledger (final)

| ID | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| **H0** | int8 model → CPU-EP matmuls, single-threaded | **CONFIRMED — root cause** | ORT profile (60 s run, 8 threads): encoder session 5.27 s of 6.84 s kernel time = CPU `MatMulInteger` + 0.35 s CPU `DynamicQuantizeLinear` + 0.74 s `MemcpyFromHost`; decoder session ~257 CPU node-execs/step. fp32 swap → 16.8× |
| H-PCIE | Per-step KV round-trips host (no IOBinding) | EXCLUDED (code read) | sherpa v1.13.2 device-binds KV (`use_cuda_iobinding_`); only logits cross PCIe |
| H-CUDNN | cudnn exhaustive conv search | EXCLUDED (code read) | `session.cc` sets `OrtCudnnConvAlgoSearchHeuristic` |
| H5 | CPU-assigned nodes → hidden Memcpy per step | CONFIRMED as H0's mechanism | profile shows Memcpy nodes at every int8 partition boundary |
| H2 | Per-token syncs + fresh `Ort::IoBinding` | EXCLUDED as root cause | fp32 serial hits RTF 0.030 *with* the per-step sync/binding intact; IOBinding-off (int8, 8 thr) was 0.244 vs 0.156 on — secondary effect only |
| H1 | Per-step GPU output alloc / arena churn | EXCLUDED as root cause; **open** for the fp32 batch-8 regression (0.109, likely arena/VRAM at 8×fp32 activations) | int8 batch-8 VRAM delta explained by CPU-side int8 work scaling with batch |
| H3 | fp32 unfused encoder is slow | EXCLUDED | fp32 encoder is fine: RTF 0.030 serial. fp16/fused attention (SPEEDUP items 4/5) remain optional upside, not a blocker |
| H4 | CPU phases (mel/normalize/argmax/self-KV upload) | EXCLUDED as dominant | contained within the 0.030 |
| H6 | Our windowing loop in `core/asr/whisper_sherpa.cpp` | EXCLUDED | same loop runs in every config; fp32 0.030 |

Also explained: the original batch-8 regression (0.526 vs 0.443) — batching
multiplied the **CPU** int8 matmul work per Run while the 1-thread CPU EP was
the bottleneck; batching a CPU-bound workload through a bigger GEMM on one
core can only lose.

## The fix (SHIPPED 2026-06-11)

1. **Mirror variants:** `KonstantK/whisper-onnx-sherpa/large-v3-turbo` now
   carries **int8 + fp32 + fp16** (meta.json contract v2: flat v1 keys stay =
   int8 for old binaries/CPU; a `variants` block + filename-keyed `sha256` map
   describe the rest — publisher: `golden/mirror_whisper_onnx.py --variant`).
   fp32 came from csukuangfj/sherpa-onnx-whisper-turbo (encoder graph + 2.6 GB
   external `.weights`); fp16 from `golden/convert_whisper_fp16.py`
   (`keep_io_types=True`, single-file 1.3 GB encoder — NB models >2 GB
   silently come back EMPTY from the converter's in-memory shape inference;
   the script does file-based inference first).
2. **Runtime switch:** `WHISPERX_ASR_PRECISION` (`fp16`|`fp32`|`int8`),
   **default fp16**, applies to GPU devices; **Cpu always loads
   int8-preferred** (int8 is CPU-optimal — the bug was int8 on the *GPU*).
   Plumbing: `config.{hpp,cpp}` (Precision enum), `models/assets.cpp`
   (precision-ranked `pick()`, fallback fp16→fp32→int8 etc., int8 last on GPU),
   `assets/downloader.cpp::ensure_whisper_dir` (fetches the variant's file set
   incl. `files` extras), `model_manager.cpp::build_asr_engine` (forces Int8 on
   Cpu, logs `precision=` + encoder filename).
3. **Validation:** fp16 WER identical to fp32 on all 7 golden clips
   (`scripts/wer_golden.py --gate-fp32`); full-file RTF 0.029 (fp32: 0.030 —
   overhead-bound now, the fp16 win is the 1.55 GB vs 3.2 GB download/VRAM).
4. **Knobs:** `WHISPERX_ASR_BATCH_SIZE` default 1 is correct (batch 4 is a
   wash, batch 8 regresses on 12 GB). The sherpa batched-decode patch stays
   harmless behind it. `threads_for(Cuda)=1` is fine once models are fp16/fp32.

## Loose ends

- fp32 batch-8 regression (RTF 0.109): likely ORT CUDA arena growth at 8×
  encoder activations near the 12 GB ceiling — only worth chasing if batching
  is ever needed (it isn't, at 0.030 serial).
- The bench harness (`scripts/bench_cuda_decode.py`) is kept: model-variant ×
  provider × threads × batch A/Bs in minutes, no server.
- ORT profiling without code changes: `--provider "cuda:build\ort_prof.config"`
  with `ProfilingFilePrefix=...` (NB this form disables sherpa's whisper
  IOBinding — `IsCudaProvider()` matches `"cuda"` exactly; fine for placement
  analysis, don't use it for timing comparisons).
