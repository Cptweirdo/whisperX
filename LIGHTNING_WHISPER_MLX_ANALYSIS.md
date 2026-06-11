# lightning-whisper-mlx — Speedup Ideas Analysis

Source: https://github.com/mustafaaljadery/lightning-whisper-mlx (cloned, full
read — ~2.8 k lines, a fork of Apple's `mlx-whisper` example with a batched
decode loop bolted on). **Stale: last commit 2024-05-08**; the advertised
"coming soon" speculative decoding never landed. Companion to
`SPEEDUP_FINDINGS.md` (key takeaways folded there) and `MLX_PORT.md` /
`METAL_INTEGRATION.md` (our Apple briefs).

## What it actually is

Apple's reference MLX Whisper with three changes, all visible in
`transcribe.py` + `decoding.py`:

1. **Batched decoding** — the headline. Audio is sliced into blind sequential
   30 s mel windows, stacked `batch_size` (default 12) deep
   (`transcribe.py:403-422`), and decoded in one batched greedy loop with a
   batched KV cache (`decoding.py` carries `n_batch` through every step;
   `rearrange_kv_cache` exists for beam reorder).
2. **4-bit / 8-bit quantized weights** — pre-quantized `mlx-community/*-mlx-{4,8}bit`
   checkpoints. On Apple unified memory, decode is memory-bandwidth-bound, so
   weight quantization is a *decode-throughput* lever, not just a size lever.
3. **Distilled models** — distil-whisper variants (English-only).

Claimed: "10× faster than whisper.cpp, 4× faster than mlx-whisper". The claim
is *decoding throughput* (their chart), not end-to-end RTF, and is not
independently verified.

## The quality price they paid for batching

Worth listing because it shows which trades are inherent to batching and which
are just this repo being quick-and-dirty:

- **Seek-rewind dropped.** OpenAI's transcribe loop re-seeks to the last
  emitted timestamp token so a window that ended mid-sentence is re-decoded
  from the right offset. Lightning always advances a full 30 s
  (`transcribe.py:404`, `seek += N_FRAMES` unconditionally) — boundary words
  can be lost or duplicated. *Not inherent* — whisperX/our pipeline avoids
  this by cutting at VAD boundaries instead.
- **Cross-window prompt conditioning degraded.** `condition_on_previous_text`
  sets the prompt once per *batch*, so the 12 windows in a batch share a stale
  prompt. *Inherent* to any parallel decode — and already the accepted trade
  in whisperX-style VAD batching (chunks decode independently).
- **Fallback re-decode is per-segment serial.** Failed segments
  (compression-ratio / logprob thresholds) are re-decoded one at a time at
  higher temperature (`transcribe.py:212-218`) — worst case degrades back to
  serial. Same shape as any batch+fallback design.
- **Blind chunking, no VAD** — silence is transcribed; hallucination risk.

## Key ideas mapped to our tree

### 1. Batching wins on Apple GPUs too — and it reorders our Metal ranking

The "10× over whisper.cpp" claim is the same lesson as
insanely-fast-whisper's table (`SPEEDUP_FINDINGS.md` item 1): **batched
mediocre engine beats serial fast engine** — whisper.cpp has no batched
decode, MLX does. Our own `MAC_PERFORMANCE.md` ranking (whisper.cpp Metal >
MLX, 1.57–2×) compared *serial vs serial*; on long multi-chunk files a
batched MLX decode could overturn it.

**Action (cheap, before committing native work):** benchmark Python
`lightning-whisper-mlx` (batch 12) against whisper.cpp Metal on the 300 s
Russian clip from `MAC_PERFORMANCE.md`. If batched MLX wins end-to-end at
equal quality, the Metal Route B backend choice (`METAL_INTEGRATION.md`)
should be revisited — MLX is usable from C++ (per `MLX_PORT.md`), and
lightning's `decoding.py` is a complete reference for a batched KV-cache
decode loop.

### 2. Weight quantization as a decode-bandwidth lever

Their 4/8-bit checkpoints speed up *decode* because autoregressive decoding
streams the full weight matrix per token — bandwidth-bound on unified memory.
The same logic applies to our planned whisper.cpp Metal backend: whisper.cpp
ships quantized ggml variants (`Q5_0`, `Q8_0`) natively. Extends
`SPEEDUP_FINDINGS.md` item 4 beyond CPU/edge INT8: quantized ggml on Metal is
a throughput lever, gated on Russian WER vs the fp16 ggml baseline.

### 3. Batched KV-cache greedy decode — reference implementation

If/when we hand-roll an MLX C++ decode loop (`MLX_PORT.md` already inventories
the model code), lightning's `decoding.py` (729 lines) is the minimal working
reference: batched `GreedyDecoder`, KV cache as a tree of arrays indexed by
batch, per-segment fallback. Smaller and clearer than HF's pipeline for this
purpose.

## Not applicable / rejected

- **Blind 30 s chunking** — same verdict as insanely-fast-whisper: VAD-gated
  chunking stays; batch the VAD output.
- **Distil models** — English-only, already rejected (`MAC_PERFORMANCE.md`
  caught distil mis-transcribing Russian).
- **Adopting the repo wholesale** — Python, stale since May 2024, quality
  trades listed above, no diarization/alignment. It is a source of two ideas
  and one reference file, not a dependency.
- **Speculative decoding** — advertised, never implemented. Ignore.
