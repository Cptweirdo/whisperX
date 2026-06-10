# MLX Port Brief — Streaming Whisper (AlignAtt) in C++ on Apple Silicon

Companion to `GPU_INTEGRATION.md` (CUDA, landed) and `METAL_INTEGRATION.md` (Metal
routes A/B). This brief folds the findings from a component study of
**[Lightning-SimulWhisper](https://github.com/altalt-org/Lightning-SimulWhisper)** — a
Python MLX implementation of *simultaneous* (streaming) Whisper — and answers: what are
its components, which already have a C++ counterpart (in our tree or in whisper.cpp),
and what would have to be hand-rolled for a native port.

**Scope caveat up front:** Lightning-SimulWhisper is a **streaming** engine (live-mic,
emit-while-speaking via the AlignAtt policy). It is *not* a drop-in for our batch
upload→transcribe pipeline. This brief matters in two ways:

1. It is the reference implementation for a future **live transcription** feature in
   the native server.
2. It revises two judgments in `METAL_INTEGRATION.md`: the CoreML-encoder +
   custom-decoder hybrid is proven in the field, and **MLX is usable from C++** — MLX
   is a C++ library (Python is bindings); only the Whisper *model code* needs
   hand-rolling, and this repo is a complete, small reference for exactly that model.

## What the repo is

The transcription engine behind the "Alt" app, open-sourced. SimulStreaming /
Simul-Whisper's **AlignAtt policy** rebuilt on MLX with an optional CoreML encoder:

```
mic/file chunks (16 kHz float)
  └─ VAC: silero VAD (ONNX) state machine — voice start/end gates the engine
       └─ audio ring buffer (≤30 s; evicted audio's tokens roll into text context)
            └─ mel (MLX STFT) → pad/trim to 30 s
                 └─ encoder: CoreML (.mlmodelc, ANE — whisper.cpp-generated) or MLX
                      └─ decode loop, one token per step (MLX decoder, greedy/beam)
                           ├─ cross-attn of alignment heads → most-attended frame
                           │    └─ STOP when attended frame within frame_threshold
                           │       of audio end  ← the AlignAtt policy
                           └─ CIF end-of-word probe → emit last word or truncate it
```

Key claim/architecture lesson: encoder on ANE via whisper.cpp's CoreML artifacts
(~18× claimed), decoder kept in MLX (~15× over PyTorch claimed) because the policy
needs **per-step cross-attention** and full control of the decode loop — exactly the
thing closed `whisper_full()`-style APIs don't give you.

## Component inventory

| Component | Where (repo paths) | What it does |
|---|---|---|
| **AlignAtt engine** | `simul_whisper/simul_whisper.py` (741 ln, `PaddedAlignAttWhisper.infer`) | The heart. Audio ring buffer (≤`audio_max_len` 30 s; evicting the oldest segment rolls its tokens into the text context), mel → pad 30 s → encode → token-by-token decode. Each step: softmax cross-attn of alignment heads → z-norm → median filter (width 7) → mean over heads → argmax = most-attended frame. Stop when `content_mel_len − frame ≤ frame_threshold` (CLI default 25 frames = 0.5 s; 4 on final flush). Rewind guard (`rewind_threshold` 200 frames) discards a hypothesis when attention jumps backwards. |
| **MLX Whisper model** | `simul_whisper/mlx_whisper/whisper.py` (285 ln) | Vendored mlx-whisper fork, **modified: `TextDecoder.__call__` returns per-layer `cross_qk`** + external KV cache list. Alignment-head IDs are the OpenAI base85 dumps baked into the file (whisper.cpp GGUF embeds the same). |
| **Decoding** | `simul_whisper/mlx_whisper/decoding.py` (946 ln, mostly unused paths) | `GreedyDecoder` (argmax), `BeamSearchDecoder` (KV-cache rearrange on beam reorder), `SuppressTokens` logit filters, `detect_language`. Engine defaults to greedy. |
| **CIF end-of-word probe** | `simul_whisper/eow_detection.py` (190 ln) | Tiny: **one `Linear(n_audio_state→1)` + sigmoid** over encoder frames, scale to round(sum), cumsum integrate-and-fire; "fires" iff last integration peak lands within 2 frames of chunk end. Decides whether the last decoded word is complete; if not it's truncated (mid-word output is usually wrong). Checkpoint = tiny npz (per-model, from upstream simul_whisper; none for large-v3). |
| **CoreML encoder wrapper** | `simul_whisper/coreml_encoder.py` (263 ln) | coremltools glue over whisper.cpp-generated `ggml-<model>-encoder.mlmodelc`/`.mlpackage`. Input `(1, n_mels, 3000)` fp32 → features `(1, 1500, n_state)`. Pure glue — no logic. |
| **Tokenizer** | `simul_whisper/mlx_whisper/tokenizer.py` (402 ln) | tiktoken + Whisper special tokens; `split_to_word_tokens` / `split_tokens_on_unicode` (word boundaries for truncation, timestamps, and the incomplete-UTF8 fix). |
| **Context buffer** | `token_buffer.py` (67 ln) | Scrolling text context behind `sot_prev`; word-level trimming; static-prompt prefix that never scrolls. |
| **Mel frontend** | `simul_whisper/mlx_whisper/audio.py` (173 ln) | STFT (mx.fft) + mel filterbank from an npz asset; same constants as ours (16 kHz, n_fft 400, hop 160, 3000 frames/30 s). |
| **VAC streaming layer** | `whisper_streaming/vac_online_processor.py` (113 ln) + `silero_vad_iterator.py` (123 ln) | State machine over a streaming silero VAD (onnxruntime): voice-start flushes buffered audio into the engine, voice-end (≥`vad_silence_ms`) forces utterance finish. Keeps 1 s of pre-roll. |
| **Online wrapper** | `simulstreaming_whisper.py` (311 ln, `SimulWhisperOnline`) | `process_iter()`: insert chunk → `infer()` → hide trailing incomplete-unicode token → word timestamps = most-attended frame × 0.02 s → `(beg, end, text)`. |
| **Drivers** | `whisper_streaming/whisper_server.py`, `whisper_online_main.py`, `line_packet.py` | TCP server + file-simulation harness. Throwaway for our purposes (we have the oat++ host). |
| Out of scope | `translate/` (LLM translation), `mlx_whisper/{transcribe,cli,writers,torch_whisper}.py` | Unused by the engine. |

Engine-core dependencies are small: `mlx`, `tiktoken`, the mel-filter npz, the CIF npz,
silero ONNX. (`torch`/`torchaudio` in requirements serve only the VAD path + legacy
imports — not the model.)

## C++ counterpart mapping

### Already in our tree (`cpp-core/host-swap-server`)

| Theirs | Ours |
|---|---|
| ffmpeg `load_audio` subprocess | `core/audio/decode.cpp` (in-process libav) — better |
| silero VAD (onnxruntime) | `core/audio/vad_silero.cpp` (sherpa, same model) — ours is offline-segments; their streaming `FixedVADIterator` is a small state machine over the same ONNX graph (sherpa also ships a streaming VAD API) |
| server/IO drivers | oat++ host, jobs runner, SSE — superior, reuse |
| model/device management | `ModelManager` + `Device` plumbing — extend, don't rebuild |

### In whisper.cpp (if taken as the C++ substrate)

| Theirs | whisper.cpp |
|---|---|
| mel frontend | native C++ mel ✓ |
| CoreML encoder | `WHISPER_COREML` loads the **same** `.mlmodelc` artifacts — their generator script literally shells out to whisper.cpp's ✓ |
| decoder forward + KV cache | low-level API (`whisper_encode` / `whisper_decode` / `whisper_get_logits`) — the decode loop *can* be driven externally ✓ |
| tokenizer (encode + decode) | BPE vocab inside the model file; `whisper_tokenize` / `whisper_token_to_str` ✓ |
| alignment-head IDs | embedded in GGUF (DTW aheads) ✓ |
| **per-step cross-attention** | ✗ **the gap** — whisper.cpp computes alignment-head QKs only inside `whisper_full` for DTW timestamps; not exposed per `whisper_decode` step through the public API. AlignAtt needs it every token. |

### MLX from C++

**MLX is C++-native** ([ml-explore/mlx](https://github.com/ml-explore/mlx); Python is a
binding layer). Their decoder uses only Linear / LayerNorm / Conv1d / softmax / concat
— the entire model definition is ~285 lines of Python. A C++ translation against
`mlx::core` is realistic and this repo is the line-for-line reference (including the
one modification that matters: returning `cross_qk`). Weights load from the same
safetensors/npz the MLX community ships.

## Hand-roll difficulty tiers

**Trivial (days each, plain C++, no framework):**
- CIF probe — one matvec + sigmoid + cumsum per chunk; load two tensors from npz.
- Attention post-processing — z-norm, median filter, argmax over a `(heads, T)` slice.
- `GreedyDecoder`, `SuppressTokens` masks.
- `TokenBuffer` / context scroll, the incomplete-UTF8 hide/restore trick.
- VAC state machine (voice start/end gating, 1 s pre-roll).
- Mel — reuse whisper.cpp's, or port `audio.py` (we already have FFT-adjacent DSP in core).

**Moderate:**
- Beam search with KV-cache rearrange on reorder (ship greedy first — their default).
- The audio ring buffer's eviction semantics — evicted audio's tokens must roll into
  the text context in the same step (`insert_audio`), or timestamps/context drift.
  Subtle but fully specified by ~20 lines of Python.
- `split_to_word_tokens` word-boundary logic (port from tokenizer.py, or drive
  whisper.cpp's vocab).

**The one hard decision — where per-step cross-attention comes from:**

1. **whisper.cpp fork/patch.** Expose the alignment-head cross-QKs per `whisper_decode`
   step (the tensors exist internally for DTW; the change is surfacing them). Least new
   code — encoder, Metal/CoreML, tokenizer, KV cache all come for free — but we then
   maintain a patched dependency, and `METAL_INTEGRATION.md` Route B planned vanilla
   whisper.cpp via FetchContent.
2. **Hand-rolled MLX C++ decoder** (this repo's architecture, translated): ~300 lines
   of model + weight loading; encoder stays whisper.cpp-CoreML (route B's optional
   `.mlmodelc` artifact, reused here as the *primary* encoder). Full control, no fork;
   adds an `mlx` dependency and an Apple-only code path. This is exactly the hybrid
   Lightning-SimulWhisper ships, so it is de-risked by example.

Recommendation: decide only when streaming is actually scheduled; prototype (1) first —
if the patch is ~50 lines against a pinned tag, the maintenance cost beats adding a
second ML runtime. Fall back to (2) if the patch fights whisper.cpp's internals.

## Fit with the existing roadmap

- **Batch pipeline (METAL_INTEGRATION.md) is unaffected.** Routes A (CoreML EP spike)
  and B (whisper.cpp Metal for Stage 1) stand. This brief adds: the `WHISPER_COREML`
  encoder artifact planned as route B's Phase 3 add-on is the same artifact a streaming
  port would reuse — shared asset, build it once.
- **Revises the MLX verdict.** `METAL_INTEGRATION.md` said "watch, don't build" because
  no maintained C/C++ Whisper-on-MLX exists. Still true as a *dependency*, but
  hand-rolling the decoder is a bounded, reference-backed task (~300 lines), so MLX
  graduates from "infeasible from C++" to "feasible, justified only by the streaming
  use case." The `mlx_available:false` status placeholder stays until then.
- **A native streaming feature** would slot in as: new engine class (per-utterance
  AlignAtt loop) + a websocket/SSE audio ingest path in the server + the VAC gate in
  front. The existing `Steps`-closure orchestrator is batch-shaped; streaming would be
  a sibling path, not a retrofit.

## Constants & contracts worth keeping (from the source)

- Frame = 20 ms (encoder downsamples 2×: 1500 frames / 30 s). Word timestamps = most-
  attended frame × 0.02 s; made non-decreasing across iterations (`last_ts + 1` clamp).
- `frame_threshold`: CLI default 25 (≈0.5 s lag); forced to 4 on final flush.
- `rewind_threshold`: 200 frames; attention jumping back further than this discards the
  current hypothesis (guard against rewind loops); `DEC_PAD = 50257`.
- `audio_max_len` 30 s / `audio_min_len` gate; `vad_silence_ms` default 500.
- Suppress set: task/sot/language/no-timestamps specials (+ space and EOT on the first
  step of each segment) — prevents the classic streaming hallucinations.
- No-speech early-out: `p(no_speech)` at the SOT position vs `nonspeech_prob`.

## Unknowns / to validate (needs Apple Silicon)

- whisper.cpp low-level path: does `whisper_decode` + external KV management actually
  support the rewind/truncate pattern (re-feeding a shortened token tail) without
  re-encoding? (Their MLX impl just resets the cache every `infer()` — cheap because
  the prompt re-feed is one batched forward; verify the same trick is acceptable in
  whisper.cpp.)
- Size of the cross-QK exposure patch against current whisper.cpp (option 1 feasibility).
- mlx C++ API stability + CMake friendliness as a FetchContent dep (option 2).
- CIF checkpoints: per-model, sourced from upstream simul_whisper, **none for
  large-v3** — without one the engine always truncates the last word (`always_fire`);
  acceptable? (Their default.)
- Real RTF/latency of the hybrid on target hardware vs our batch numbers — streaming
  competes on latency, not throughput.
