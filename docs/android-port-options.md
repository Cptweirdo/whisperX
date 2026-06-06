# Android Port of WhisperX — Options & Decision Doc

> **Runtime decision (committed): ONNX Runtime.** Every model runs on **ONNX
> Runtime** via **sherpa-onnx**; LiteRT is no longer part of the plan. An earlier
> LiteRT prototype proved that an on-device Whisper forward pass is feasible and
> performant on our target phones — that feasibility carries over, but we
> standardize on ONNX Runtime for **one runtime across Android, iOS, and the
> desktop C++ core** (see [`cpp-core-spa-architecture.md`](./cpp-core-spa-architecture.md)
> C2 and [`single-language-runtime-options.md`](./single-language-runtime-options.md)).

## Context

We want **all four WhisperX stages running fully on-device on Android** (VAD →
batched transcribe → word-level alignment → speaker diarization), offline, no
server.

The blocker is that **WhisperX is a Python library**, not a portable engine. Its
inference is glued to native deps that have **no Android story**:

- **PyTorch + torchaudio** (alignment forward pass, mel spectrogram via
  `torch.stft`, VAD) — `whisperx/alignment.py`, `whisperx/audio.py:112`.
- **CTranslate2 / faster-whisper** (the default Whisper backend) —
  `whisperx/asr.py`. C++ wheels, none for Android.
- **pyannote-audio** (diarization + default VAD) — `whisperx/diarize.py`,
  `whisperx/vads/pyannote.py`. Torch-based, gated HF downloads.
- **ffmpeg subprocess** for *all* audio loading — `whisperx/audio.py:44`
  (`subprocess.run(["ffmpeg", ...])`). No CLI on Android.

So "porting WhisperX" does **not** mean shipping the Python package. It means:
**treat WhisperX as the reference spec, run every model through ONNX Runtime, and
re-implement WhisperX's pure-Python "secret sauce" natively.** The good news:
the parts that are genuinely WhisperX (not just Whisper) are mostly pure
algorithm and port cleanly.

### What is actually portable vs. what must be replaced

| WhisperX piece | Nature | Android path |
|---|---|---|
| Audio load (`audio.py:44`, ffmpeg) | subprocess | **Replace**: Android `AudioRecord`/`MediaCodec`/`MediaExtractor` → PCM 16 kHz mono float |
| Mel spectrogram (`audio.py:112`, `torch.stft`) | torch | **Replace**: sherpa-onnx computes Whisper features internally → usually unneeded; else native STFT or bake into the ONNX graph |
| VAD (`vads/`) | torch / onnx | **Swap model**: silero-VAD ONNX (sherpa-onnx ships it) + port the batching logic |
| Whisper transcribe (`asr.py`) | CTranslate2 | **Swap engine**: sherpa-onnx Whisper (ONNX Runtime) |
| **Forced alignment** (`alignment.py`) | wav2vec2 + **pure-numpy Viterbi** | **Swap model + port algorithm** (see below) — *this is the real work* |
| **Diarization** (`diarize.py`) | pyannote + **pure-Python IntervalTree** | **Swap model + port glue** — model is off-the-shelf |
| Trellis/backtrack `get_trellis`/`backtrack` (`alignment.py:425-490`) | **pure numpy** | **Port verbatim** to Kotlin/C++ |
| Word-timestamp interpolation, char→word merge (`alignment.py:293-407`) | pandas/numpy | **Port** to native |
| `assign_word_speakers` + `IntervalTree` (`diarize.py:14-263`) | **pure Python** | **Port verbatim** |
| Subtitle/segment formatting (`SubtitlesProcessor.py`, `conjunctions.py`, `utils.py` writers) | pure Python | **Port as needed** |

The headline: **only two models need an inference engine beyond what sherpa-onnx
ships** — really just the wav2vec2 alignment model (sherpa already provides VAD,
Whisper, and diarization). Everything WhisperX-specific around them is pure
Python/numpy that ports directly.

---

## The options

### Option A — sherpa-onnx (ONNX Runtime) for 3 stages + our wav2vec2 alignment ✅ Recommended

One native runtime (ONNX Runtime, via sherpa-onnx) covers **VAD + Whisper +
speaker diarization** on Android in a single C++/JNI library with Kotlin bindings
and prebuilt APKs. We add only the **wav2vec2 forced-alignment** stage ourselves
(an ONNX model + ported Viterbi). WhisperX repo = reference implementation.

- **VAD** — silero-VAD ONNX (already packaged & Android-tested in sherpa-onnx).
  Port WhisperX's voiced-segment **batching** logic from `whisperx/vads/`.
- **Transcribe** — **sherpa-onnx Whisper (ONNX Runtime)**. Feed it the
  VAD-batched segments. ORT dispatches to NNAPI / GPU / CPU via execution
  providers.
- **Align** — export a wav2vec2-CTC model (e.g. `WAV2VEC2_ASR_BASE_960H` for
  English, the per-language HF models in `DEFAULT_ALIGN_MODELS_HF` for others) to
  **ONNX**; it's a transformer encoder + linear CTC head, exports cleanly and
  runs on the **same ORT instance**. Then **port WhisperX's `get_trellis` /
  `backtrack` / word-merge / interpolation** (`alignment.py:240-490`) to
  Kotlin/C++ — these are ~150 lines of pure numpy. (torchaudio's `forced_align`
  is the same algorithm if we'd rather lean on a reference.)
- **Diarize** — **sherpa-onnx offline speaker diarization, off the shelf**: it
  ships `pyannote-segmentation-3.0` (ONNX) + a 3D-Speaker **CAM++** embedding
  model, runs on Android (prebuilt arm64 APKs exist), no PyTorch/Python. This is
  the *same pyannote segmentation family* WhisperX uses. Then **port
  `assign_word_speakers` + `IntervalTree`** (`diarize.py`) to map turns onto our
  aligned words.

**Pros:** Fully offline; **one runtime** (ORT) for all four models → one
threading model, one memory arena, one set of EPs; diarization + VAD essentially
free via sherpa-onnx; the WhisperX-specific IP ports as pure algorithm with
golden-output tests against the Python lib; **shares the exact runtime with the
desktop C++ core and the iOS port**.
**Cons:** wav2vec2 export + numerical-parity validation is the main risk (it's
the one DIY piece in any plan); per-language alignment models multiply model
bundling.

### Option B — whisper.cpp word timestamps + sherpa-onnx diarization (skip wav2vec2)

whisper.cpp has Android (NDK) builds and **its own DTW-based word timestamps**
(`--max-len` / token-level). If its word timing is good enough, **drop the
wav2vec2 alignment stage entirely**; keep sherpa-onnx for diarization.

**Pros:** Eliminates the hardest custom piece (wav2vec2 export + Viterbi port).
**Cons:** Whisper-native timestamps are generally **less accurate than WhisperX's
wav2vec2 forced alignment** (that accuracy is precisely *why WhisperX exists*);
diverges from the WhisperX result contract; adds a second engine (whisper.cpp/GGML)
alongside ORT, breaking the one-runtime goal.

### Option C — Run the WhisperX Python stack on Android (Chaquopy/Termux) ❌ Rejected

**Non-starter.** No Android wheels for torch / torchaudio / ctranslate2 /
pyannote; ffmpeg-subprocess audio loading (`audio.py:44`); torchcodec & triton
already excluded on aarch64 in `pyproject.toml`. Cross-compiling this tree is
not viable. Documented here only to close it off.

---

## Recommendation

**Option A.** Concretely:

1. **Transcribe** → **sherpa-onnx Whisper (ONNX Runtime)**, the same runtime as
   every other stage.
2. **Diarization** → adopt **sherpa-onnx** off-the-shelf (lowest risk, fully
   solved on Android). Port only `assign_word_speakers`/`IntervalTree`.
3. **VAD** → silero ONNX (sherpa-onnx's) + port WhisperX batching.
4. **Alignment** → the one real build: wav2vec2→**ONNX** export + native port of
   `alignment.py`'s trellis/backtrack/interpolation, validated to numerical
   parity against the Python lib.

This keeps the WhisperX repo as the **authoritative spec** for the glue code and
result schema (`whisperx/schema.py`), runs **everything on one runtime (ONNX
Runtime)**, and leans on a proven Android library (sherpa-onnx) for three of the
four stages.

---

## Suggested sequencing (for whichever PoC follows this doc)

1. **Audio I/O** — Android capture/decode → PCM float32 16 kHz mono (replaces
   ffmpeg + `load_audio`). Validate against `whisperx/audio.py` output.
2. **Transcribe** — wire VAD-batched segments into sherpa-onnx Whisper; match
   `TranscriptionResult` shape (`schema.py`).
3. **Alignment PoC** — export one wav2vec2 (English) to ONNX; port
   `get_trellis`/`backtrack`/merge; **golden-file test** word timings vs. the
   Python `align()` on the same clip.
4. **Diarization** — drop in sherpa-onnx; port `assign_word_speakers`; verify
   speaker labels on a 2-speaker clip.
5. **Glue + formatting** — port segment/subtitle formatting as needed; assemble
   the `AlignedTranscriptionResult` end-to-end.

## Verification strategy

- **Numerical parity** is the core test: for a fixed set of audio clips, run the
  Python WhisperX pipeline and dump intermediate tensors/JSON (CTC emissions,
  trellis, word timings, diarization turns), then assert the native outputs match
  within tolerance. The existing `tests/` (e.g.
  `test_word_timestamp_interpolation.py`) and `whisperx/schema.py` define the
  contract to match.
- Per-stage on-device latency/RAM benchmarks on target hardware.

---

## Sources (on-device ecosystem grounding)

- sherpa-onnx offline speaker diarization (pyannote-segmentation-3.0 + CAM++,
  Android APKs, ONNX, no PyTorch/Python):
  <https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html> ·
  <https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/apk.html>
- sherpa-onnx Whisper on Android (ONNX Runtime, prebuilt APKs):
  <https://k2-fsa.github.io/sherpa/onnx/index.html>
- ONNX Runtime Android (NNAPI execution provider):
  <https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html>
- wav2vec2 CTC forced alignment (algorithm reference):
  <https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html> ·
  <https://deepwiki.com/m-bain/whisperX/3.3-forced-alignment-system>
