# Android Port of WhisperX — Options & Decision Doc

## Context

We want **all four WhisperX stages running fully on-device on Android** (VAD →
batched transcribe → word-level alignment → speaker diarization), offline, no
server. We already have a separate project running **LiteRT** inference on
Android with good performance, so a Whisper encoder/decoder forward pass on the
phone is a solved problem for us.

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
**treat WhisperX as the reference spec, run every model through LiteRT/ONNX, and
re-implement WhisperX's pure-Python "secret sauce" natively.** The good news:
the parts that are genuinely WhisperX (not just Whisper) are mostly pure
algorithm and port cleanly.

### What is actually portable vs. what must be replaced

| WhisperX piece | Nature | Android path |
|---|---|---|
| Audio load (`audio.py:44`, ffmpeg) | subprocess | **Replace**: Android `AudioRecord`/`MediaCodec`/`MediaExtractor` → PCM 16 kHz mono float |
| Mel spectrogram (`audio.py:112`, `torch.stft`) | torch | **Replace**: native STFT (Kotlin/C++) or bake into the LiteRT graph |
| VAD (`vads/`) | torch / onnx | **Swap model**: silero-VAD ONNX (sherpa-onnx ships it) + port the batching logic |
| Whisper transcribe (`asr.py`) | CTranslate2 | **Swap engine**: our existing **LiteRT** Whisper |
| **Forced alignment** (`alignment.py`) | wav2vec2 + **pure-numpy Viterbi** | **Swap model + port algorithm** (see below) — *this is the real work* |
| **Diarization** (`diarize.py`) | pyannote + **pure-Python IntervalTree** | **Swap model + port glue** — model is off-the-shelf |
| Trellis/backtrack `get_trellis`/`backtrack` (`alignment.py:425-490`) | **pure numpy** | **Port verbatim** to Kotlin/C++ |
| Word-timestamp interpolation, char→word merge (`alignment.py:293-407`) | pandas/numpy | **Port** to native |
| `assign_word_speakers` + `IntervalTree` (`diarize.py:14-263`) | **pure Python** | **Port verbatim** |
| Subtitle/segment formatting (`SubtitlesProcessor.py`, `conjunctions.py`, `utils.py` writers) | pure Python | **Port as needed** |

The headline: **only two models need a mobile inference engine that we don't
already have** — the wav2vec2 alignment model and the diarization
segmentation+embedding models. Everything WhisperX-specific around them is pure
Python/numpy that ports directly.

---

## The options

### Option A — Native on-device stack: LiteRT (ours) + sherpa-onnx + ported WhisperX algorithms  ✅ Recommended

Build the pipeline natively, picking the best-supported runtime per stage and
porting WhisperX's glue code. WhisperX repo = reference implementation.

- **VAD** — silero-VAD ONNX (already packaged & Android-tested in sherpa-onnx),
  or reuse our LiteRT. Port WhisperX's voiced-segment **batching** logic from
  `whisperx/vads/`.
- **Transcribe** — **our existing LiteRT Whisper** (already performant). Feed it
  the VAD-batched segments.
- **Align** — export a wav2vec2-CTC model (e.g. `WAV2VEC2_ASR_BASE_960H` for
  English, the per-language HF models in `DEFAULT_ALIGN_MODELS_HF` for others) to
  **LiteRT or ONNX**; it's a transformer encoder + linear CTC head, exports
  cleanly. Then **port WhisperX's `get_trellis` / `backtrack` / word-merge /
  interpolation** (`alignment.py:240-490`) to Kotlin/C++ — these are ~150 lines
  of pure numpy. (torchaudio's `forced_align` is the same algorithm if we'd
  rather lean on a reference.)
- **Diarize** — **sherpa-onnx offline speaker diarization, off the shelf**: it
  ships `pyannote-segmentation-3.0` (ONNX) + a 3D-Speaker **CAM++** embedding
  model, runs on Android (prebuilt arm64 APKs exist), no PyTorch/Python. This is
  the *same pyannote segmentation family* WhisperX uses. Then **port
  `assign_word_speakers` + `IntervalTree`** (`diarize.py`) to map turns onto our
  aligned words.

**Pros:** Fully offline; best-of-breed per stage; reuses our LiteRT investment;
diarization is essentially free via sherpa-onnx; the WhisperX-specific IP ports
as pure algorithm with golden-output tests against the Python lib.
**Cons:** Most integration surface — two runtimes (LiteRT + sherpa-onnx/ONNX
Runtime) plus native algorithm code; wav2vec2 export + numerical-parity
validation is the main risk; per-language alignment models multiply model
bundling.

### Option B — sherpa-onnx as the single umbrella runtime (+ our alignment)

sherpa-onnx already provides **VAD + Whisper + speaker diarization** on Android
in one C++/JNI library with Kotlin bindings and prebuilt APKs. Adopt it for
three of four stages; add only the wav2vec2 alignment ourselves (Option A's
align sub-plan) as an ONNX model + ported Viterbi.

**Pros:** One mature native runtime, no Python, fewest moving parts for 3/4
stages, active project. **Cons:** Whisper runs under sherpa's ONNX (we'd shelve
our LiteRT Whisper, or run two engines); alignment still DIY; less control over
the Whisper decode path than our own LiteRT.

### Option C — whisper.cpp word timestamps + sherpa-onnx diarization (skip wav2vec2)

whisper.cpp has Android (NDK) builds and **its own DTW-based word timestamps**
(`--max-len` / token-level). If its word timing is good enough, **drop the
wav2vec2 alignment stage entirely**; add sherpa-onnx for diarization.

**Pros:** Eliminates the hardest custom piece (wav2vec2 export + Viterbi port);
two well-supported native libs. **Cons:** Whisper-native timestamps are
generally **less accurate than WhisperX's wav2vec2 forced alignment** (that
accuracy is precisely *why WhisperX exists*); diverges from the WhisperX result
contract; still need to swap our LiteRT for whisper.cpp or run both.

### Option D — Run the WhisperX Python stack on Android (Chaquopy/Termux)  ❌ Rejected

**Non-starter.** No Android wheels for torch / torchaudio / ctranslate2 /
pyannote; ffmpeg-subprocess audio loading (`audio.py:44`); torchcodec & triton
already excluded on aarch64 in `pyproject.toml`. Cross-compiling this tree is
not viable. Documented here only to close it off.

---

## Recommendation

**Option A**, with **Option B's sherpa-onnx as a strong fallback** for the
non-LiteRT stages (it removes nearly all diarization risk and gives us a
ready-made VAD). Concretely:

1. **Diarization** → adopt **sherpa-onnx** off-the-shelf (lowest risk, fully
   solved on Android). Port only `assign_word_speakers`/`IntervalTree`.
2. **Transcribe** → keep **our LiteRT Whisper**.
3. **VAD** → silero ONNX (reuse sherpa-onnx's) + port WhisperX batching.
4. **Alignment** → the one real build: wav2vec2→LiteRT/ONNX export + native port
   of `alignment.py`'s trellis/backtrack/interpolation, validated to numerical
   parity against the Python lib.

This keeps the WhisperX repo as the **authoritative spec** for the glue code and
result schema (`whisperx/schema.py`), reuses our LiteRT, and leans on a proven
Android runtime for the parts that would otherwise be hardest.

---

## Suggested sequencing (for whichever PoC follows this doc)

1. **Audio I/O** — Android capture/decode → PCM float32 16 kHz mono (replaces
   ffmpeg + `load_audio`). Validate against `whisperx/audio.py` output.
2. **Transcribe** — wire VAD-batched segments into our LiteRT Whisper; match
   `TranscriptionResult` shape (`schema.py`).
3. **Alignment PoC** — export one wav2vec2 (English) to LiteRT/ONNX; port
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
- Whisper on Android via TFLite/LiteRT: <https://github.com/nyadla-sys/whisper.tflite> ·
  <https://github.com/vilassn/whisper_android> · Argmax Pro SDK (Kotlin-first on
  LiteRT): <https://www.argmaxinc.com/blog/argmax-pro-sdk-for-android>
- wav2vec2 CTC forced alignment (algorithm reference):
  <https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html> ·
  <https://deepwiki.com/m-bain/whisperX/3.3-forced-alignment-system>
