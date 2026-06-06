# iOS Port of WhisperX — Options & Decision Doc

> Companion to [`android-port-options.md`](./android-port-options.md). Same goal,
> different platform. Read that doc first for the shared framing — this one only
> calls out where iOS **differs**.

## Context

We want **all four WhisperX stages running fully on-device on iOS** (VAD →
batched transcribe → word-level alignment → speaker diarization), offline, no
server. The same hard constraint as Android applies: **WhisperX is a Python
library** glued to PyTorch / torchaudio / CTranslate2 / pyannote and an **ffmpeg
subprocess** (`whisperx/audio.py:44`), none of which ship for iOS. There is **no
Chaquopy/Termux equivalent** on iOS, so "running the Python package" is even more
firmly off the table than on Android.

So the strategy is identical to the Android plan: **treat WhisperX as the
reference spec, run each model through an Apple-native engine, and re-implement
WhisperX's pure-Python glue in Swift.**

### What iOS changes vs. Android (this is the important part)

1. **iOS already has a mature, Apple-native equivalent of most of WhisperX.**
   Argmax's MIT-licensed Swift SDK (`argmaxinc/argmax-oss-swift`, v1.0.0 May
   2026 — formerly `WhisperKit`) ships, in one Swift Package:
   - **WhisperKit** — on-device Whisper on **Core ML / Apple Neural Engine**,
     with built-in **word timestamps + VAD + real-time streaming**. No PyTorch,
     no Python, no manual conversion.
   - **SpeakerKit** — **pyannote diarization**, merged onto WhisperKit segments
     to produce speaker-attributed output.
   This covers **3 of the 4 stages (VAD, transcribe, diarize) off the shelf**,
   ANE-accelerated, in idiomatic Swift.
2. **The repo already runs on Apple silicon for transcription.** `whisperx/asr_mlx.py`
   (MLX backend) and `whisperx/asr_whispercpp.py` (whisper.cpp, Metal/Core ML)
   exist and are shipped in the macOS Tauri bundle. So Whisper-on-Apple is
   already proven inside this project; an iOS port inherits that lineage.
3. **Audio I/O** is `AVAudioEngine` / `AVAudioFile` / `AVAudioConverter`
   (decode + resample to PCM 16 kHz mono float) instead of ffmpeg — first-class,
   no native lib to vendor.
4. **The one genuinely custom piece is the same as Android**: WhisperX's
   **wav2vec2 forced alignment**. WhisperKit's word timestamps are *Whisper-native*
   (cross-attention/DTW), **not** the wav2vec2-CTC alignment that is WhisperX's
   accuracy headline. If we require WhisperX-grade alignment we still export a
   wav2vec2-CTC model (to **Core ML** via `coremltools`, or **MLX**) and **port
   the same `get_trellis`/`backtrack`/interpolation algorithm** from
   `whisperx/alignment.py:240-490`. That port is **pure algorithm and identical
   across iOS and Android** — write it once.

### Portability table (deltas from the Android doc)

| WhisperX piece | iOS path |
|---|---|
| Audio load (ffmpeg, `audio.py:44`) | **AVFoundation** (`AVAudioFile`/`AVAudioConverter`) → PCM 16 kHz mono float |
| Mel spectrogram (`audio.py:112`) | `vDSP`/Accelerate FFT, or baked into the Core ML graph |
| VAD (`vads/`) | **WhisperKit** built-in VAD, or silero ONNX via sherpa-onnx, or Core ML |
| Whisper transcribe (`asr.py`) | **WhisperKit (Core ML/ANE)** — or reuse repo's MLX / whisper.cpp backend |
| **Forced alignment** (`alignment.py`) | wav2vec2 → **Core ML / MLX** + **shared Viterbi port** (same code as Android) |
| **Diarization** (`diarize.py`) | **SpeakerKit** (pyannote) — or sherpa-onnx pyannote-seg-3.0 + CAM++ |
| `get_trellis`/`backtrack`/merge (`alignment.py:425-490`) | **Port verbatim to Swift** (pure numpy → Swift) |
| `assign_word_speakers` + `IntervalTree` (`diarize.py:14-263`) | **Port verbatim to Swift** |
| Subtitle/segment formatting (`SubtitlesProcessor.py`, `utils.py`) | Port as needed |

---

## The options

### Option A — Argmax OSS Swift (WhisperKit + SpeakerKit) + custom wav2vec2 alignment  ✅ Recommended for an iOS-first app

Adopt `argmax-oss-swift` for VAD + transcribe + diarization (Core ML / ANE), and
add **only** the wav2vec2 forced-alignment stage ourselves (Core ML or MLX export
+ the shared Viterbi port). Use WhisperX as the spec for the alignment glue and
the `schema.py` result contract.

**Pros:** Most iOS-idiomatic; runs on the **Neural Engine** (best battery/latency
on iPhone); 3/4 stages are maintained, MIT-licensed Swift with word timestamps +
VAD + streaming already solved; minimal custom code (alignment only).
**Cons:** Apple-only (no code shared with the Android effort); we're betting on
Argmax's roadmap/model coverage; alignment export + numerical-parity validation
is still ours to own; SpeakerKit's diarization output must be reconciled with our
aligned words via the ported `assign_word_speakers`.

### Option B — sherpa-onnx (Swift) as a cross-platform shared core  ✅ Recommended if shipping iOS **and** Android

sherpa-onnx runs on **iOS and Android** (Swift + Kotlin bindings, same ONNX
models) and provides **VAD + Whisper + speaker diarization** (pyannote-seg-3.0 +
CAM++). Build the pipeline once on sherpa-onnx, share it across both platforms,
and add the wav2vec2 alignment (shared ONNX model + shared Viterbi port).

**Pros:** **One engine, one set of models, one porting effort across iOS +
Android** — directly reuses the Android plan; no per-platform ML stack; fully
offline. **Cons:** ONNX Runtime on iOS doesn't use the ANE as effectively as
Core ML/WhisperKit (CPU/GPU via Core ML EP) — likely higher latency/energy than
Option A on iPhone; less "native Apple" feel; Swift/C++ bridging-header setup.

### Option C — MLX-swift / Core ML hand-rolled (reuse repo's MLX backend as reference)

Run Whisper + wav2vec2 via **MLX-swift** and/or Core ML directly, using
`whisperx/asr_mlx.py` as the reference for the Whisper decode path. Diarization
via a Core ML/MLX pyannote port (cf. `soniqo/speech-swift`, which ships
Apple-Silicon ASR/VAD/diarization on MLX + Core ML, and a wav2vec2+CTC Core ML
model — proof the alignment model converts cleanly).

**Pros:** Maximum control; aligns with the project's existing MLX investment;
MLX runs on iPhone today. **Cons:** Most engineering — we own all four stages,
including a diarization port that Options A/B get for free; largest validation
surface.

### Option D — whisper.cpp + Core ML + sherpa-onnx diarization, skip wav2vec2

Mirror of the Android "Option C": use whisper.cpp (Core ML encoder on iOS) with
its **own** word timestamps and skip wav2vec2 alignment; add sherpa-onnx for
diarization. The repo's `asr_whispercpp.py` is the reference.

**Pros:** Drops the hardest custom piece; whisper.cpp Core ML on iOS is
well-trodden. **Cons:** Whisper-native timestamps are **less accurate than
WhisperX's wav2vec2 alignment** — which is the whole reason WhisperX exists;
diverges from the result contract.

### Option E — Python (torch/ctranslate2/pyannote) on iOS  ❌ Rejected

Even less viable than on Android: no Chaquopy/Termux, no torch/ctranslate2/pyannote
iOS wheels, no ffmpeg CLI. Closed off here for completeness.

---

## Recommendation

It hinges on one question — **iOS-only, or iOS + Android?**

- **iOS-first / iOS-only → Option A** (Argmax OSS Swift + custom wav2vec2
  alignment). Best on-device performance (ANE), least code, most maintainable on
  Apple.
- **Shipping both platforms → Option B** (sherpa-onnx shared core). Accept some
  iOS latency/energy cost in exchange for a **single pipeline + single porting
  effort** reused verbatim from the Android plan. Given an Android effort is
  already underway, this is likely the pragmatic choice; the Apple-native polish
  of Option A can come later if energy/latency demands it.

**Either way, the wav2vec2 forced-alignment stage is the same shared deliverable**
— a model export (Core ML/ONNX/MLX) plus the pure-algorithm Viterbi port from
`alignment.py`. Build it once, platform-agnostically, and validate to numerical
parity against the Python `align()`.

---

## Suggested sequencing (iOS PoC)

1. **Audio I/O** — `AVAudioFile` + `AVAudioConverter` → PCM float32 16 kHz mono;
   validate against `whisperx/audio.py`.
2. **Transcribe** — drop in WhisperKit (Option A) or sherpa-onnx (Option B); map
   output to the `TranscriptionResult` shape (`schema.py`).
3. **Alignment PoC** — export one wav2vec2 (English) to Core ML (`coremltools`)
   or MLX; port `get_trellis`/`backtrack`/merge to Swift; **golden-file test**
   word timings vs. Python `align()`.
4. **Diarization** — SpeakerKit (A) or sherpa-onnx (B); port `assign_word_speakers`.
5. **Glue + formatting** — assemble `AlignedTranscriptionResult` end-to-end.

## Verification strategy

Same as the Android doc: **numerical parity** against the Python pipeline on a
fixed clip set (dump CTC emissions / trellis / word timings / diarization turns,
assert within tolerance), plus on-device latency/energy benchmarks on target
iPhones (ANE vs. ONNX-EP is the key comparison between Options A and B). The
existing `tests/` and `whisperx/schema.py` define the contract.

---

## Sources (iOS on-device ecosystem grounding)

- Argmax OSS Swift / WhisperKit + SpeakerKit (Core ML/ANE Whisper, word
  timestamps, VAD, pyannote diarization, MIT, v1.0.0 May 2026):
  <https://github.com/argmaxinc/WhisperKit> ·
  <https://www.argmaxinc.com/blog/whisperkit> ·
  <https://huggingface.co/argmaxinc/whisperkit-coreml>
- sherpa-onnx on iOS (Swift, offline diarization pyannote-seg-3.0 + CAM++,
  cross-platform with Android):
  <https://github.com/k2-fsa/sherpa-onnx> ·
  <https://carlosmbe.medium.com/running-speech-models-with-swift-using-sherpa-onnx-for-apple-development-d31fdbd0898f>
- MLX-swift on iPhone + Apple-silicon speech toolkit (wav2vec2+CTC Core ML):
  <https://github.com/soniqo/speech-swift> ·
  <https://rudrank.com/exploring-mlx-swift-adding-on-device-inference-to-your-app>
- wav2vec2 CTC forced alignment (algorithm reference):
  <https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html> ·
  <https://deepwiki.com/m-bain/whisperX/3.3-forced-alignment-system>
