# Why C++ + ONNX Runtime for the Core

> Rationale for building the engine core in **C++ on ONNX Runtime**, referenced by
> [`cpp-core-spa-architecture.md`](./cpp-core-spa-architecture.md) and
> [`cpp-core-migration-plan.md`](./cpp-core-migration-plan.md). Answers: *why C++,
> and why ORT, for a pipeline of Whisper + wav2vec2 alignment + pyannote
> diarization + silero VAD?*

## The core truth

**No high-level language has a stable, from-scratch implementation of these
models.** Whisper, wav2vec2-CTC, pyannote (segmentation + embedding), and silero
VAD all bottom out in the *same* C/C++ inference engines — **ONNX Runtime**,
**GGML/whisper.cpp**, **LiteRT**, **Core ML**. So the real question is never "which
language reimplements the deps," but:

> **Which language has stable bindings to a runtime that can run all four models,
> plus the pure-algorithm glue?**

Under that framing every "single-language" plan is really **"one language + ONNX
Runtime under the hood."** They differ only in *how native* that coverage is and
*how stable* the bindings are — and on both axes **C++ wins, because the runtimes
*are* C++.**

- **C++ is the runtime layer itself.** sherpa-onnx (VAD + Whisper + diarization) +
  ONNX Runtime (wav2vec2 alignment) + a C audio decoder + your Viterbi = the whole
  pipeline, natively, with **no FFI tax** and the highest stability (these libs are
  C++).
- Every managed alternative (C#, Dart, JS, Rust) is **binding to that same C++**
  through an interop layer. They can reach the same acceleration, but they add a
  runtime + an FFI seam without removing any of the native dependency.
- The one cost of C++ is ergonomics (manual memory/builds, no UI story) — which the
  architecture absorbs by keeping the core **headless** and pairing it with thin
  adapters (server/SPA, pybind oracle).

## The one constant, in every language

**wav2vec2 forced alignment has no off-the-shelf binding anywhere.** A runtime runs
the wav2vec2 *model*, but the Viterbi trellis + word-timestamp logic
(`alignment.py:425-490`) is yours to write — see
[`pipeline-reference.md`](./pipeline-reference.md) Stage 4 and the
[Phase 3 brief](./cpp-core-migration-briefs.md). No language choice removes this; it
only changes the language you write it in. In C++ it lives next to the runtime with
zero marshalling.

## Why ONNX Runtime specifically

- **sherpa-onnx (built on ORT) supplies 3 of the 4 models** off-the-shelf in C++
  (VAD, Whisper, diarization); ORT adds the 4th (wav2vec2). One runtime → one
  threading model, one memory arena, one set of execution providers
  (CUDA/DirectML/Core ML/CPU).
- **Transformer-friendly.** ORT's optimizer was built with transformers in mind
  (symbolic shape inference, transformer fusions, INT8 quantization) — exactly what
  Whisper and wav2vec2 are.
- **The one exception is ASR on Apple GPU**, where ORT has no Metal EP — handled by
  keeping **whisper.cpp/GGML** as a pluggable ASR backend (see
  [`cpp-core-migration-briefs.md`](./cpp-core-migration-briefs.md) §"Runtime &
  acceleration").

## Bottom line

C++ is the **substrate** the inference engines are written in, so it's the most
stable, covers-everything choice with no FFI overhead. Pair the headless C++ core
with thin adapters for UI, and write the one DIY piece — wav2vec2 forced alignment —
directly against the runtime.

## Sources

- ONNX Runtime (C++ API, execution providers): <https://onnxruntime.ai/docs/>
- sherpa-onnx (C++ VAD/ASR/diarization): <https://github.com/k2-fsa/sherpa-onnx>
- wav2vec2 forced alignment (algorithm reference): <https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html>
