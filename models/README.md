# Vendored models

## `silero_vad.onnx`

Silero VAD (ONNX) used by the C++ engine-core silero VAD
(`core/audio/vad_silero.cpp`, sherpa-onnx / ONNX Runtime) behind the **`decode`**
stage token. Loaded by `whisperx/vads/silero.py` when `decode ∈ WHISPERX_CORE_STAGES`
(override the path with `WHISPERX_SILERO_ONNX`).

- Source: snakers4/silero-vad (the v5 ONNX export shipped with the
  `silero_vad` PyTorch-Hub package).
- `sha256`: `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`
- Size: ~2.3 MB.

Per the **decoupled-golden rule** (docs/cpp-core-migration-briefs.md, fact 3) this
model's exact segment boundaries are *not* byte-compared against torch silero — only
smoke-/loose-checked (`bindings/test/test_vad_smoke.py`); the pure `merge_chunks`
that consumes its segments is what's golden-tested exactly (against a *fixed* input).
So the precise revision only needs to load in sherpa-onnx's silero VAD.
