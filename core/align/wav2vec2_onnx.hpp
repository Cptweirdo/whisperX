// Native wav2vec2-CTC forward under ONNX Runtime (Phase 3B). Replaces the torch
// model forward at whisperx/alignment.py:278-285 with a raw Ort::Session over the
// parity-pinned mirror graph (KonstantK/wav2vec2-align-onnx, contract v2): inputs
// (waveform (B,N) f32, attention_mask (B,N) int64), outputs (emissions (B,T,V) raw
// logits, frame_lengths (B,) int64). The caller applies log_softmax + wildcard +
// tokenize via emission_post, then align_assemble (3A).
//
// pImpl so this header pulls no ONNX Runtime headers — only the .cpp (and the audio
// stage that builds it) sees ORT. Built only under WHISPERX_CORE_AUDIO.
#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace whisperx::align {

class Wav2Vec2Onnx {
 public:
    explicit Wav2Vec2Onnx(const std::string& onnx_path, int num_threads = 1);
    ~Wav2Vec2Onnx();
    Wav2Vec2Onnx(Wav2Vec2Onnx&&) noexcept;
    Wav2Vec2Onnx& operator=(Wav2Vec2Onnx&&) noexcept;
    Wav2Vec2Onnx(const Wav2Vec2Onnx&) = delete;
    Wav2Vec2Onnx& operator=(const Wav2Vec2Onnx&) = delete;

    // Forward over `waveforms` (each a 16 kHz mono f32 span). Returns per-segment
    // **raw** logits, flat row-major of length T_i * V, with `shapes[i] == {T_i, V}`
    // (resized to waveforms.size()). Each row is padded to its batch max (≥ the conv
    // minimum), fed the 1/0 attention_mask, and trimmed back to frame_lengths[i].
    //   batched=true  — pack segments into padded+masked batches (bucket by length).
    //                   Only valid for layer_norm models (meta["batchable"]).
    //   batched=false — run each segment alone (group_norm models corrupt padded
    //                   batch-mates; the per-segment path has no padding).
    std::vector<std::vector<float>> forward(
        const std::vector<std::span<const float>>& waveforms,
        std::vector<std::pair<std::size_t, std::size_t>>& shapes, bool batched);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace whisperx::align
