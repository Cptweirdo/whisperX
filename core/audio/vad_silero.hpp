// Silero voice-activity detection via sherpa-onnx (ONNX Runtime), producing the
// raw pre-merge segments that whisperx::vad::merge_chunks consumes. This is the
// ORT replacement for whisperx/vads/silero.py's torch.hub silero. Per the
// decoupled-golden rule the segments are NOT chased to torch-silero byte parity
// — only smoke-/loose-boundary checked.
#pragma once

#include <string>
#include <vector>

#include "audio/audio_buffer.hpp"
#include "vad/merge_chunks.hpp"  // whisperx::vad::VadSegment

namespace whisperx::audio {

// Run silero VAD over `audio`; returns {start, end} (seconds) speech segments
// labelled "UNKNOWN" (matching Silero.__call__), ready for merge_chunks.
// `onset` == vad_onset threshold; `chunk_size` == max_speech_duration_s.
std::vector<whisperx::vad::VadSegment> silero_segments(
    const AudioBuffer& audio, const std::string& model_path,
    double onset = 0.5, double chunk_size = 30.0);

}  // namespace whisperx::audio
