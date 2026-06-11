// Native Whisper ASR backend on whisper.cpp + GGML Metal (METAL_INTEGRATION.md
// Route B). The Apple-GPU analog of WhisperSherpa: same AsrEngine surface, different
// runtime (ggml `.bin` weights, full inference on the Metal backend with unified
// memory). Selected at runtime by AsrBackend::WhisperCpp / WHISPERX_ASR_BACKEND.
//
// Unlike the sherpa backend there is no Device knob here — whisper.cpp runs on the
// Metal GPU via its own use_gpu flag regardless of WHISPERX_DEVICE (which still drives
// align/diarize). Stage 2 re-times every word, so segment timestamps are unused and
// we decode with no_timestamps for speed.
//
// pImpl so this header pulls no whisper.h/ggml headers — only the .cpp sees them.
// Built only under WHISPERX_WHISPERCPP_BUILD (the FetchContent'd whisper lib).
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "asr/asr_engine.hpp"
#include "audio/audio_buffer.hpp"

namespace whisperx::asr {

class WhisperCpp : public AsrEngine {
 public:
    // model_path is a whisper.cpp ggml `.bin` (fp16 or quantized q5_0/q8_0).
    // use_gpu enables the Metal backend (default); flash_attn turns on
    // flash-attention (faster, slightly more memory). num_threads bounds the CPU
    // fallback / non-GPU ops.
    explicit WhisperCpp(const std::string& model_path, int num_threads = 4,
                        bool use_gpu = true, bool flash_attn = false);
    ~WhisperCpp();
    WhisperCpp(WhisperCpp&&) noexcept;
    WhisperCpp& operator=(WhisperCpp&&) noexcept;
    WhisperCpp(const WhisperCpp&) = delete;
    WhisperCpp& operator=(const WhisperCpp&) = delete;

    // Transcribe each VAD span [start_s, end_s) of `audio` — one whisper_full() call
    // per span over the sliced 16 kHz mono PCM, greedy decode, language pinned (or ""
    // = auto-detect per span). Serial: whisper.cpp has no batched decode. Returns one
    // AsrChunk per span, in order.
    std::vector<AsrChunk> transcribe(
        const whisperx::audio::AudioBuffer& audio,
        const std::vector<std::pair<double, double>>& spans,
        const std::string& language = "", const std::string& task = "") override;

    // Whisper language identification over the first 30 s — returns the bare code.
    std::string detect_language(
        const whisperx::audio::AudioBuffer& audio) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace whisperx::asr
