// Native Whisper ASR backend (Phase 4 / slice 4a). Wraps sherpa-onnx's
// non-streaming Whisper OfflineRecognizer (ONNX Runtime) — mel + encoder/decoder
// forward + greedy detokenize + language detection all live inside sherpa, so this
// is a thin batched front-end over the VAD chunks (the analog of
// whisperx/asr.py::FasterWhisperPipeline.transcribe, but the decode engine is
// sherpa-onnx Whisper, not CTranslate2).
//
// Decoupled goldens (settled fact 3): Whisper text is not byte-stable across
// decoders, so this backend is judged by WER/CER, not exact text — greedy vs
// faster-whisper beam search is a non-issue. avg_logprob is best-effort (a
// populated field only).
//
// pImpl so this header pulls no ORT/sherpa headers — only the .cpp (and the audio
// stage that builds it) sees the sherpa C API. Built only under WHISPERX_CORE_AUDIO.
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "audio/audio_buffer.hpp"

namespace whisperx::asr {

// One transcribed VAD chunk — the text + a best-effort mean token log-prob (the
// field whisperx/asr.py populates per segment; never load-bearing downstream).
struct AsrChunk {
    std::string text;
    float avg_logprob = 0.0f;
};

class WhisperSherpa {
 public:
    // encoder/decoder/tokens are the three sherpa Whisper ONNX assets;
    // feature_dim is the mel bin count (80 for tiny…medium/large-v2, 128 for
    // large-v3). language/task are defaults ("" language = auto-detect per chunk).
    WhisperSherpa(const std::string& encoder, const std::string& decoder,
                  const std::string& tokens, int num_threads = 1,
                  int feature_dim = 80, const std::string& language = "",
                  const std::string& task = "transcribe");
    ~WhisperSherpa();
    WhisperSherpa(WhisperSherpa&&) noexcept;
    WhisperSherpa& operator=(WhisperSherpa&&) noexcept;
    WhisperSherpa(const WhisperSherpa&) = delete;
    WhisperSherpa& operator=(const WhisperSherpa&) = delete;

    // Transcribe each VAD span [start_s, end_s) of `audio` (one sherpa offline
    // stream per span — Whisper pads every chunk to a fixed 30 s mel, so there is
    // no cross-segment batch-norm hazard; "batch" is a serial loop). `language`
    // (non-empty) overrides the per-chunk language; "" keeps the construction
    // default / auto. Returns one AsrChunk per span, in order.
    std::vector<AsrChunk> transcribe(
        const whisperx::audio::AudioBuffer& audio,
        const std::vector<std::pair<double, double>>& spans,
        const std::string& language = "", const std::string& task = "");

    // Whisper language identification over the first 30 s — returns the bare code
    // (e.g. "en"), mirroring FasterWhisperPipeline.detect_language.
    std::string detect_language(const whisperx::audio::AudioBuffer& audio);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace whisperx::asr
