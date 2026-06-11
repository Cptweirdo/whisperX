// Backend-neutral ASR engine interface. Stage 1 (Whisper) has two interchangeable
// implementations behind this base: WhisperSherpa (sherpa-onnx / ONNX Runtime, the
// default) and WhisperCpp (whisper.cpp + GGML Metal, METAL_INTEGRATION.md Route B).
// The model manager caches a shared_ptr<AsrEngine> and the job runner drives it
// through this surface, so the decode engine is a runtime choice (AsrBackend /
// WHISPERX_ASR_BACKEND) the rest of the pipeline never sees.
//
// AsrChunk + strip_blank_audio live here (not in a backend header) because both are
// engine-neutral: every backend returns the same per-chunk shape and the same
// "[BLANK_AUDIO]" hygiene applies.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "audio/audio_buffer.hpp"

namespace whisperx::asr {

// One transcribed VAD chunk — the text + a best-effort mean token log-prob (the
// field whisperx/asr.py populates per segment; never load-bearing downstream).
// blank_audio_removed counts the "[BLANK_AUDIO]" markers stripped from `text`
// (see strip_blank_audio) so the host can warn once per job.
struct AsrChunk {
    std::string text;
    float avg_logprob = 0.0f;
    int blank_audio_removed = 0;
};

// Remove every "[BLANK_AUDIO]" marker (case-insensitive, tolerating inner
// whitespace e.g. "[ BLANK_AUDIO ]") from `text`, collapse the double space that
// removal leaves behind, and trim. Returns the count removed. sherpa-onnx Whisper
// emits this literal text on silent input (the Python faster-whisper path dropped
// such segments via no_speech_threshold, which sherpa's C-API does not expose).
int strip_blank_audio(std::string& text);

// Abstract Stage-1 ASR engine. Whisper text is not byte-stable across decoders, so
// implementations are judged by WER/CER, not exact text (see whisper_sherpa.hpp).
class AsrEngine {
 public:
    virtual ~AsrEngine() = default;

    // Transcribe each VAD span [start_s, end_s) of `audio`, returning one AsrChunk
    // per span, in order. `language` "" = auto-detect; a non-empty code pins it.
    // `task` is "transcribe" (default) or "translate".
    virtual std::vector<AsrChunk> transcribe(
        const whisperx::audio::AudioBuffer& audio,
        const std::vector<std::pair<double, double>>& spans,
        const std::string& language = "", const std::string& task = "") = 0;

    // Whisper language identification over the first 30 s — returns the bare code
    // (e.g. "en"), mirroring FasterWhisperPipeline.detect_language.
    virtual std::string detect_language(
        const whisperx::audio::AudioBuffer& audio) = 0;
};

}  // namespace whisperx::asr
