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

class WhisperSherpa {
 public:
    // encoder/decoder/tokens are the three sherpa Whisper ONNX assets;
    // feature_dim is the mel bin count (80 for tiny…medium/large-v2, 128 for
    // large-v3). language/task are defaults ("" language = auto-detect per chunk).
    // provider is the ONNX Runtime execution provider ("cpu" | "cuda"); sherpa
    // selects the CUDA EP from this string once the GPU ORT build is present.
    // batch_size > 1 decodes that many VAD chunks per sherpa call — one encoder
    // pass + lockstep greedy decode (our sherpa batched-whisper patch, see
    // third_party/sherpa-onnx-patches/). NB: measured on CUDA this is currently
    // *not* a win (3080 Ti, turbo fp32: batch 8 RTF 0.526 vs serial 0.443);
    // cause not yet isolated — see CUDA_DECODE_HANDOFF.md before turning it
    // on. VRAM also scales with it (~0.5 GB/row on turbo fp32).
    WhisperSherpa(const std::string& encoder, const std::string& decoder,
                  const std::string& tokens, int num_threads = 1,
                  int feature_dim = 80, const std::string& language = "",
                  const std::string& task = "transcribe",
                  const std::string& provider = "cpu", int batch_size = 1);
    ~WhisperSherpa();
    WhisperSherpa(WhisperSherpa&&) noexcept;
    WhisperSherpa& operator=(WhisperSherpa&&) noexcept;
    WhisperSherpa(const WhisperSherpa&) = delete;
    WhisperSherpa& operator=(const WhisperSherpa&) = delete;

    // Transcribe each VAD span [start_s, end_s) of `audio` — one sherpa offline
    // stream per ≤29.5 s decode window, decoded in groups of batch_size
    // (Whisper pads every chunk to a fixed 30 s mel, so there is no
    // cross-segment batch-norm hazard). `language`/`task` are pinned on the
    // recognizer for the call (sherpa's whisper impl reads them from the
    // recognizer config — per-stream options don't reach it); a non-empty
    // language also enables the batched decode path. "" language = per-chunk
    // auto-detect (serial). Returns one AsrChunk per span, in order.
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
