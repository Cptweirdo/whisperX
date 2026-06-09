#include "asr/whisper_sherpa.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "sherpa-onnx/c-api/c-api.h"

namespace whisperx::asr {

namespace {

// sherpa-onnx's offline Whisper silently truncates any wave reaching
// max_num_frames-50 = 2950 feature frames (~29.5 s) and warns "Only waves less
// than 30 seconds are supported" (offline-recognizer-whisper-impl.h DecodeStream).
// Cap each decode below that; longer spans are split into consecutive sub-windows
// and re-joined so the tail past ~29.5 s is transcribed, not dropped. 2900 frames
// keeps a 0.5 s margin for feature-extraction edge effects.
constexpr std::size_t kMaxDecodeSamples = static_cast<std::size_t>(
    (whisperx::audio::kNFrames - 100) * whisperx::audio::kHopLength);

// Strip Whisper's "<|en|>" language label down to the bare code "en". sherpa
// returns the raw token in result->lang; faster-whisper's detect_language does
// the same trim (asr.py:309-310, language_token[2:-2]).
std::string strip_lang(const char* raw) {
    if (!raw) return "";
    std::string s(raw);
    if (s.size() >= 4 && s.front() == '<' && s.back() == '>') {
        // "<|en|>" -> "en"
        return s.substr(2, s.size() - 4);
    }
    return s;
}

// Mean of the per-token log-probs sherpa optionally returns (best-effort; 0 when
// the model/build doesn't expose them).
float mean_logprob(const SherpaOnnxOfflineRecognizerResult* r) {
    if (!r || !r->ys_log_probs || r->count <= 0) return 0.0f;
    double sum = 0.0;
    for (int32_t i = 0; i < r->count; ++i) sum += r->ys_log_probs[i];
    return static_cast<float>(sum / r->count);
}

}  // namespace

struct WhisperSherpa::Impl {
    // Owned C strings (the config holds raw const char* into these).
    std::string encoder, decoder, tokens, language, task, provider;
    const SherpaOnnxOfflineRecognizer* recognizer = nullptr;

    Impl(const std::string& enc, const std::string& dec, const std::string& tok,
         int num_threads, int feature_dim, const std::string& lang,
         const std::string& tsk, const std::string& prov)
        : encoder(enc), decoder(dec), tokens(tok), language(lang), task(tsk),
          provider(prov) {
        SherpaOnnxOfflineRecognizerConfig config;
        std::memset(&config, 0, sizeof(config));
        config.feat_config.sample_rate = whisperx::audio::kSampleRate;
        config.feat_config.feature_dim = feature_dim;
        config.model_config.whisper.encoder = encoder.c_str();
        config.model_config.whisper.decoder = decoder.c_str();
        config.model_config.whisper.language = language.c_str();
        config.model_config.whisper.task = task.c_str();
        config.model_config.whisper.tail_paddings = -1;  // sherpa default
        config.model_config.tokens = tokens.c_str();
        config.model_config.num_threads = num_threads;
        config.model_config.provider =
            provider.c_str();  // non-null: sherpa wraps in string ("cpu" | "cuda")
        config.model_config.debug = 0;
        config.decoding_method = "greedy_search";

        recognizer = SherpaOnnxCreateOfflineRecognizer(&config);
        if (!recognizer)
            throw std::runtime_error(
                "failed to create sherpa Whisper recognizer (encoder: " + encoder +
                ")");
    }
    ~Impl() {
        if (recognizer) SherpaOnnxDestroyOfflineRecognizer(recognizer);
    }
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    // Decode one waveform; returns {text, avg_logprob}. `lang` (non-empty) sets
    // the per-stream language option when the model supports it.
    AsrChunk decode(const float* samples, int32_t n, const std::string& lang,
                    const std::string& tsk) {
        const SherpaOnnxOfflineStream* stream =
            SherpaOnnxCreateOfflineStream(recognizer);
        if (!stream) throw std::runtime_error("failed to create offline stream");
        if (!lang.empty() &&
            SherpaOnnxOfflineStreamHasOption(stream, "language"))
            SherpaOnnxOfflineStreamSetOption(stream, "language", lang.c_str());
        if (!tsk.empty() && SherpaOnnxOfflineStreamHasOption(stream, "task"))
            SherpaOnnxOfflineStreamSetOption(stream, "task", tsk.c_str());

        SherpaOnnxAcceptWaveformOffline(stream, whisperx::audio::kSampleRate,
                                        samples, n);
        SherpaOnnxDecodeOfflineStream(recognizer, stream);
        const SherpaOnnxOfflineRecognizerResult* r =
            SherpaOnnxGetOfflineStreamResult(stream);

        AsrChunk out;
        if (r) {
            out.text = r->text ? r->text : "";
            out.avg_logprob = mean_logprob(r);
            // Strip the "[BLANK_AUDIO]" text Whisper emits for silent input
            // before it reaches downstream alignment (where the marker would be
            // force-aligned into a bogus timestamped word).
            out.blank_audio_removed = strip_blank_audio(out.text);
        }
        SherpaOnnxDestroyOfflineRecognizerResult(r);
        SherpaOnnxDestroyOfflineStream(stream);
        return out;
    }

    // decode() but split waves longer than the sherpa frame limit into consecutive
    // sub-windows, re-joined into one chunk (text concatenated, avg_logprob
    // sample-weighted) so nothing past ~29.5 s is discarded. One chunk per span is
    // load-bearing: orchestrate/align map spans<->chunks by index.
    AsrChunk decode_capped(const float* samples, std::size_t n,
                           const std::string& lang, const std::string& tsk) {
        if (n <= kMaxDecodeSamples)
            return decode(samples, static_cast<int32_t>(n), lang, tsk);
        AsrChunk merged;
        double lp = 0.0;
        std::size_t covered = 0;
        for (std::size_t off = 0; off < n; off += kMaxDecodeSamples) {
            const std::size_t len = std::min(kMaxDecodeSamples, n - off);
            AsrChunk c =
                decode(samples + off, static_cast<int32_t>(len), lang, tsk);
            // decode() already stripped any markers from c.text; just accumulate
            // the per-sub-window count (a window that was only a marker is now
            // empty and skipped, so no stray separator is appended).
            merged.blank_audio_removed += c.blank_audio_removed;
            if (!c.text.empty()) {
                if (!merged.text.empty()) merged.text += ' ';
                merged.text += c.text;
            }
            lp += static_cast<double>(c.avg_logprob) * static_cast<double>(len);
            covered += len;
        }
        merged.avg_logprob =
            covered ? static_cast<float>(lp / static_cast<double>(covered)) : 0.0f;
        return merged;
    }

    std::string detect(const float* samples, int32_t n) {
        const SherpaOnnxOfflineStream* stream =
            SherpaOnnxCreateOfflineStream(recognizer);
        if (!stream) throw std::runtime_error("failed to create offline stream");
        SherpaOnnxAcceptWaveformOffline(stream, whisperx::audio::kSampleRate,
                                        samples, n);
        SherpaOnnxDecodeOfflineStream(recognizer, stream);
        const SherpaOnnxOfflineRecognizerResult* r =
            SherpaOnnxGetOfflineStreamResult(stream);
        std::string lang = r ? strip_lang(r->lang) : "";
        SherpaOnnxDestroyOfflineRecognizerResult(r);
        SherpaOnnxDestroyOfflineStream(stream);
        return lang;
    }
};

WhisperSherpa::WhisperSherpa(const std::string& encoder,
                             const std::string& decoder,
                             const std::string& tokens, int num_threads,
                             int feature_dim, const std::string& language,
                             const std::string& task,
                             const std::string& provider)
    : impl_(std::make_unique<Impl>(encoder, decoder, tokens, num_threads,
                                   feature_dim, language, task, provider)) {}
WhisperSherpa::~WhisperSherpa() = default;
WhisperSherpa::WhisperSherpa(WhisperSherpa&&) noexcept = default;
WhisperSherpa& WhisperSherpa::operator=(WhisperSherpa&&) noexcept = default;

std::vector<AsrChunk> WhisperSherpa::transcribe(
    const whisperx::audio::AudioBuffer& audio,
    const std::vector<std::pair<double, double>>& spans,
    const std::string& language, const std::string& task) {
    std::vector<AsrChunk> out;
    out.reserve(spans.size());
    const double sr = audio.sample_rate;
    for (const auto& [start_s, end_s] : spans) {
        const auto f0 = static_cast<std::size_t>(start_s * sr);
        const auto f1 = static_cast<std::size_t>(end_s * sr);
        std::span<const float> s = audio.slice(f0, f1);
        out.push_back(impl_->decode_capped(s.data(), s.size(), language, task));
    }
    return out;
}

std::string WhisperSherpa::detect_language(
    const whisperx::audio::AudioBuffer& audio) {
    // First 30 s (kNSamples), the window FasterWhisperPipeline.detect_language uses.
    std::span<const float> s = audio.slice(0, whisperx::audio::kNSamples);
    return impl_->detect(s.data(), static_cast<int32_t>(s.size()));
}

}  // namespace whisperx::asr
