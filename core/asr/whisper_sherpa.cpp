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
    int batch_size = 1;
    const SherpaOnnxOfflineRecognizer* recognizer = nullptr;
    // Last language/task applied via SherpaOnnxOfflineRecognizerSetConfig —
    // sherpa's Whisper impl reads language/task from the *recognizer* config
    // (per-stream options are ignored by it), so transcribe()/detect() retarget
    // the recognizer and this cache skips redundant SetConfig calls.
    std::string applied_language, applied_task;

    Impl(const std::string& enc, const std::string& dec, const std::string& tok,
         int num_threads, int feature_dim, const std::string& lang,
         const std::string& tsk, const std::string& prov, int batch)
        : encoder(enc), decoder(dec), tokens(tok), language(lang), task(tsk),
          provider(prov), batch_size(batch < 1 ? 1 : batch),
          applied_language(lang), applied_task(tsk) {
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

    // Retarget the recognizer's whisper config to `lang`/`tsk` (sherpa reads
    // both from the recognizer config at decode time; "" language = per-chunk
    // auto-detect). No-op when already applied — SetConfig is cheap (a struct
    // copy, no model reload) but not free.
    void apply_options(const std::string& lang, const std::string& tsk) {
        // sherpa normalizes an empty task to "transcribe" (c-api
        // GetOfflineRecognizerConfig), so cache the normalized value.
        const std::string t = tsk.empty() ? "transcribe" : tsk;
        if (lang == applied_language && t == applied_task) return;
        SherpaOnnxOfflineRecognizerConfig config;
        std::memset(&config, 0, sizeof(config));
        // Only model_config.whisper is consumed by the whisper impl's
        // SetConfig; keep the asset paths truthful anyway.
        config.model_config.whisper.encoder = encoder.c_str();
        config.model_config.whisper.decoder = decoder.c_str();
        config.model_config.whisper.language = lang.c_str();
        config.model_config.whisper.task = t.c_str();
        config.model_config.whisper.tail_paddings = -1;  // sherpa default
        SherpaOnnxOfflineRecognizerSetConfig(recognizer, &config);
        applied_language = lang;
        applied_task = t;
    }

    static AsrChunk chunk_from(const SherpaOnnxOfflineRecognizerResult* r) {
        AsrChunk out;
        if (r) {
            out.text = r->text ? r->text : "";
            out.avg_logprob = mean_logprob(r);
            // Strip the "[BLANK_AUDIO]" text Whisper emits for silent input
            // before it reaches downstream alignment (where the marker would be
            // force-aligned into a bogus timestamped word).
            out.blank_audio_removed = strip_blank_audio(out.text);
        }
        return out;
    }

    // Decode one waveform with the recognizer's current language/task.
    AsrChunk decode(const float* samples, int32_t n) {
        const SherpaOnnxOfflineStream* stream =
            SherpaOnnxCreateOfflineStream(recognizer);
        if (!stream) throw std::runtime_error("failed to create offline stream");
        SherpaOnnxAcceptWaveformOffline(stream, whisperx::audio::kSampleRate,
                                        samples, n);
        SherpaOnnxDecodeOfflineStream(recognizer, stream);
        const SherpaOnnxOfflineRecognizerResult* r =
            SherpaOnnxGetOfflineStreamResult(stream);
        AsrChunk out = chunk_from(r);
        SherpaOnnxDestroyOfflineRecognizerResult(r);
        SherpaOnnxDestroyOfflineStream(stream);
        return out;
    }

    // One ≤29.5 s decode window of a span (spans longer than sherpa's frame
    // limit are split into consecutive sub-windows and re-joined per span).
    struct Window {
        std::size_t span;  // index into the spans/output vector
        const float* ptr;
        std::size_t len;
    };

    // Decode `k` windows as one sherpa batch (one encoder pass + lockstep
    // greedy decode via our sherpa patch). Stream i ↔ window i, so results
    // keep window order regardless of batch grouping.
    void decode_group(const Window* ws, std::size_t k, AsrChunk* out) {
        std::vector<const SherpaOnnxOfflineStream*> streams(k);
        for (std::size_t i = 0; i < k; ++i) {
            streams[i] = SherpaOnnxCreateOfflineStream(recognizer);
            if (!streams[i]) {
                for (std::size_t j = 0; j < i; ++j)
                    SherpaOnnxDestroyOfflineStream(streams[j]);
                throw std::runtime_error("failed to create offline stream");
            }
            SherpaOnnxAcceptWaveformOffline(streams[i],
                                            whisperx::audio::kSampleRate,
                                            ws[i].ptr,
                                            static_cast<int32_t>(ws[i].len));
        }
        SherpaOnnxDecodeMultipleOfflineStreams(
            recognizer, streams.data(), static_cast<int32_t>(k));
        for (std::size_t i = 0; i < k; ++i) {
            const SherpaOnnxOfflineRecognizerResult* r =
                SherpaOnnxGetOfflineStreamResult(streams[i]);
            out[i] = chunk_from(r);
            SherpaOnnxDestroyOfflineRecognizerResult(r);
            SherpaOnnxDestroyOfflineStream(streams[i]);
        }
    }

    std::string detect(const float* samples, int32_t n) {
        // Detection needs the auto-detect config — reset the language a
        // previous transcribe() may have pinned, or every job after the first
        // would "detect" the prior job's language.
        apply_options("", "transcribe");
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
                             const std::string& provider, int batch_size)
    : impl_(std::make_unique<Impl>(encoder, decoder, tokens, num_threads,
                                   feature_dim, language, task, provider,
                                   batch_size)) {}
WhisperSherpa::~WhisperSherpa() = default;
WhisperSherpa::WhisperSherpa(WhisperSherpa&&) noexcept = default;
WhisperSherpa& WhisperSherpa::operator=(WhisperSherpa&&) noexcept = default;

std::vector<AsrChunk> WhisperSherpa::transcribe(
    const whisperx::audio::AudioBuffer& audio,
    const std::vector<std::pair<double, double>>& spans,
    const std::string& language, const std::string& task) {
    // Pin language/task on the recognizer for the whole job (sherpa's whisper
    // impl reads them from the recognizer config; with a fixed language the
    // patched DecodeStreams takes the batched path).
    impl_->apply_options(language, task);

    // Flatten spans into ≤29.5 s decode windows. sherpa's offline Whisper
    // silently truncates waves reaching max_num_frames-50 = 2950 feature
    // frames ("Only waves less than 30 seconds are supported"), so longer
    // spans become consecutive sub-windows, re-joined below. One chunk per
    // span is load-bearing: orchestrate/align map spans<->chunks by index.
    const double sr = audio.sample_rate;
    std::vector<Impl::Window> windows;
    windows.reserve(spans.size());
    for (std::size_t i = 0; i < spans.size(); ++i) {
        const auto f0 = static_cast<std::size_t>(spans[i].first * sr);
        const auto f1 = static_cast<std::size_t>(spans[i].second * sr);
        std::span<const float> s = audio.slice(f0, f1);
        for (std::size_t off = 0; off < s.size(); off += kMaxDecodeSamples) {
            const std::size_t len = std::min(kMaxDecodeSamples, s.size() - off);
            windows.push_back({i, s.data() + off, len});
        }
    }

    // Decode windows in groups of batch_size (group of 1 == today's serial
    // path; sherpa preserves stream order within a group).
    std::vector<AsrChunk> per_window(windows.size());
    const std::size_t bs = static_cast<std::size_t>(impl_->batch_size);
    for (std::size_t g = 0; g < windows.size(); g += bs) {
        const std::size_t k = std::min(bs, windows.size() - g);
        if (k == 1) {
            per_window[g] = impl_->decode(
                windows[g].ptr, static_cast<int32_t>(windows[g].len));
        } else {
            impl_->decode_group(&windows[g], k, &per_window[g]);
        }
    }

    // Re-join sub-windows per span: text concatenated, avg_logprob
    // sample-weighted, marker counts accumulated (a window that was only a
    // "[BLANK_AUDIO]" marker is empty after stripping and skipped, so no
    // stray separator is appended). An empty span yields an empty chunk.
    std::vector<AsrChunk> out(spans.size());
    std::vector<double> lp(spans.size(), 0.0);
    std::vector<std::size_t> covered(spans.size(), 0);
    for (std::size_t w = 0; w < windows.size(); ++w) {
        const std::size_t i = windows[w].span;
        AsrChunk& c = per_window[w];
        out[i].blank_audio_removed += c.blank_audio_removed;
        if (!c.text.empty()) {
            if (!out[i].text.empty()) out[i].text += ' ';
            out[i].text += c.text;
        }
        lp[i] += static_cast<double>(c.avg_logprob) *
                 static_cast<double>(windows[w].len);
        covered[i] += windows[w].len;
    }
    for (std::size_t i = 0; i < spans.size(); ++i) {
        out[i].avg_logprob =
            covered[i] ? static_cast<float>(lp[i] / static_cast<double>(
                                                        covered[i]))
                       : 0.0f;
    }
    return out;
}

std::string WhisperSherpa::detect_language(
    const whisperx::audio::AudioBuffer& audio) {
    // FasterWhisperPipeline.detect_language uses the first 30 s (kNSamples), but
    // a full kNSamples wave plus sherpa's tail padding overflows its 30 s circular
    // buffer (circular-buffer.cc Push overflow) and trips the "waves less than 30
    // seconds" truncation in DecodeStream — so cap to the same sub-30 s window as
    // decode_capped.
    std::span<const float> s = audio.slice(0, kMaxDecodeSamples);
    return impl_->detect(s.data(), static_cast<int32_t>(s.size()));
}

}  // namespace whisperx::asr
