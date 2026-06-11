#include "asr/whisper_cpp.hpp"

#ifdef WHISPERX_WHISPERCPP_BUILD

#include <whisper.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "audio/audio_constants.hpp"

namespace whisperx::asr {

namespace {

// Clamp a [start_s, end_s) span to a contiguous sample range of `audio`.
std::pair<std::size_t, std::size_t> span_frames(
    const whisperx::audio::AudioBuffer& audio, double start_s, double end_s) {
    const double sr = static_cast<double>(audio.sample_rate);
    auto f0 = static_cast<std::size_t>(std::max<long long>(0, std::llround(start_s * sr)));
    auto f1 = static_cast<std::size_t>(std::max<long long>(0, std::llround(end_s * sr)));
    f0 = std::min(f0, audio.size());
    f1 = std::min(std::max(f1, f0), audio.size());
    return {f0, f1};
}

}  // namespace

struct WhisperCpp::Impl {
    whisper_context* ctx = nullptr;
    int num_threads = 4;

    ~Impl() {
        if (ctx) whisper_free(ctx);
    }
};

WhisperCpp::WhisperCpp(const std::string& model_path, int num_threads,
                       bool use_gpu, bool flash_attn)
    : impl_(std::make_unique<Impl>()) {
    impl_->num_threads = num_threads > 0 ? num_threads : 1;

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = use_gpu;
    cparams.flash_attn = flash_attn;

    impl_->ctx =
        whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (!impl_->ctx)
        throw std::runtime_error("whisper.cpp: failed to load ggml model '" +
                                 model_path + "'");
}

WhisperCpp::~WhisperCpp() = default;
WhisperCpp::WhisperCpp(WhisperCpp&&) noexcept = default;
WhisperCpp& WhisperCpp::operator=(WhisperCpp&&) noexcept = default;

std::vector<AsrChunk> WhisperCpp::transcribe(
    const whisperx::audio::AudioBuffer& audio,
    const std::vector<std::pair<double, double>>& spans,
    const std::string& language, const std::string& task) {
    std::vector<AsrChunk> out;
    out.reserve(spans.size());

    whisper_full_params params =
        whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = impl_->num_threads;
    params.translate = (task == "translate");
    params.language = language.empty() ? nullptr : language.c_str();
    params.detect_language = false;
    params.no_timestamps = true;  // Stage 2 re-times; segment ts unused.
    params.print_progress = false;
    params.print_realtime = false;
    params.print_special = false;
    params.print_timestamps = false;
    params.suppress_blank = true;
    params.single_segment = false;

    for (const auto& [start_s, end_s] : spans) {
        AsrChunk chunk;
        auto [f0, f1] = span_frames(audio, start_s, end_s);
        auto pcm = audio.slice(f0, f1);

        if (!pcm.empty() &&
            whisper_full(impl_->ctx, params, pcm.data(),
                         static_cast<int>(pcm.size())) == 0) {
            std::string text;
            double logprob_sum = 0.0;
            int tok_count = 0;
            const int n_seg = whisper_full_n_segments(impl_->ctx);
            for (int s = 0; s < n_seg; ++s) {
                const char* seg = whisper_full_get_segment_text(impl_->ctx, s);
                if (seg) text += seg;
                const int n_tok = whisper_full_n_tokens(impl_->ctx, s);
                for (int t = 0; t < n_tok; ++t) {
                    const float p =
                        whisper_full_get_token_p(impl_->ctx, s, t);
                    if (p > 0.0f) {
                        logprob_sum += std::log(static_cast<double>(p));
                        ++tok_count;
                    }
                }
            }
            chunk.blank_audio_removed = strip_blank_audio(text);
            chunk.text = std::move(text);
            chunk.avg_logprob =
                tok_count > 0
                    ? static_cast<float>(logprob_sum / tok_count)
                    : 0.0f;
        }
        out.push_back(std::move(chunk));
    }
    return out;
}

std::string WhisperCpp::detect_language(
    const whisperx::audio::AudioBuffer& audio) {
    const std::size_t n =
        std::min(audio.size(),
                 static_cast<std::size_t>(whisperx::audio::kSampleRate) * 30);
    auto pcm = audio.slice(0, n);
    if (pcm.empty()) return "";
    if (whisper_pcm_to_mel(impl_->ctx, pcm.data(),
                           static_cast<int>(pcm.size()),
                           impl_->num_threads) != 0)
        return "";
    const int id = whisper_lang_auto_detect(impl_->ctx, /*offset_ms=*/0,
                                            impl_->num_threads, nullptr);
    if (id < 0) return "";
    const char* code = whisper_lang_str(id);
    return code ? std::string(code) : "";
}

}  // namespace whisperx::asr

#endif  // WHISPERX_WHISPERCPP_BUILD
