#include "align/wav2vec2_onnx.hpp"

#include <algorithm>
#include <array>
#include <numeric>

#include "onnxruntime_cxx_api.h"

namespace whisperx::align {

namespace {
constexpr std::size_t kMinSamples = 400;  // wav2vec2 conv feature-extractor minimum
constexpr std::size_t kMaxBatch = 8;      // segments per padded batch (batched mode)

Ort::SessionOptions make_options(int num_threads) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(num_threads);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return opts;
}
}  // namespace

struct Wav2Vec2Onnx::Impl {
    Ort::Env env;
    Ort::SessionOptions options;
    Ort::Session session;

    Impl(const std::string& path, int num_threads)
        : env(ORT_LOGGING_LEVEL_WARNING, "wav2vec2_align"),
          options(make_options(num_threads)),
          session(env, path.c_str(), options) {}
};

Wav2Vec2Onnx::Wav2Vec2Onnx(const std::string& onnx_path, int num_threads)
    : impl_(std::make_unique<Impl>(onnx_path, num_threads)) {}
Wav2Vec2Onnx::~Wav2Vec2Onnx() = default;
Wav2Vec2Onnx::Wav2Vec2Onnx(Wav2Vec2Onnx&&) noexcept = default;
Wav2Vec2Onnx& Wav2Vec2Onnx::operator=(Wav2Vec2Onnx&&) noexcept = default;

std::vector<std::vector<float>> Wav2Vec2Onnx::forward(
    const std::vector<std::span<const float>>& waveforms,
    std::vector<std::pair<std::size_t, std::size_t>>& shapes, bool batched) {
    const std::size_t n = waveforms.size();
    std::vector<std::vector<float>> out(n);
    shapes.assign(n, {0, 0});
    if (n == 0) return out;

    // Group original indices into buckets. Per-segment = one index per bucket;
    // batched = sort by length (minimize pad waste) then chunk by kMaxBatch.
    std::vector<std::vector<std::size_t>> buckets;
    if (!batched) {
        buckets.reserve(n);
        for (std::size_t i = 0; i < n; ++i) buckets.push_back({i});
    } else {
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return waveforms[a].size() < waveforms[b].size();
        });
        for (std::size_t i = 0; i < n; i += kMaxBatch)
            buckets.emplace_back(order.begin() + static_cast<std::ptrdiff_t>(i),
                                 order.begin() + static_cast<std::ptrdiff_t>(
                                                     std::min(n, i + kMaxBatch)));
    }

    const Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* in_names[] = {"waveform", "attention_mask"};
    const char* out_names[] = {"emissions", "frame_lengths"};

    for (const auto& bucket : buckets) {
        const std::size_t b = bucket.size();
        std::size_t maxn = kMinSamples;
        for (std::size_t idx : bucket) maxn = std::max(maxn, waveforms[idx].size());

        // Right-pad each row to maxn; attention_mask is 1 over real samples, 0 pad.
        std::vector<float> wav(b * maxn, 0.0f);
        std::vector<int64_t> mask(b * maxn, 0);
        for (std::size_t j = 0; j < b; ++j) {
            const auto& w = waveforms[bucket[j]];
            std::copy(w.begin(), w.end(),
                      wav.begin() + static_cast<std::ptrdiff_t>(j * maxn));
            std::fill_n(mask.begin() + static_cast<std::ptrdiff_t>(j * maxn),
                        w.size(), int64_t{1});
        }

        const std::array<int64_t, 2> dims{static_cast<int64_t>(b),
                                          static_cast<int64_t>(maxn)};
        std::array<Ort::Value, 2> inputs{
            Ort::Value::CreateTensor<float>(mem, wav.data(), wav.size(),
                                            dims.data(), dims.size()),
            Ort::Value::CreateTensor<int64_t>(mem, mask.data(), mask.size(),
                                              dims.data(), dims.size())};

        auto outs = impl_->session.Run(Ort::RunOptions{nullptr}, in_names,
                                       inputs.data(), inputs.size(), out_names, 2);

        const auto oshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
        const std::size_t t_max = static_cast<std::size_t>(oshape[1]);
        const std::size_t v = static_cast<std::size_t>(oshape[2]);
        const float* emissions = outs[0].GetTensorData<float>();
        const int64_t* frame_lengths = outs[1].GetTensorData<int64_t>();

        for (std::size_t j = 0; j < b; ++j) {
            std::size_t ti =
                static_cast<std::size_t>(std::max<int64_t>(0, frame_lengths[j]));
            ti = std::min(ti, t_max);  // trim padded frames; clamp defensively
            const std::size_t idx = bucket[j];
            const float* base = emissions + j * t_max * v;
            out[idx].assign(base, base + ti * v);
            shapes[idx] = {ti, v};
        }
    }
    return out;
}

}  // namespace whisperx::align
