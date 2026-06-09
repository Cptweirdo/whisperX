#include "align/wav2vec2_onnx.hpp"

#include <algorithm>
#include <array>
#include <numeric>

#include "onnxruntime_cxx_api.h"

namespace whisperx::align {

bool ort_cuda_available() {
#ifdef WHISPERX_GPU_BUILD
    try {
        for (const auto& p : Ort::GetAvailableProviders())
            if (p == "CUDAExecutionProvider") return true;
    } catch (...) {
        // ORT provider enumeration failed — treat as unavailable.
    }
    return false;
#else
    return false;
#endif
}

namespace {
constexpr std::size_t kMinSamples = 400;  // wav2vec2 conv feature-extractor minimum
constexpr std::size_t kMaxBatch = 8;      // segment-count cap per padded batch
// Padded-sample budget per batch (batched mode). Self-attention memory scales with
// B*maxn**2; capping the padded footprint B*maxn keeps each forward's peak bounded
// regardless of segment length. Native ASR emits one segment per 30 s VAD chunk, so
// without this a long clip packs several 30 s rows into one batch and OOMs (the
// attention tensor B*heads*T*T blows past RAM). 480000 = 30 s @ 16 kHz: one full
// chunk fills a batch alone; shorter segments still pack up to kMaxBatch.
constexpr std::size_t kMaxBatchSamples = 30 * 16000;

Ort::SessionOptions make_options(int num_threads, const std::string& provider) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(num_threads);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef WHISPERX_GPU_BUILD
    // Stage 2 is the only stage that drives ORT directly, so it appends the CUDA
    // EP itself (sherpa does this internally for stages 1 & 3). ORT auto-copies the
    // CPU-built input tensors host->device; IoBinding is a later optimization.
    if (provider == "cuda") {
        OrtCUDAProviderOptions cuda_opts{};  // device_id 0, library defaults
        opts.AppendExecutionProvider_CUDA(cuda_opts);
    }
#else
    (void)provider;  // CPU build: the CUDA-EP symbol may be absent in these headers
#endif
    return opts;
}
}  // namespace

struct Wav2Vec2Onnx::Impl {
    Ort::Env env;
    Ort::SessionOptions options;
    Ort::Session session;
    // The mirror ships two contracts: a 2-in/2-out graph (waveform + attention_mask
    // -> emissions + frame_lengths, layer_norm/batchable) and a 1-in/1-out graph
    // (waveform -> emissions, group_norm). Introspect the session so forward() binds
    // only the io the graph actually has — feeding a name the graph lacks aborts ORT.
    bool has_attention_mask = false;
    bool has_frame_lengths = false;

    Impl(const std::string& path, int num_threads, const std::string& provider)
        : env(ORT_LOGGING_LEVEL_WARNING, "wav2vec2_align"),
          options(make_options(num_threads, provider)),
          session(env, path.c_str(), options) {
        Ort::AllocatorWithDefaultOptions alloc;
        for (std::size_t i = 0; i < session.GetInputCount(); ++i)
            if (session.GetInputNameAllocated(i, alloc).get() ==
                std::string("attention_mask"))
                has_attention_mask = true;
        for (std::size_t i = 0; i < session.GetOutputCount(); ++i)
            if (session.GetOutputNameAllocated(i, alloc).get() ==
                std::string("frame_lengths"))
                has_frame_lengths = true;
    }
};

Wav2Vec2Onnx::Wav2Vec2Onnx(const std::string& onnx_path, int num_threads,
                           const std::string& provider)
    : impl_(std::make_unique<Impl>(onnx_path, num_threads, provider)) {}
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

    // A graph with no attention_mask cannot mask padding, so a padded batch corrupts
    // every row but the longest — only the per-segment (b=1, no pad) path is correct.
    // batchable models always ship the mask; this just hard-guards the invariant.
    const bool can_batch = batched && impl_->has_attention_mask;

    // Group original indices into buckets. Per-segment = one index per bucket;
    // batched = sort by length (minimize pad waste) then chunk by kMaxBatch.
    std::vector<std::vector<std::size_t>> buckets;
    if (!can_batch) {
        buckets.reserve(n);
        for (std::size_t i = 0; i < n; ++i) buckets.push_back({i});
    } else {
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return waveforms[a].size() < waveforms[b].size();
        });
        // Greedy pack: ascending order means the candidate is the bucket's longest,
        // so the padded footprint is (count+1) * candidate. Start a new bucket when
        // adding would exceed either the count cap or the padded-sample budget; a
        // single over-budget segment still runs alone (a bucket holds >=1).
        std::vector<std::size_t> cur;
        for (std::size_t idx : order) {
            const std::size_t cand = std::max(waveforms[idx].size(), kMinSamples);
            const bool over_count = cur.size() >= kMaxBatch;
            const bool over_budget =
                !cur.empty() && (cur.size() + 1) * cand > kMaxBatchSamples;
            if (!cur.empty() && (over_count || over_budget)) {
                buckets.push_back(std::move(cur));
                cur.clear();
            }
            cur.push_back(idx);
        }
        if (!cur.empty()) buckets.push_back(std::move(cur));
    }

    const Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Bind only the io the graph exposes (see Impl introspection). The 1-in/1-out
    // group_norm graph never pads (can_batch is false ⇒ b==1), so the omitted mask
    // would be all-ones and the omitted frame_lengths equals the full emission T.
    const bool want_mask = impl_->has_attention_mask;
    const bool want_flens = impl_->has_frame_lengths;
    std::vector<const char*> in_names{"waveform"};
    if (want_mask) in_names.push_back("attention_mask");
    std::vector<const char*> out_names{"emissions"};
    if (want_flens) out_names.push_back("frame_lengths");

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
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem, wav.data(), wav.size(), dims.data(), dims.size()));
        if (want_mask)
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                mem, mask.data(), mask.size(), dims.data(), dims.size()));

        auto outs = impl_->session.Run(
            Ort::RunOptions{nullptr}, in_names.data(), inputs.data(), inputs.size(),
            out_names.data(), out_names.size());

        const auto oshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
        const std::size_t t_max = static_cast<std::size_t>(oshape[1]);
        const std::size_t v = static_cast<std::size_t>(oshape[2]);
        const float* emissions = outs[0].GetTensorData<float>();
        // No frame_lengths output ⇒ unpadded (b==1) ⇒ every row spans the full T.
        const int64_t* frame_lengths =
            want_flens ? outs[1].GetTensorData<int64_t>() : nullptr;

        for (std::size_t j = 0; j < b; ++j) {
            std::size_t ti =
                frame_lengths
                    ? static_cast<std::size_t>(std::max<int64_t>(0, frame_lengths[j]))
                    : t_max;
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
