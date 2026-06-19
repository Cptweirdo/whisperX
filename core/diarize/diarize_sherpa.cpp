#include "diarize/diarize_sherpa.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "diarize/merge_clusters.hpp"
#include "sherpa-onnx/c-api/c-api.h"

namespace whisperx::diarize {

struct SherpaDiarizer::Impl {
    // Owned C strings (the configs hold raw const char* into these).
    std::string segmentation, embedding, provider;
    int num_threads;
    float threshold, min_duration_on, min_duration_off;
    float merge_threshold = 0.0f;  // set by the SherpaDiarizer ctor
    const SherpaOnnxOfflineSpeakerDiarization* sd = nullptr;
    // Built lazily on the first embeddings() call (the diarization handle has its
    // own internal extractor we can't reach, so we run a second one ourselves).
    const SherpaOnnxSpeakerEmbeddingExtractor* extractor = nullptr;

    Impl(const std::string& seg, const std::string& emb, int threads,
         const std::string& prov, float thr, float dur_on, float dur_off)
        : segmentation(seg), embedding(emb), provider(prov), num_threads(threads),
          threshold(thr), min_duration_on(dur_on), min_duration_off(dur_off) {
        SherpaOnnxOfflineSpeakerDiarizationConfig config;
        std::memset(&config, 0, sizeof(config));
        config.segmentation.pyannote.model = segmentation.c_str();
        config.segmentation.num_threads = num_threads;
        config.segmentation.provider = provider.c_str();
        config.segmentation.debug = 0;
        config.embedding.model = embedding.c_str();
        config.embedding.num_threads = num_threads;
        config.embedding.provider = provider.c_str();
        config.embedding.debug = 0;
        // Default clustering = auto (count from the cosine threshold); diarize()
        // overrides per call via SetConfig when a speaker count is requested.
        config.clustering.num_clusters = -1;
        config.clustering.threshold = threshold;
        config.min_duration_on = min_duration_on;
        config.min_duration_off = min_duration_off;

        sd = SherpaOnnxCreateOfflineSpeakerDiarization(&config);
        if (!sd)
            throw std::runtime_error(
                "failed to create sherpa speaker diarization (segmentation: " +
                segmentation + ")");
    }
    ~Impl() {
        if (extractor) SherpaOnnxDestroySpeakerEmbeddingExtractor(extractor);
        if (sd) SherpaOnnxDestroyOfflineSpeakerDiarization(sd);
    }
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    void set_clustering(int num_clusters) {
        SherpaOnnxOfflineSpeakerDiarizationConfig config;
        std::memset(&config, 0, sizeof(config));
        // SetConfig only reads config->clustering — other fields are ignored.
        config.clustering.num_clusters = num_clusters > 0 ? num_clusters : -1;
        config.clustering.threshold = threshold;
        SherpaOnnxOfflineSpeakerDiarizationSetConfig(sd, &config);
    }

    void ensure_extractor() {
        if (extractor) return;
        SherpaOnnxSpeakerEmbeddingExtractorConfig config;
        std::memset(&config, 0, sizeof(config));
        config.model = embedding.c_str();
        config.num_threads = num_threads;
        config.provider = provider.c_str();
        config.debug = 0;
        extractor = SherpaOnnxCreateSpeakerEmbeddingExtractor(&config);
        if (!extractor)
            throw std::runtime_error(
                "failed to create sherpa speaker embedding extractor (model: " +
                embedding + ")");
    }
};

SherpaDiarizer::SherpaDiarizer(const std::string& segmentation,
                               const std::string& embedding, int num_threads,
                               const std::string& provider, float threshold,
                               float min_duration_on, float min_duration_off,
                               float merge_threshold)
    : impl_(std::make_unique<Impl>(segmentation, embedding, num_threads, provider,
                                   threshold, min_duration_on,
                                   min_duration_off)) {
    impl_->merge_threshold = merge_threshold;
}
SherpaDiarizer::~SherpaDiarizer() = default;
SherpaDiarizer::SherpaDiarizer(SherpaDiarizer&&) noexcept = default;
SherpaDiarizer& SherpaDiarizer::operator=(SherpaDiarizer&&) noexcept = default;

std::vector<DiarSegment> SherpaDiarizer::diarize(
    const whisperx::audio::AudioBuffer& audio, int num_clusters) {
    impl_->set_clustering(num_clusters);

    std::span<const float> s = audio.slice(0, audio.size());
    const SherpaOnnxOfflineSpeakerDiarizationResult* r =
        SherpaOnnxOfflineSpeakerDiarizationProcess(
            impl_->sd, s.data(), static_cast<int32_t>(s.size()));

    std::vector<DiarSegment> out;
    if (r) {
        const int32_t n =
            SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(r);
        const SherpaOnnxOfflineSpeakerDiarizationSegment* segs =
            SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(r);
        out.reserve(n);
        for (int32_t i = 0; i < n; ++i)
            out.push_back({static_cast<double>(segs[i].start),
                           static_cast<double>(segs[i].end), segs[i].speaker});
        SherpaOnnxOfflineSpeakerDiarizationDestroySegment(segs);
        SherpaOnnxOfflineSpeakerDiarizationDestroyResult(r);
    }

    // Centroid merge post-pass — only in auto-count mode (a user-forced count is
    // exact already) and only when FastClustering produced multiple clusters.
    if (num_clusters <= 0 && impl_->merge_threshold > 0.0f && !out.empty()) {
        std::map<int, double> talk;
        for (const auto& seg : out) talk[seg.speaker] += seg.end - seg.start;
        if (talk.size() > 1) {
            auto mapping = merge_speaker_clusters(embeddings(audio, out), talk,
                                                  impl_->merge_threshold);
            // Relabel, then renumber compactly by first appearance (turns are
            // sorted by start time, so ids read naturally in the transcript).
            std::map<int, int> compact;
            for (auto& seg : out) {
                int merged = mapping.at(seg.speaker);
                auto [it, inserted] =
                    compact.emplace(merged, static_cast<int>(compact.size()));
                seg.speaker = it->second;
            }
        }
    }
    return out;
}

std::map<int, std::vector<float>> SherpaDiarizer::embeddings(
    const whisperx::audio::AudioBuffer& audio,
    const std::vector<DiarSegment>& segments) {
    impl_->ensure_extractor();
    const int dim = SherpaOnnxSpeakerEmbeddingExtractorDim(impl_->extractor);
    const double sr = audio.sample_rate;

    // One embedding per speaker: feed all of that speaker's turns into a single
    // stream (a per-speaker vector, like the pyannote `speaker_embeddings` map).
    std::map<int, std::vector<float>> out;
    // Preserve first-seen speaker ids; iterate the set of distinct speakers.
    std::vector<int> speakers;
    for (const auto& seg : segments)
        if (std::find(speakers.begin(), speakers.end(), seg.speaker) ==
            speakers.end())
            speakers.push_back(seg.speaker);

    for (int spk : speakers) {
        const SherpaOnnxOnlineStream* stream =
            SherpaOnnxSpeakerEmbeddingExtractorCreateStream(impl_->extractor);
        bool any = false;
        for (const auto& seg : segments) {
            if (seg.speaker != spk) continue;
            const auto f0 = static_cast<std::size_t>(seg.start * sr);
            const auto f1 = static_cast<std::size_t>(seg.end * sr);
            std::span<const float> s = audio.slice(f0, f1);
            if (s.empty()) continue;
            SherpaOnnxOnlineStreamAcceptWaveform(
                stream, whisperx::audio::kSampleRate, s.data(),
                static_cast<int32_t>(s.size()));
            any = true;
        }
        SherpaOnnxOnlineStreamInputFinished(stream);

        std::vector<float> vec(static_cast<std::size_t>(dim), 0.0f);
        if (any &&
            SherpaOnnxSpeakerEmbeddingExtractorIsReady(impl_->extractor, stream)) {
            const float* v = SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(
                impl_->extractor, stream);
            if (v) {
                vec.assign(v, v + dim);
                SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(v);
            }
        }
        SherpaOnnxDestroyOnlineStream(stream);
        out[spk] = std::move(vec);
    }
    return out;
}

int SherpaDiarizer::embedding_dim() {
    impl_->ensure_extractor();
    return SherpaOnnxSpeakerEmbeddingExtractorDim(impl_->extractor);
}

}  // namespace whisperx::diarize
