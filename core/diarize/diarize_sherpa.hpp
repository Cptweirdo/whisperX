// Native speaker-diarization backend (Phase 4 / slice 4b). Wraps sherpa-onnx's
// offline speaker diarization (pyannote-segmentation-3.0 + a speaker-embedding
// extractor + FastClustering, all on ONNX Runtime) — the A/B replacement for the
// Python pyannote `community-1` pipeline in whisperx/diarize.py::DiarizationPipeline.
//
// A/B, not parity (settled): community-1 is pyannote.audio 4.0 (its own
// segmentation + embedding + VBxClustering/PLDA), sherpa uses a different
// segmentation model, a different embedding extractor and cosine FastClustering —
// so the turns differ by construction. The downstream assign_word_speakers glue
// (already native, the `assign` token) is model-independent and parity-tested
// separately; this backend is judged by speaker-count + DER vs ground-truth RTTM.
//
// pImpl so this header pulls no ORT/sherpa headers — only the .cpp (and the audio
// stage that builds it) sees the sherpa C API. Built only under WHISPERX_CORE_AUDIO.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "audio/audio_buffer.hpp"

namespace whisperx::diarize {

// One diarization turn: [start, end) seconds with an integer cluster id (the
// Python facade maps it to "SPEAKER_xx").
struct DiarSegment {
    double start;
    double end;
    int speaker;
};

class SherpaDiarizer {
 public:
    // segmentation = pyannote-segmentation-3.0 ONNX; embedding = the speaker
    // embedding extractor ONNX (wespeaker_en_voxceleb CAM++). threshold is the
    // FastClustering cosine distance used when the speaker count is unknown;
    // min_duration_{on,off} drop tiny segments / merge tiny gaps (sherpa defaults).
    // merge_threshold drives the centroid post-pass (merge_clusters.hpp) that
    // re-joins clusters FastClustering fragmented: pooled per-cluster embeddings
    // closer than it are one speaker. It lives on the pooled-embedding distance
    // scale (~0.3) — much tighter than the chunk-level `threshold`. 0 disables.
    SherpaDiarizer(const std::string& segmentation, const std::string& embedding,
                   int num_threads = 1, const std::string& provider = "cpu",
                   float threshold = 0.5f, float min_duration_on = 0.3f,
                   float min_duration_off = 0.5f, float merge_threshold = 0.25f);
    ~SherpaDiarizer();
    SherpaDiarizer(SherpaDiarizer&&) noexcept;
    SherpaDiarizer& operator=(SherpaDiarizer&&) noexcept;
    SherpaDiarizer(const SherpaDiarizer&) = delete;
    SherpaDiarizer& operator=(const SherpaDiarizer&) = delete;

    // Diarize the whole buffer. `num_clusters > 0` forces exactly that many
    // speakers (FastClustering num_clusters); `<= 0` lets the cosine threshold
    // decide the count and then applies the centroid merge post-pass (when
    // merge_threshold > 0), renumbering speakers by first appearance. Returns
    // turns sorted by start time.
    std::vector<DiarSegment> diarize(const whisperx::audio::AudioBuffer& audio,
                                     int num_clusters = 0);

    // Per-speaker embedding vector (a second extractor pass — the diarization
    // result exposes none). For each cluster, all its turns' audio is fed into one
    // embedding stream → a single vector. Keyed by the DiarSegment.speaker id.
    std::map<int, std::vector<float>> embeddings(
        const whisperx::audio::AudioBuffer& audio,
        const std::vector<DiarSegment>& segments);

    // Embedding dimensionality of the extractor model.
    int embedding_dim();

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace whisperx::diarize
