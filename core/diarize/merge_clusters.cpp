#include "diarize/merge_clusters.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <utility>

namespace whisperx::diarize {

namespace {

double cosine_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na == 0.0 || nb == 0.0) return 2.0;  // zero-norm: maximally far, never merges
    return 1.0 - dot / (std::sqrt(na) * std::sqrt(nb));
}

}  // namespace

std::map<int, int> merge_speaker_clusters(
    const std::map<int, std::vector<float>>& centroids,
    const std::map<int, double>& talk_time, float threshold) {
    // Live clusters: id -> (centroid, weight). doubles for the weighted means.
    std::map<int, std::pair<std::vector<double>, double>> live;
    for (const auto& [spk, vec] : centroids) {
        std::vector<double> v(vec.begin(), vec.end());
        auto it = talk_time.find(spk);
        live.emplace(spk, std::make_pair(std::move(v),
                                         it != talk_time.end() ? it->second : 0.0));
    }

    std::map<int, int> mapping;
    for (const auto& [spk, _] : live) mapping[spk] = spk;

    while (live.size() > 1) {
        std::optional<std::pair<int, int>> best;
        double best_d = threshold;
        for (auto a = live.begin(); a != live.end(); ++a) {
            auto b = a;
            for (++b; b != live.end(); ++b) {
                double d = cosine_distance(a->second.first, b->second.first);
                if (d < best_d) {
                    best_d = d;
                    best = {a->first, b->first};
                }
            }
        }
        if (!best) break;
        auto& [keep_vec, keep_w] = live[best->first];   // smaller id survives
        auto& [gone_vec, gone_w] = live[best->second];  // (map iterates ascending)
        double total = keep_w + gone_w;
        if (total > 0.0)
            for (std::size_t i = 0; i < keep_vec.size(); ++i)
                keep_vec[i] = (keep_vec[i] * keep_w + gone_vec[i] * gone_w) / total;
        keep_w = total;
        live.erase(best->second);
        for (auto& [from, to] : mapping)
            if (to == best->second) to = best->first;
    }
    return mapping;
}

}  // namespace whisperx::diarize
