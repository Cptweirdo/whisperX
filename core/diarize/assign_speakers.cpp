#include "diarize/assign_speakers.hpp"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace whisperx::diarize {

namespace {

// Sum overlap durations per speaker preserving first-seen order, then return the
// dominant speaker. Replicates Python's `speaker_intersections` dict (insertion
// order) + `max(..., key=x[1])` (first item wins on a tie). `overlaps` empty ->
// nullopt (the caller only calls this when overlaps is non-empty).
std::optional<std::string> dominant_speaker(
    const std::vector<std::pair<std::string, double>>& overlaps) {
    std::vector<std::pair<std::string, double>> sums;  // insertion-ordered
    for (const auto& [speaker, intersection] : overlaps) {
        bool found = false;
        for (auto& s : sums) {
            if (s.first == speaker) {
                s.second += intersection;
                found = true;
                break;
            }
        }
        if (!found) sums.emplace_back(speaker, intersection);
    }
    if (sums.empty()) return std::nullopt;
    const std::string* best = &sums[0].first;
    double best_val = sums[0].second;
    for (std::size_t i = 1; i < sums.size(); ++i) {
        if (sums[i].second > best_val) {  // strict -> first max wins
            best_val = sums[i].second;
            best = &sums[i].first;
        }
    }
    return *best;
}

}  // namespace

json assign_word_speakers(const std::vector<Turn>& turns, json segments,
                          bool fill_nearest) {
    if (!segments.is_array()) return segments;
    IntervalTree tree(turns);

    for (auto& seg : segments) {
        const double seg_start = seg.value("start", 0.0);
        const double seg_end = seg.value("end", 0.0);

        auto overlaps = tree.query(seg_start, seg_end);
        if (!overlaps.empty()) {
            if (auto sp = dominant_speaker(overlaps)) seg["speaker"] = *sp;
        } else if (fill_nearest) {
            if (auto sp = tree.find_nearest((seg_start + seg_end) / 2.0))
                seg["speaker"] = *sp;
        }

        if (!seg.contains("words") || !seg["words"].is_array()) continue;
        for (auto& word : seg["words"]) {
            if (!word.contains("start")) continue;  // untimed -> skip
            const double word_start = word["start"].get<double>();
            const double word_end = word.value("end", word_start);

            auto word_overlaps = tree.query(word_start, word_end);
            if (!word_overlaps.empty()) {
                if (auto sp = dominant_speaker(word_overlaps)) word["speaker"] = *sp;
            } else if (fill_nearest) {
                if (auto sp = tree.find_nearest((word_start + word_end) / 2.0))
                    word["speaker"] = *sp;
            }
        }
    }
    return segments;
}

}  // namespace whisperx::diarize
