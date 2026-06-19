#include "diarize/interval_tree.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace whisperx::diarize {

IntervalTree::IntervalTree(const std::vector<Turn>& turns) {
    // Stable sort by start (np `sorted(..., key=lambda x: x[0])` is stable).
    std::vector<std::size_t> order(turns.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) {
                         return turns[a].start < turns[b].start;
                     });
    starts_.reserve(turns.size());
    ends_.reserve(turns.size());
    speakers_.reserve(turns.size());
    for (std::size_t i : order) {
        starts_.push_back(turns[i].start);
        ends_.push_back(turns[i].end);
        speakers_.push_back(turns[i].speaker);
    }
}

std::vector<std::pair<std::string, double>> IntervalTree::query(
    double start, double end) const {
    std::vector<std::pair<std::string, double>> results;
    if (starts_.empty()) return results;

    // np.searchsorted(starts, end, side='left'): first index with start >= end.
    // Only turns before it can start < end (candidate overlaps).
    const std::size_t right_idx = static_cast<std::size_t>(
        std::lower_bound(starts_.begin(), starts_.end(), end) - starts_.begin());
    if (right_idx == 0) return results;

    for (std::size_t idx = 0; idx < right_idx; ++idx) {
        // (starts < end) & (ends > start) — the starts<end half always holds for
        // idx < right_idx, kept for faithfulness to the Python mask.
        if (starts_[idx] < end && ends_[idx] > start) {
            const double intersection =
                std::min(ends_[idx], end) - std::max(starts_[idx], start);
            if (intersection > 0) results.emplace_back(speakers_[idx], intersection);
        }
    }
    return results;
}

std::optional<std::string> IntervalTree::find_nearest(double time) const {
    if (starts_.empty()) return std::nullopt;
    // argmin |midpoint - time|, first-min wins (np.argmin returns the first).
    std::size_t nearest = 0;
    double best = std::abs((starts_[0] + ends_[0]) / 2.0 - time);
    for (std::size_t i = 1; i < starts_.size(); ++i) {
        const double d = std::abs((starts_[i] + ends_[i]) / 2.0 - time);
        if (d < best) {
            best = d;
            nearest = i;
        }
    }
    return speakers_[nearest];
}

}  // namespace whisperx::diarize
