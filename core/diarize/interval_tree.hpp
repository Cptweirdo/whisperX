// Speaker-turn interval tree — the C++ port of whisperx/diarize.py::IntervalTree
// (diarize.py:14-88). Sorted-array + binary-search overlap queries (O(log n)),
// reproducing numpy's searchsorted(side='left') / argmin(first-min) tie-breaks
// exactly so the downstream assign_word_speakers labels match Python bit-for-bit.
// Phase 4 `assign` token — pure (no model/deps), lives in whisperx_core_lib.
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace whisperx::diarize {

// One diarization turn: [start, end) seconds + speaker label.
struct Turn {
    double start;
    double end;
    std::string speaker;
};

class IntervalTree {
   public:
    // Turns are sorted by start (stable, like Python's `sorted(key=x[0])`).
    explicit IntervalTree(const std::vector<Turn>& turns);

    // Turns overlapping [start, end), each with its intersection duration, in
    // ascending-start order (the order Python's np.where(overlaps)[0] yields).
    std::vector<std::pair<std::string, double>> query(double start,
                                                      double end) const;

    // Speaker of the turn whose midpoint is nearest `time` (first-min wins, as
    // np.argmin). nullopt iff the tree is empty.
    std::optional<std::string> find_nearest(double time) const;

    bool empty() const { return starts_.empty(); }

   private:
    std::vector<double> starts_;  // sorted ascending by start
    std::vector<double> ends_;
    std::vector<std::string> speakers_;
};

}  // namespace whisperx::diarize
