#include "vad/merge_chunks.hpp"

namespace whisperx::vad {

json merge_chunks(const std::vector<VadSegment>& segments, double chunk_size,
                  double onset, std::optional<double> offset) {
    (void)onset;   // accepted for fidelity with Vad.merge_chunks; unused here.
    (void)offset;  // (the merge logic never references onset/offset.)

    json merged = json::array();
    if (segments.empty()) return merged;  // wrappers guard; bare Python indexes [0]

    double curr_end = 0.0;
    json seg_idxs = json::array();
    double curr_start = segments[0].start;
    for (const auto& seg : segments) {
        if (seg.end - curr_start > chunk_size && curr_end - curr_start > 0) {
            merged.push_back({{"start", curr_start},
                              {"end", curr_end},
                              {"segments", seg_idxs}});
            curr_start = seg.start;
            seg_idxs = json::array();
        }
        curr_end = seg.end;
        seg_idxs.push_back(json::array({seg.start, seg.end}));
    }
    // add final
    merged.push_back(
        {{"start", curr_start}, {"end", curr_end}, {"segments", seg_idxs}});
    return merged;
}

}  // namespace whisperx::vad
