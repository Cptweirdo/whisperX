// assign_word_speakers — the C++ port of whisperx/diarize.py::assign_word_speakers
// (diarize.py:185-263). Labels each transcript segment + each timed word with the
// speaker whose summed overlap dominates (an IntervalTree query), with an optional
// fill_nearest fallback when there is no overlap. Pure JSON in/out so it carries
// arbitrary segment/word fields untouched; only "speaker" keys are written.
//
// The `speaker_embeddings` passthrough stays in the Python facade — it's a dict
// copy with no algorithm. Phase 4 `assign` token, whisperx_core_lib (no deps).
#pragma once

#include <nlohmann/json.hpp>

#include <vector>

#include "diarize/interval_tree.hpp"

namespace whisperx::diarize {

using nlohmann::json;

// Mutate `segments` (a JSON array of segment objects) in place and return it:
// set "speaker" on each segment (and each word with a "start") by dominant
// overlap with `turns`. `fill_nearest` assigns the nearest turn's speaker when a
// segment/word has no overlap. Tie-break: first speaker (insertion order from the
// IntervalTree query) wins, matching Python's max(dict.items(), key=...).
json assign_word_speakers(const std::vector<Turn>& turns, json segments,
                          bool fill_nearest);

}  // namespace whisperx::diarize
