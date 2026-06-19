// Forced-alignment assembly — the C++ replacement for whisperx/alignment.py's
// per-segment trellis→backtrack→merge_repeats→char/word/sentence assembly
// (alignment.py:285-411). Phase 3A: the model forward + log_softmax + char
// cleaning + wildcard extension + tokenization stay in Python (the facade); this
// takes the **already log-softmax'd, wildcard-extended** emission (the committed
// fixed-input golden) + the precomputed tokens, and reproduces the aligned
// subsegments — replacing pandas with struct loops and nltk-punkt with the native
// sentence splitter (text/sentence_split.hpp).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "align/trellis.hpp"

namespace whisperx::align {

using nlohmann::json;

struct AssembleResult {
    bool ok;            // false == backtrack failed (caller appends a stub segment)
    json subsegments;   // array of aligned subsegments (sentence-level), on ok
};

// Assemble one transcript segment's aligned subsegments.
//   emission        — (T, W) extended log-softmax emission view (W incl. wildcard)
//   tokens, blank_id— the mapped token ids + blank, as alignment.py computed them
//   text_clean      — the cleaned char string (merge_repeats labels; codepoints)
//   text            — the original segment text (char/word/sentence enumeration)
//   clean_cdx       — codepoint indices of `text` that map to char_segments, in order
//   t1, t2          — segment start/end seconds (timestamp scaling)
//   language        — selects nospaces handling + the splitter prefix list
//   interpolate_method — "nearest" | "ignore"
//   avg_logprob     — copied onto each subsegment when present
AssembleResult align_assemble(const Emission& emission,
                              const std::vector<int>& tokens, int blank_id,
                              const std::string& text_clean,
                              const std::string& text,
                              const std::vector<int>& clean_cdx, double t1,
                              double t2, const std::string& language,
                              const std::string& interpolate_method,
                              bool return_char_alignments,
                              const std::optional<double>& avg_logprob);

}  // namespace whisperx::align
