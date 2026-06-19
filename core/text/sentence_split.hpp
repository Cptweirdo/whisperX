// Rule-based sentence splitter — replaces nltk punkt in the forced-alignment
// stage (whisperx/alignment.py:189-196). Returns the per-sentence spans that drive
// the char→sentence grouping + per-sentence interpolate_nans in align().
//
// Why hand-rolled (not ICU / FreeLing / a punkt port): see
// docs/cpp-core-migration-briefs.md Phase 3. Abbreviation handling comes from the
// vendored Moses `nonbreaking_prefixes` (core/text/data/, embedded at build time).
// The algorithm here is the exact spec mirrored by alignment.py::_py_sentence_spans
// — the two must agree byte-for-byte (a parity test gate); keep them in lockstep.
#pragma once

#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>

namespace whisperx::text {

// Split `text` into sentence spans, each a [start, end) pair of **codepoint**
// offsets (Python str-index semantics — what alignment.py's `cdx` indexing and
// `text[sstart:send]` slicing expect), with leading/trailing whitespace excluded.
// `lang` selects the Moses non-breaking-prefix list (fallback: "en"). A text with
// no detected internal boundary yields a single span over its trimmed extent;
// empty/whitespace-only text yields no spans.
std::vector<std::pair<std::size_t, std::size_t>> sentence_spans(
    std::string_view text, std::string_view lang = "en");

}  // namespace whisperx::text
