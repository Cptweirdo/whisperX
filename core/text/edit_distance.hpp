// Pure, dependency-free text metrics shared by the engine core.
//
// Ported verbatim from the Python golden baseline
// (`tests/test_baseline_golden.py::_edit_distance`) so the C++ core and the
// Python oracle compute WER/CER identically — the first link in the
// strangler-fig parity loop (Phase 0).
#pragma once

#include <string>
#include <vector>

namespace whisperx::text {

// Levenshtein distance between two token sequences (insert/delete/substitute
// cost 1). O(len(a) * len(b)) time, O(len(b)) space — same recurrence as the
// Python reference.
std::size_t edit_distance(const std::vector<std::string>& a,
                          const std::vector<std::string>& b);

// Convenience overload: character-level distance over two strings.
std::size_t edit_distance(const std::string& a, const std::string& b);

}  // namespace whisperx::text
