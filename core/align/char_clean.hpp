// Native char-cleaning — the C++ replacement for the per-segment preprocessing
// in whisperx/alignment.py::align (alignment.py:252-280): lowercase each codepoint,
// map spaces to the wav2vec2 "|" separator, drop leading/trailing whitespace, and
// keep only codepoints whose lowered form is a dictionary key (or any non-space
// codepoint, which align_emission_post maps to the OOV wildcard). Produces exactly
// what the align driver hands to emission_post + align_assemble.
//
// Pure + deterministic. Lowercasing goes through utf8proc (simple 1:1 Unicode
// lower) — matches Python str.lower() on the align-language domain; the
// words.json-exact parity gate guards the 1->many special-casing tail. Lives in the
// always-built core lib so it's Catch2-testable torch-free.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace whisperx::align {

struct CleanResult {
    std::string clean_char;       // "".join(clean_char) == the segment's text_clean
    std::vector<int> clean_cdx;   // codepoint indices into `text` of the kept chars
};

// Simple Unicode lowercase of one scalar value, re-encoded to UTF-8 (utf8proc).
std::string lower_utf8(char32_t cp);

// Char-clean one transcript segment. `language` selects the no-spaces handling
// (ja/zh keep characters verbatim, no ' '->'|'); `dictionary` carries the model's
// UTF-8 char keys (same map emission_post takes). Codepoint indexing == Python str.
CleanResult clean_segment(const std::string& text, const std::string& language,
                          const std::map<std::string, int>& dictionary);

}  // namespace whisperx::align
