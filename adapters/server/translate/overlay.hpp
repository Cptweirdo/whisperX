// Structure-locked translation overlay — the C++ port of
// app/translation_overlay.py. A translation is NOT a frozen copy: it stores only
// the translated strings keyed by each source segment's start time, alongside the
// source text they came from. Segment boundaries / speaker / turn grouping always
// come from the *current* original at render time, so a speaker reassignment (or
// rename) propagates to every language for free and a translation can never drift
// to a different turn count than the original. build_entries() builds the v2
// payload; apply_overlay() joins it onto the current original segments, marking a
// segment `stale` (falling back to the original text) when the source text changed
// or no translation exists yet.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace whisperx::server::translate {

using nlohmann::json;

// Stable per-segment key: start time to ms precision, or nullopt for a start-less
// segment (which therefore can carry no translation). Mirrors start_key().
std::optional<std::string> start_key(const json& start);

// Map start_key -> {"src": original_text, "tr": translated_text}, pairing each
// source segment with its translation positionally. Start-less segments skipped.
json build_entries(const json& segments,
                   const std::vector<std::string>& translated_texts);

// Join a translation overlay onto the current original segments: one view segment
// per original, carrying the original's start/end/speaker and either the
// translated text (stale=false) or the original text as fallback (stale=true).
// Handles both v2 (entries) and legacy v1 (frozen segments) payload shapes.
json apply_overlay(const json& orig_segments, const json& overlay);

}  // namespace whisperx::server::translate
