// Output writers — the C++ port of whisperx/utils.py's ResultWriter family
// (srt/vtt/txt/tsv/aud/json) + SubtitlesWriter.iterate_result (utils.py:215-468).
// Each takes the result dict + the writer options as JSON and returns the exact
// file bytes the Python writer's `write_result` produces (verbatim, incl. trailing
// newlines from Python `print`). Pure JSON in / std::string out — no deps, the
// `merge_chunks`/`assign` pattern (whisperx_core_lib). Phase 5 `writers` token.
//
// Byte-parity is provable: srt/vtt/txt/tsv timecodes are integer-ms math
// (`format_timestamp` = round(seconds*1000)); aud prints raw floats via a
// Python-`repr`-compatible formatter; json is `dump()` (semantic round-trip
// parity, the Phase-1 store-JSON precedent — values round-trip, key order free).
#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace whisperx::writers {

using nlohmann::json;

// SRT/VTT timecode (utils.py:194). always_hours -> "HH:" always (SRT); else only
// when hours>0 (VTT). decimal_marker ',' for SRT, '.' for VTT. Integer-ms exact.
std::string format_timestamp(double seconds, bool always_include_hours,
                             char decimal_marker);

// One subtitle cue (the (start, end, text) yielded by iterate_result).
struct Cue {
    std::string start;
    std::string end;
    std::string text;
};

// Port of SubtitlesWriter.iterate_result (utils.py:252-361): word-grouping into
// cues, honoring options {max_line_width, max_line_count, highlight_words}, the
// LANGUAGES_WITHOUT_SPACES join, speaker "[SPK]: " prefix, the highlight <u> path,
// and the no-`words` segment fallback. always_hours/decimal_marker pick the
// writer's timecode format (SRT vs VTT).
std::vector<Cue> iterate_result(const json& result, const json& options,
                                bool always_include_hours, char decimal_marker);

// The six write_result bodies — return the full file content (bytes).
std::string write_txt(const json& result, const json& options);
std::string write_srt(const json& result, const json& options);
std::string write_vtt(const json& result, const json& options);
std::string write_tsv(const json& result, const json& options);
std::string write_aud(const json& result, const json& options);
std::string write_json(const json& result, const json& options);

}  // namespace whisperx::writers
