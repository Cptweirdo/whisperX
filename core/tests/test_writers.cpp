// Catch2 coverage for whisperx::writers (the ResultWriter family port). Full
// cross-language byte/round-trip parity vs the Python writers is in
// bindings/test/test_writers_parity.py; here we pin the pieces easiest to break
// under ASan/UBSan: format_timestamp's integer-ms rounding + hours gating, the
// iterate_result branch matrix (segment cues, line continuation, highlight), and
// the <u> highlight wrap.
#include <catch2/catch_test_macros.hpp>

#include "writers/writers.hpp"

using nlohmann::json;
namespace wr = whisperx::writers;

TEST_CASE("format_timestamp: integer-ms, hours gating, markers", "[writers]") {
    // VTT style: hours omitted when zero, '.' marker.
    REQUIRE(wr::format_timestamp(0.0, false, '.') == "00:00.000");
    REQUIRE(wr::format_timestamp(1.5, false, '.') == "00:01.500");
    REQUIRE(wr::format_timestamp(61.25, false, '.') == "01:01.250");
    // SRT style: hours always shown, ',' marker.
    REQUIRE(wr::format_timestamp(0.0, true, ',') == "00:00:00,000");
    REQUIRE(wr::format_timestamp(3661.0, true, ',') == "01:01:01,000");
    REQUIRE(wr::format_timestamp(3661.0, false, '.') == "01:01:01.000");  // hours>0
    // round(seconds*1000) is banker's (round-half-to-even).
    REQUIRE(wr::format_timestamp(0.0025, false, '.') == "00:00.002");  // 2.5 -> 2
    REQUIRE(wr::format_timestamp(0.0035, false, '.') == "00:00.004");  // 3.5 -> 4
}

TEST_CASE("no-words segments: one cue per segment + speaker prefix", "[writers]") {
    json result = {
        {"segments",
         {{{"start", 0.0}, {"end", 1.5}, {"text", " Hello "}, {"speaker", "SPK"}},
          {{"start", 2.0}, {"end", 3.0}, {"text", "a --> b"}}}}};
    auto cues = wr::iterate_result(result, json::object(), true, ',');
    REQUIRE(cues.size() == 2);
    REQUIRE(cues[0].start == "00:00:00,000");
    REQUIRE(cues[0].end == "00:00:01,500");
    REQUIRE(cues[0].text == "[SPK]: Hello");
    REQUIRE(cues[1].text == "a -> b");  // "-->" sanitized
}

TEST_CASE("word cues: cue spans word min/max, default options group a segment",
          "[writers]") {
    json result = {
        {"language", "en"},
        {"segments",
         {{{"start", 0.0},
           {"end", 1.5},
           {"text", " Hello there"},
           {"words",
            {{{"word", "Hello"}, {"start", 0.1}, {"end", 0.5}},
             {{"word", " there"}, {"start", 0.6}, {"end", 1.4}}}}}}}};
    // No max_line_width/count -> preserve_segments, one cue, joined with spaces.
    auto cues = wr::iterate_result(result, json::object(), false, '.');
    REQUIRE(cues.size() == 1);
    REQUIRE(cues[0].start == "00:00.100");  // min word start
    REQUIRE(cues[0].end == "00:01.400");    // max word end
    REQUIRE(cues[0].text == "Hello  there");
}

TEST_CASE("highlight_words underlines the spoken word per cue", "[writers]") {
    json result = {
        {"language", "en"},
        {"segments",
         {{{"start", 0.0},
           {"end", 1.0},
           {"text", "a b"},
           {"words",
            {{{"word", "a"}, {"start", 0.0}, {"end", 0.4}},
             {{"word", " b"}, {"start", 0.4}, {"end", 1.0}}}}}}}};
    json opts = {{"highlight_words", true}};
    auto cues = wr::iterate_result(result, opts, false, '.');
    // first word starts at the cue start, so no leading gap cue; one cue per word.
    REQUIRE(cues.size() == 2);
    REQUIRE(cues[0].text == "<u>a</u>  b");
    REQUIRE(cues[1].text == "a  <u>b</u>");  // leading space preserved before <u>
}

TEST_CASE("write_tsv: integer-ms columns + header", "[writers]") {
    json result = {
        {"segments", {{{"start", 0.0}, {"end", 1.5}, {"text", "hi\tthere"}}}}};
    REQUIRE(wr::write_tsv(result, json::object()) ==
            "start\tend\ttext\n0\t1500\thi there\n");
}

TEST_CASE("write_aud: python-repr floats + [[speaker]] prefix", "[writers]") {
    json result = {
        {"segments", {{{"start", 0.0}, {"end", 1.5}, {"text", "hi"},
                       {"speaker", "S0"}}}}};
    REQUIRE(wr::write_aud(result, json::object()) == "0.0\t1.5\t[[S0]]hi\n");
}

TEST_CASE("write_json: round-trips the result verbatim", "[writers]") {
    json result = {{"language", "en"},
                   {"segments", {{{"start", 0.0}, {"text", "hi"}}}}};
    REQUIRE(json::parse(wr::write_json(result, json::object())) == result);
}
