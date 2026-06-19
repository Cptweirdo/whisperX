// Catch2 coverage for whisperx::vad::merge_chunks (the Vad.merge_chunks port).
// Cross-language parity with the Python oracle is in
// bindings/test/test_vad_parity.py; here we pin the packing contract that is
// easiest to break — the exact flush condition, the curr_end-curr_start>0 guard,
// final-chunk append, and speaker-irrelevance — under ASan/UBSan.
#include <catch2/catch_test_macros.hpp>

#include "vad/merge_chunks.hpp"

using nlohmann::json;
namespace vad = whisperx::vad;

namespace {

vad::VadSegment seg(double s, double e) { return {s, e, std::string("UNKNOWN")}; }

}  // namespace

TEST_CASE("empty input yields no chunks", "[vad]") {
    REQUIRE(vad::merge_chunks({}, 30.0) == json::array());
}

TEST_CASE("a single segment is one chunk", "[vad]") {
    auto out = vad::merge_chunks({seg(1.5, 4.2)}, 30.0);
    REQUIRE(out.size() == 1);
    REQUIRE(out[0]["start"] == 1.5);
    REQUIRE(out[0]["end"] == 4.2);
    REQUIRE(out[0]["segments"] == json::array({json::array({1.5, 4.2})}));
}

TEST_CASE("segments under chunk_size merge into one chunk", "[vad]") {
    // spans 0..25 < 30 -> never flushes mid-loop, one chunk, three sub-segments.
    auto out = vad::merge_chunks({seg(0.0, 10.0), seg(11.0, 20.0), seg(21.0, 25.0)},
                                 30.0);
    REQUIRE(out.size() == 1);
    REQUIRE(out[0]["start"] == 0.0);
    REQUIRE(out[0]["end"] == 25.0);
    REQUIRE(out[0]["segments"].size() == 3);
}

TEST_CASE("crossing chunk_size flushes a new chunk", "[vad]") {
    // seg3.end(35) - curr_start(0) = 35 > 30 and curr_end(20)-0 > 0 -> flush at 20.
    auto out = vad::merge_chunks(
        {seg(0.0, 10.0), seg(15.0, 20.0), seg(30.0, 35.0)}, 30.0);
    REQUIRE(out.size() == 2);
    REQUIRE(out[0]["start"] == 0.0);
    REQUIRE(out[0]["end"] == 20.0);
    REQUIRE(out[0]["segments"].size() == 2);
    REQUIRE(out[1]["start"] == 30.0);
    REQUIRE(out[1]["end"] == 35.0);
    REQUIRE(out[1]["segments"].size() == 1);
}

TEST_CASE("the curr_end-curr_start>0 guard blocks a degenerate first flush",
          "[vad]") {
    // A lone long segment exceeds chunk_size on the first iteration, but
    // curr_end(0)-curr_start>0 is false, so it does NOT flush — one chunk.
    auto out = vad::merge_chunks({seg(0.0, 40.0)}, 30.0);
    REQUIRE(out.size() == 1);
    REQUIRE(out[0]["start"] == 0.0);
    REQUIRE(out[0]["end"] == 40.0);
}

TEST_CASE("onset/offset are ignored; speaker does not affect output", "[vad]") {
    std::vector<vad::VadSegment> a = {{0.0, 5.0, std::string("A")},
                                      {6.0, 9.0, std::string("B")}};
    std::vector<vad::VadSegment> b = {{0.0, 5.0, std::nullopt},
                                      {6.0, 9.0, std::nullopt}};
    REQUIRE(vad::merge_chunks(a, 30.0, 0.5, 0.363) ==
            vad::merge_chunks(b, 30.0, 0.9, std::nullopt));
}
