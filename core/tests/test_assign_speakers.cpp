// Native unit tests for the Phase-4 speaker-assignment glue (core/diarize):
// IntervalTree overlap/nearest + assign_word_speakers dominant-by-overlap. These
// pin the tie-break + edge branches the RTTM golden won't reach (ties, no-overlap,
// fill_nearest, untimed words, empties) under ASan/UBSan; end-to-end label parity
// vs the Python oracle lives in bindings/test/test_assign_parity.py.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "diarize/assign_speakers.hpp"
#include "diarize/interval_tree.hpp"

using namespace whisperx::diarize;
using Catch::Approx;

namespace {
// A segment with one timed word covering the same span (the common shape).
json seg(double start, double end) {
    return json{{"start", start},
                {"end", end},
                {"words", json::array({json{{"start", start}, {"end", end}}})}};
}
}  // namespace

TEST_CASE("IntervalTree: query overlap + intersection durations") {
    IntervalTree tree({{0.0, 2.0, "A"}, {1.0, 3.0, "B"}, {5.0, 6.0, "C"}});

    auto r = tree.query(0.5, 1.5);
    REQUIRE(r.size() == 2);
    CHECK(r[0].first == "A");
    CHECK(r[0].second == Approx(1.0));  // min(2,1.5)-max(0,0.5)=1.0
    CHECK(r[1].first == "B");
    CHECK(r[1].second == Approx(0.5));  // min(3,1.5)-max(1,0.5)=0.5

    CHECK(tree.query(3.5, 4.5).empty());      // gap between turns
    CHECK(tree.query(6.0, 7.0).empty());      // touching end is not overlap (>)
}

TEST_CASE("IntervalTree: find_nearest is first-min on ties") {
    // Two turns equidistant from t=3.0 (mids 2.0 and 4.0). argmin -> first.
    IntervalTree tree({{1.0, 3.0, "A"}, {3.0, 5.0, "B"}});
    CHECK(tree.find_nearest(3.0) == "A");
    CHECK(tree.find_nearest(4.5) == "B");
    CHECK_FALSE(IntervalTree({}).find_nearest(1.0).has_value());
}

TEST_CASE("IntervalTree: unsorted turns are sorted by start") {
    IntervalTree tree({{5.0, 6.0, "C"}, {0.0, 2.0, "A"}, {1.0, 3.0, "B"}});
    auto r = tree.query(0.0, 10.0);  // all three, ascending start
    REQUIRE(r.size() == 3);
    CHECK(r[0].first == "A");
    CHECK(r[1].first == "B");
    CHECK(r[2].first == "C");
}

TEST_CASE("assign: dominant speaker by summed overlap") {
    std::vector<Turn> turns = {{0.0, 1.2, "A"}, {1.2, 2.0, "B"}};
    json segs = json::array({seg(0.0, 2.0)});
    auto out = assign_word_speakers(turns, segs, false);
    CHECK(out[0]["speaker"] == "A");          // 1.2s A > 0.8s B
    CHECK(out[0]["words"][0]["speaker"] == "A");
}

TEST_CASE("assign: tie goes to the first speaker (insertion order)") {
    // Equal summed overlap (0.5 each). Python max keeps the first -> A (lower
    // start, so first in the IntervalTree query order).
    std::vector<Turn> turns = {{0.0, 0.5, "A"}, {0.5, 1.0, "B"}};
    json segs = json::array({seg(0.0, 1.0)});
    auto out = assign_word_speakers(turns, segs, false);
    CHECK(out[0]["speaker"] == "A");
}

TEST_CASE("assign: no overlap leaves speaker unset unless fill_nearest") {
    std::vector<Turn> turns = {{10.0, 11.0, "A"}};
    json segs = json::array({seg(0.0, 1.0)});

    auto plain = assign_word_speakers(turns, segs, false);
    CHECK_FALSE(plain[0].contains("speaker"));
    CHECK_FALSE(plain[0]["words"][0].contains("speaker"));

    auto filled = assign_word_speakers(turns, segs, true);
    CHECK(filled[0]["speaker"] == "A");
    CHECK(filled[0]["words"][0]["speaker"] == "A");
}

TEST_CASE("assign: word without start is skipped") {
    std::vector<Turn> turns = {{0.0, 2.0, "A"}};
    json segs = json::array({json{
        {"start", 0.0},
        {"end", 2.0},
        {"words", json::array({json{{"word", "uh"}},                       // untimed
                               json{{"start", 0.5}, {"end", 1.0}}})}}});  // timed
    auto out = assign_word_speakers(turns, segs, true);
    CHECK_FALSE(out[0]["words"][0].contains("speaker"));  // untimed: never set
    CHECK(out[0]["words"][1]["speaker"] == "A");
}

TEST_CASE("assign: word end defaults to word start when absent") {
    std::vector<Turn> turns = {{0.0, 2.0, "A"}};
    json segs = json::array({json{
        {"start", 0.0},
        {"end", 2.0},
        {"words", json::array({json{{"start", 1.0}}})}}});  // no end
    auto out = assign_word_speakers(turns, segs, false);
    // query(1.0, 1.0): intersection = min(2,1)-max(0,1) = 0, not > 0 -> no overlap.
    CHECK_FALSE(out[0]["words"][0].contains("speaker"));
}

TEST_CASE("assign: empty turns / empty segments are safe") {
    json segs = json::array({seg(0.0, 1.0)});
    auto no_turns = assign_word_speakers({}, segs, true);
    CHECK_FALSE(no_turns[0].contains("speaker"));

    json empty = json::array();
    auto out = assign_word_speakers({{0.0, 1.0, "A"}}, empty, true);
    CHECK(out.empty());
}
