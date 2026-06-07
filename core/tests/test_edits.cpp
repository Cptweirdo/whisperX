// Catch2 coverage for the whisperx::edits port (the app/edits.py algorithms).
// Cross-language parity with the Python oracle is checked by
// bindings/test/test_edits_parity.py; here we pin the contracts that are easiest
// to break in C++ — exact-double borrow arithmetic, untimed words carrying
// NEITHER timing key, input non-mutation, and the IndexError/NoChange throws —
// under ASan/UBSan.
#include <catch2/catch_test_macros.hpp>

#include "edits/edits.hpp"

using nlohmann::json;
namespace ed = whisperx::edits;

namespace {

json word(const char* w, double s, double e) {
    return json{{"word", w}, {"start", s}, {"end", e}};
}

json seg(double s, double e, const char* text, const char* speaker,
         json words) {
    json o = json{{"start", s}, {"end", e}, {"text", text}};
    if (speaker) o["speaker"] = speaker;
    o["words"] = std::move(words);
    return o;
}

}  // namespace

TEST_CASE("group_turns ranges, speaker, bounds and joined text", "[edits]") {
    json segs = json::array(
        {seg(0.0, 1.0, "Hello there.", "SPEAKER_00",
             json::array({word("Hello", 0.0, 0.5), word("there.", 0.5, 1.0)})),
         seg(1.0, 2.0, "How are you?", "SPEAKER_00", json::array()),
         seg(2.0, 3.0, "Fine.", "SPEAKER_01", json::array())});
    json turns = ed::group_turns(segs);
    REQUIRE(turns.size() == 2);
    CHECK(turns[0]["seg_indices"] == json::array({0, 1}));
    CHECK(turns[0]["speaker"] == "SPEAKER_00");
    CHECK(turns[0]["start"] == 0.0);
    CHECK(turns[0]["end"] == 2.0);
    CHECK(turns[0]["text"] == "Hello there. How are you?");
    CHECK(turns[1]["index"] == 1);
    CHECK(turns[1]["speaker"] == "SPEAKER_01");
}

TEST_CASE("undiarized + empty speaker collapse to one None-speaker turn",
          "[edits]") {
    json segs = json::array({seg(0, 1, "a", nullptr, json::array()),
                             seg(1, 2, "b", "", json::array()),
                             seg(2, 3, "c", nullptr, json::array())});
    json turns = ed::group_turns(segs);
    REQUIRE(turns.size() == 1);
    CHECK(turns[0]["speaker"].is_null());
    CHECK(turns[0]["seg_indices"] == json::array({0, 1, 2}));
}

TEST_CASE("next_speaker_key is one past the highest, ignoring non-conforming",
          "[edits]") {
    CHECK(ed::next_speaker_key(json::array({"SPEAKER_00", "SPEAKER_01"})) ==
          "SPEAKER_02");
    CHECK(ed::next_speaker_key(json::array({"SPEAKER_00", "SPEAKER_05", "Alice"})) ==
          "SPEAKER_06");
    CHECK(ed::next_speaker_key(json::array()) == "SPEAKER_00");
}

TEST_CASE("realign keeps survivor timing; untimed words carry NEITHER key",
          "[edits]") {
    json old_words =
        json::array({word("Hello", 0.0, 0.5), word("there.", 0.5, 1.0)});
    json out = ed::realign_words(old_words, "Hello brand there.", 0.0, 1.0);
    REQUIRE(out.size() == 3);
    CHECK(out[0]["word"] == "Hello");
    CHECK(out[0]["start"] == 0.0);
    CHECK(out[2]["word"] == "there.");
    // The inserted "brand" got interpolated timing here (gap exists), but the
    // key-presence contract is what we assert below in the no-timing case.
    CHECK(out[1]["word"] == "brand");
}

TEST_CASE("realign returns [] when no token keeps real timing", "[edits]") {
    // A turn that never had word timing can't preserve any.
    CHECK(ed::realign_words(json::array(), "some new text", nullptr, nullptr)
              .empty());
    CHECK(ed::realign_words(json::array({json{{"word", "x"}}}), "x y", nullptr,
                            nullptr)
              .empty());
}

TEST_CASE("an untimed inserted word with no anchor has neither start nor end",
          "[edits]") {
    // old "A B" timed; new "A B C" — C trails with end=null anchor unusable,
    // so it stays untimed and must carry neither key (not null values).
    json old_words = json::array({word("A", 0.0, 0.5), word("B", 0.5, 1.0)});
    json out = ed::realign_words(old_words, "A B C", nullptr, nullptr);
    REQUIRE(out.size() == 3);
    CHECK(out[2]["word"] == "C");
    CHECK_FALSE(out[2].contains("start"));
    CHECK_FALSE(out[2].contains("end"));
}

TEST_CASE("borrow arithmetic is exact: zero-gap split is bit-for-bit", "[edits]") {
    // A(0,.5) B(.5,1) + inserted "mid": each neighbour lends 0.05.
    json old_words = json::array({word("A", 0.0, 0.5), word("B", 0.5, 1.0)});
    json out = ed::realign_words(old_words, "A mid B", 0.0, 1.0);
    REQUIRE(out.size() == 3);
    CHECK(out[0]["end"].get<double>() == 0.45);    // A lent 0.05
    CHECK(out[1]["start"].get<double>() == 0.45);  // mid
    CHECK(out[1]["end"].get<double>() == 0.55);
    CHECK(out[2]["start"].get<double>() == 0.55);  // B lent 0.05
}

TEST_CASE("borrow respects the neighbour floor: a maxed neighbour can't lend",
          "[edits]") {
    // B is only MIN_WORD_WIDTH wide, so A must cover the whole 0.1 deficit.
    json old_words = json::array({word("A", 0.0, 0.5), word("B", 0.5, 0.6)});
    json out = ed::realign_words(old_words, "A mid B", 0.0, 0.6);
    REQUIRE(out.size() == 3);
    CHECK(out[0]["end"].get<double>() == 0.4);     // A lent the full 0.1
    CHECK(out[1]["start"].get<double>() == 0.4);
    CHECK(out[1]["end"].get<double>() == 0.5);
    CHECK(out[2]["start"].get<double>() == 0.5);   // B unchanged (at the floor)
}

TEST_CASE("coalesce merges short same-speaker neighbours; pure", "[edits]") {
    json segs = json::array(
        {seg(0.0, 0.1, "uh", "SPEAKER_00", json::array({word("uh", 0.0, 0.1)})),
         seg(0.1, 0.5, "hello there", "SPEAKER_00",
             json::array({word("hello", 0.1, 0.3), word("there", 0.3, 0.5)}))});
    const json before = segs;  // value snapshot
    json out = ed::coalesce_segments(segs, 0.2);
    REQUIRE(out.size() == 1);
    CHECK(out[0]["start"] == 0.0);
    CHECK(out[0]["end"] == 0.5);
    CHECK(out[0]["text"] == "uh hello there");
    CHECK(segs == before);  // input never mutated
}

TEST_CASE("apply_turn_edit collapses a multi-segment turn; input untouched",
          "[edits]") {
    json segs = json::array(
        {seg(0.0, 1.0, "a", "SPEAKER_00", json::array()),
         seg(1.0, 2.0, "b", "SPEAKER_00", json::array()),
         seg(2.0, 3.0, "c", "SPEAKER_01", json::array())});
    const json before = segs;
    auto [new_segs, delta] = ed::apply_turn_edit(segs, 0, "Brand new.");
    CHECK(new_segs.size() == 2);
    CHECK(new_segs[0]["text"] == "Brand new.");
    CHECK(new_segs[0]["words"].empty());          // no survivor timing
    CHECK(new_segs[0]["start"] == 0.0);
    CHECK(new_segs[0]["end"] == 2.0);
    CHECK(new_segs[0]["speaker"] == "SPEAKER_00");
    CHECK(delta["seg_range"] == json::array({0, 1}));
    CHECK(delta.contains("ts"));
    CHECK(segs == before);
}

TEST_CASE("empty edit deletes the turn; undo restores it", "[edits]") {
    json segs = json::array({seg(0, 1, "lonely", "SPEAKER_00", json::array())});
    const json before = segs;
    auto [new_segs, delta] = ed::apply_turn_edit(segs, 0, "   ");
    CHECK(new_segs.empty());
    CHECK(delta["new_segment"].is_null());
    auto [restored, hist] = ed::undo_last(new_segs, json::array({delta}));
    CHECK(restored == before);
    CHECK(hist.empty());
}

TEST_CASE("reassign rewrites speaker on segments + words; NoChange on no-op",
          "[edits]") {
    json segs = json::array(
        {seg(0.0, 1.0, "hi", "SPEAKER_00",
             json::array({word("hi", 0.0, 1.0)}))});
    auto [new_segs, delta] = ed::apply_turn_reassign(segs, 0, "SPEAKER_01");
    CHECK(new_segs[0]["speaker"] == "SPEAKER_01");
    CHECK(new_segs[0]["words"][0]["speaker"] == "SPEAKER_01");
    CHECK(delta["new_len"] == 1);
    REQUIRE_THROWS_AS(ed::apply_turn_reassign(segs, 0, "SPEAKER_00"), ed::NoChange);
}

TEST_CASE("out-of-range and negative turn indices throw", "[edits]") {
    json segs = json::array({seg(0, 1, "a", "SPEAKER_00", json::array())});
    REQUIRE_THROWS_AS(ed::apply_turn_edit(segs, 5, "x"), std::out_of_range);
    REQUIRE_THROWS_AS(ed::apply_turn_edit(segs, -1, "x"), std::out_of_range);
    REQUIRE_THROWS_AS(ed::apply_turn_reassign(segs, -1, "S"), std::out_of_range);
    REQUIRE_THROWS_AS(ed::apply_turn_edit(json::array(), 0, "x"),
                      std::out_of_range);
}
