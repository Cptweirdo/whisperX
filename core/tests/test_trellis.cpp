// Native unit tests for the forced-alignment Viterbi core (core/align/trellis).
// Golden parity vs the committed emissions lives in bindings/test/test_align_parity.py;
// these exercise the algorithm shape + edge cases in isolation under ASan/UBSan.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <vector>

#include "align/trellis.hpp"

using namespace whisperx::align;

namespace {
// (T, V) row-major emission helper.
Emission view(const std::vector<float>& buf, std::size_t T, std::size_t V) {
    return {buf.data(), T, V};
}
}  // namespace

TEST_CASE("get_trellis shape + boundaries", "[trellis]") {
    // T=2, V=2 (blank=0, token=1), one token.
    std::vector<float> e = {/*t0*/ -1.0f, 0.0f, /*t1*/ -1.0f, 0.0f};
    auto emi = view(e, 2, 2);
    std::vector<int> tokens = {1};
    auto tr = get_trellis(emi, tokens, 0);
    REQUIRE(tr.size() == (2 + 1) * (1 + 1));  // (T+1)*(J+1)
    const std::size_t cols = 2;
    REQUIRE(tr[0 * cols + 0] == 0.0f);                  // trellis[0,0] = 0
    REQUIRE(tr[0 * cols + 1] == -std::numeric_limits<float>::infinity());
    // cumsum of blank column: trellis[1,0] = e[0,0], trellis[2,0]=... but the
    // last J rows of column 0 are overwritten with +inf (T+1-J=2 .. T=2).
    REQUIRE(tr[2 * cols + 0] == std::numeric_limits<float>::infinity());
}

TEST_CASE("backtrack + merge_repeats align two tokens", "[trellis]") {
    // T=4, V=3 (blank=0, a=1, b=2). Frames 0-1 favour a, 2-3 favour b.
    std::vector<float> e = {
        -1.0f, 0.0f, -10.0f,  // t0
        -1.0f, 0.0f, -10.0f,  // t1
        -1.0f, -10.0f, 0.0f,  // t2
        -1.0f, -10.0f, 0.0f,  // t3
    };
    auto emi = view(e, 4, 3);
    std::vector<int> tokens = {1, 2};  // "a","b"
    auto trellis = get_trellis(emi, tokens, 0);
    auto path = backtrack(trellis, emi, tokens, 0);
    REQUIRE(path.has_value());
    REQUIRE_FALSE(path->empty());
    // token indices are 0 then 1, time indices strictly ascending.
    int prev_t = -1;
    bool saw0 = false, saw1 = false;
    for (const auto& p : *path) {
        REQUIRE(p.token_index >= 0);
        REQUIRE(p.token_index <= 1);
        REQUIRE(p.time_index > prev_t);
        prev_t = p.time_index;
        saw0 = saw0 || p.token_index == 0;
        saw1 = saw1 || p.token_index == 1;
    }
    REQUIRE(saw0);
    REQUIRE(saw1);

    auto segs = merge_repeats(*path, "ab");
    REQUIRE(segs.size() == 2);
    REQUIRE(segs[0].label == "a");
    REQUIRE(segs[1].label == "b");
    REQUIRE(segs[0].start < segs[0].end);
    REQUIRE(segs[1].start < segs[1].end);
    REQUIRE(segs[0].start <= segs[1].start);
    // scores are probabilities in [0,1].
    for (const auto& s : segs) {
        REQUIRE(s.score >= 0.0f);
        REQUIRE(s.score <= 1.0001f);
    }
}

TEST_CASE("backtrack fails when tokens cannot be consumed", "[trellis]") {
    // blank dominates every frame; too many tokens for too few frames → None.
    const std::size_t T = 3, V = 4;
    std::vector<float> e(T * V, -100.0f);
    for (std::size_t t = 0; t < T; ++t) e[t * V + 0] = 0.0f;  // blank
    auto emi = view(e, T, V);
    std::vector<int> tokens = {1, 2, 3, 1, 2, 3};
    auto trellis = get_trellis(emi, tokens, 0);
    auto path = backtrack(trellis, emi, tokens, 0);
    REQUIRE_FALSE(path.has_value());
}

TEST_CASE("merge_repeats collapses repeated tokens + averages score", "[trellis]") {
    // Three points on the same token then one on the next.
    std::vector<Point> path = {
        {0, 0, 0.8f}, {0, 1, 0.6f}, {0, 2, 1.0f}, {1, 3, 0.5f}};
    auto segs = merge_repeats(path, "xy");
    REQUIRE(segs.size() == 2);
    REQUIRE(segs[0].label == "x");
    REQUIRE(segs[0].start == 0);
    REQUIRE(segs[0].end == 3);  // last time_index (2) + 1
    REQUIRE(segs[0].score == Catch::Approx((0.8f + 0.6f + 1.0f) / 3.0f));
    REQUIRE(segs[1].label == "y");
    REQUIRE(segs[1].start == 3);
    REQUIRE(segs[1].end == 4);
}

TEST_CASE("merge_repeats indexes UTF-8 transcript by codepoint", "[trellis]") {
    // Cyrillic transcript: token 0 -> "Я" (2-byte), token 1 -> "б".
    std::vector<Point> path = {{0, 0, 1.0f}, {1, 1, 1.0f}};
    auto segs = merge_repeats(path, "Яб");
    REQUIRE(segs.size() == 2);
    REQUIRE(segs[0].label == "Я");
    REQUIRE(segs[1].label == "б");
}

TEST_CASE("merge_words splits on separator", "[trellis]") {
    std::vector<Segment> chars = {
        {"a", 0, 2, 1.0f}, {"b", 2, 4, 1.0f}, {"|", 4, 5, 1.0f},
        {"c", 5, 7, 1.0f}};
    auto words = merge_words(chars, "|");
    REQUIRE(words.size() == 2);
    REQUIRE(words[0].label == "ab");
    REQUIRE(words[0].start == 0);
    REQUIRE(words[0].end == 4);
    REQUIRE(words[1].label == "c");
    REQUIRE(words[1].start == 5);
    REQUIRE(words[1].end == 7);
}
