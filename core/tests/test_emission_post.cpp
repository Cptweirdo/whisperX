// Native unit tests for the Phase-3B post-forward step (core/align/emission_post):
// log_softmax + OOV wildcard column + tokenization. Numeric parity vs the committed
// torch emissions lives in bindings/test/test_align_onnx_forward_parity.py; these
// pin the algorithm shape + edge cases in isolation under ASan/UBSan.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "align/emission_post.hpp"

using namespace whisperx::align;
using Catch::Approx;

namespace {
// log_softmax of [1,2,3] (max-subtract): logsumexp = 3 + log(e^-2+e^-1+1).
constexpr double kLs0 = -2.4076059644443806;  // 1 - logsumexp
constexpr double kLs1 = -1.4076059644443806;  // 2 - logsumexp
constexpr double kLs2 = -0.4076059644443806;  // 3 - logsumexp
}  // namespace

TEST_CASE("emission_post: log_softmax, no wildcard") {
    // T=2, V=3; row1 is row0 reversed so the log-softmax mirrors.
    std::vector<float> logits = {1, 2, 3, 3, 2, 1};
    std::map<std::string, int> dict = {{"|", 0}, {"a", 1}, {"b", 2}};

    auto r = emission_post(logits.data(), 2, 3, /*blank_id=*/0, "ab", dict);

    REQUIRE(r.T == 2);
    REQUIRE(r.V == 3);  // no extension
    REQUIRE(r.tokens == std::vector<int>{1, 2});
    CHECK(r.emission[0] == Approx(kLs0).margin(1e-5));
    CHECK(r.emission[1] == Approx(kLs1).margin(1e-5));
    CHECK(r.emission[2] == Approx(kLs2).margin(1e-5));
    // row1 = [3,2,1] -> reversed log-softmax
    CHECK(r.emission[3] == Approx(kLs2).margin(1e-5));
    CHECK(r.emission[4] == Approx(kLs1).margin(1e-5));
    CHECK(r.emission[5] == Approx(kLs0).margin(1e-5));
    // each frame's log-softmax exponentiates to 1.
    CHECK(std::exp(r.emission[0]) + std::exp(r.emission[1]) +
              std::exp(r.emission[2]) ==
          Approx(1.0).margin(1e-6));
}

TEST_CASE("emission_post: OOV char extends a wildcard column") {
    std::vector<float> logits = {1, 2, 3, 3, 2, 1};
    std::map<std::string, int> dict = {{"a", 1}, {"b", 2}};  // 'x' is OOV

    auto r = emission_post(logits.data(), 2, 3, /*blank_id=*/0, "x", dict);

    REQUIRE(r.V == 4);                            // V+1
    REQUIRE(r.tokens == std::vector<int>{3});     // wildcard id == old V
    // row0 cols 0..2 unchanged; col3 = max non-blank (cols 1,2) = kLs2.
    CHECK(r.emission[0] == Approx(kLs0).margin(1e-5));
    CHECK(r.emission[1] == Approx(kLs1).margin(1e-5));
    CHECK(r.emission[2] == Approx(kLs2).margin(1e-5));
    CHECK(r.emission[3] == Approx(kLs2).margin(1e-5));  // wildcard col, row0
    // row1 = [3,2,1] -> ls [kLs2,kLs1,kLs0]; non-blank max(cols1,2)=max(kLs1,kLs0)=kLs1
    CHECK(r.emission[7] == Approx(kLs1).margin(1e-5));  // wildcard col, row1
}

TEST_CASE("emission_post: blank_id excluded from the wildcard max") {
    // Make the blank column the largest; the wildcard max must ignore it.
    std::vector<float> logits = {9, 2, 1};  // T=1, V=3, blank col0 dominates
    std::map<std::string, int> dict = {{"a", 1}};  // 'z' OOV -> wildcard

    auto r = emission_post(logits.data(), 1, 3, /*blank_id=*/0, "z", dict);

    REQUIRE(r.V == 4);
    // log-softmax: m=9, logsumexp≈9; ls0≈0(-eps), ls1≈-7, ls2≈-8.
    // wildcard = max(ls1, ls2) = ls1 (NOT ls0, the blank).
    CHECK(r.emission[3] == Approx(r.emission[1]).margin(1e-6));
    CHECK(r.emission[3] < r.emission[0]);  // strictly below the (excluded) blank
}

TEST_CASE("emission_post: mixed in-dict + OOV tokens in one string") {
    // The realistic wildcard case: some codepoints map to real ids, some (a digit
    // here) fall through to the wildcard column — the token list interleaves both.
    std::vector<float> logits = {1, 2, 3};  // T=1, V=3
    std::map<std::string, int> dict = {{"a", 1}, {"b", 2}};  // '5' is OOV

    auto r = emission_post(logits.data(), 1, 3, /*blank_id=*/0, "a5b", dict);

    REQUIRE(r.V == 4);                                  // extended once
    REQUIRE(r.tokens == std::vector<int>{1, 3, 2});     // real, wildcard(=V), real
    CHECK(r.emission[3] == Approx(kLs2).margin(1e-5));  // wildcard col = max non-blank
}

TEST_CASE("emission_post: non-zero blank_id excluded from wildcard max") {
    // blank is column 1 (not 0) and dominates; the wildcard max must skip col 1 and
    // take max(col0, col2). log_softmax([1,5,1]): col1≈-0.036, col0≈col2≈-4.036.
    std::vector<float> logits = {1, 5, 1};  // T=1, V=3
    std::map<std::string, int> dict = {{"a", 0}, {"b", 2}};  // 'x' OOV

    auto r = emission_post(logits.data(), 1, 3, /*blank_id=*/1, "x", dict);

    REQUIRE(r.V == 4);
    REQUIRE(r.tokens == std::vector<int>{3});
    CHECK(r.emission[3] == Approx(r.emission[0]).margin(1e-6));  // == col0 (non-blank)
    CHECK(r.emission[3] < r.emission[1]);  // strictly below the excluded blank (col1)
}

TEST_CASE("emission_post: degenerate inputs stay memory-safe (ASan/UBSan)") {
    // These aren't reachable through align() (empty clean_char is skipped upstream;
    // real models have blank_id 0 and V≥29), but assert no OOB read/write — blank_id
    // is only ever a comparison, never an index, and the loops are bounded by T,V.
    std::vector<float> logits = {1, 2, 3};  // T=1, V=3
    std::map<std::string, int> dict = {{"a", 1}};

    SECTION("empty text_clean -> empty tokens, no extension") {
        auto r = emission_post(logits.data(), 1, 3, /*blank_id=*/0, "", dict);
        CHECK(r.V == 3);
        CHECK(r.tokens.empty());
        CHECK(r.emission.size() == 3);
    }
    SECTION("out-of-range blank_id excludes nothing (no indexing)") {
        // blank_id=99 (>= V): the wildcard max spans every column, no OOB.
        auto r = emission_post(logits.data(), 1, 3, /*blank_id=*/99, "x", dict);
        REQUIRE(r.V == 4);
        CHECK(r.emission[3] == Approx(kLs2).margin(1e-5));  // max over all cols
    }
    SECTION("negative blank_id is equally safe") {
        auto r = emission_post(logits.data(), 1, 3, /*blank_id=*/-7, "x", dict);
        REQUIRE(r.V == 4);
        CHECK(r.emission[3] == Approx(kLs2).margin(1e-5));
    }
}

TEST_CASE("emission_post: UTF-8 codepoint dictionary lookup") {
    std::vector<float> logits = {0, 0, 0};  // T=1, V=3 (uniform)
    std::map<std::string, int> dict = {{"ё", 1}, {"|", 2}};  // Cyrillic key (2 bytes)

    auto r = emission_post(logits.data(), 1, 3, /*blank_id=*/0, "ё", dict);

    REQUIRE(r.V == 3);  // 'ё' is in-dict -> no wildcard
    REQUIRE(r.tokens == std::vector<int>{1});
}
