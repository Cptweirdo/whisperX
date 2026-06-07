// Catch2 coverage for the difflib SequenceMatcher port (autojunk=False case).
// Cross-language parity against CPython difflib is checked separately by
// bindings/test/test_edits_parity.py; here we pin the block-structure invariants
// and the tie-break in isolation under ASan/UBSan.
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "text/sequence_matcher.hpp"

using whisperx::text::Match;
using whisperx::text::SequenceMatcher;

namespace {

std::vector<std::string> toks(std::initializer_list<const char*> ws) {
    return std::vector<std::string>(ws.begin(), ws.end());
}

// (a, b, size) triples, sentinel included — easy to eyeball against difflib.
std::vector<std::array<int, 3>> blocks(SequenceMatcher& sm) {
    std::vector<std::array<int, 3>> out;
    for (const auto& m : sm.get_matching_blocks()) {
        out.push_back({m.a, m.b, m.size});
    }
    return out;
}

}  // namespace

TEST_CASE("identical sequences are one block + sentinel", "[seqmatch]") {
    SequenceMatcher sm(toks({"a", "b", "c"}), toks({"a", "b", "c"}));
    REQUIRE(blocks(sm) == std::vector<std::array<int, 3>>{{0, 0, 3}, {3, 3, 0}});
}

TEST_CASE("disjoint sequences yield only the sentinel", "[seqmatch]") {
    SequenceMatcher sm(toks({"a", "b"}), toks({"x", "y"}));
    REQUIRE(blocks(sm) == std::vector<std::array<int, 3>>{{2, 2, 0}});
}

TEST_CASE("empty inputs yield the (0,0,0) sentinel", "[seqmatch]") {
    SequenceMatcher sm({}, {});
    REQUIRE(blocks(sm) == std::vector<std::array<int, 3>>{{0, 0, 0}});
}

TEST_CASE("a deletion splits into two coalesced-free blocks", "[seqmatch]") {
    // old "a b c", new "a c": b deleted. Blocks: (0,0,1)=a, (2,1,1)=c, sentinel.
    SequenceMatcher sm(toks({"a", "b", "c"}), toks({"a", "c"}));
    REQUIRE(blocks(sm) ==
            std::vector<std::array<int, 3>>{{0, 0, 1}, {2, 1, 1}, {3, 2, 0}});
}

TEST_CASE("an insertion keeps surrounding matches", "[seqmatch]") {
    // old "a c", new "a b c": b inserted. Blocks: (0,0,1)=a, (1,2,1)=c, then the
    // (len(a)=2, len(b)=3, 0) sentinel.
    SequenceMatcher sm(toks({"a", "c"}), toks({"a", "b", "c"}));
    REQUIRE(blocks(sm) ==
            std::vector<std::array<int, 3>>{{0, 0, 1}, {1, 2, 1}, {2, 3, 0}});
}

TEST_CASE("find_longest_match breaks ties toward the earliest i then j",
          "[seqmatch]") {
    // "a" appears twice in both; the longest match is the leading pair, and the
    // tie-break picks the earliest indices (besti/bestj == 0,0).
    SequenceMatcher sm(toks({"a", "x", "a"}), toks({"a", "y", "a"}));
    const Match m = sm.find_longest_match(0, 3, 0, 3);
    CHECK(m.a == 0);
    CHECK(m.b == 0);
    CHECK(m.size == 1);
}

TEST_CASE("repeats past 200 elements still match (no autojunk pruning)",
          "[seqmatch]") {
    // difflib's autojunk would prune an element appearing > n/100+1 times once
    // n >= 200; with autojunk=False it must not. A 300-long run of one token
    // matches in full.
    std::vector<std::string> a(300, "x");
    std::vector<std::string> b(300, "x");
    SequenceMatcher sm(a, b);
    const auto bl = sm.get_matching_blocks();
    REQUIRE(bl.size() == 2);
    CHECK(bl[0].a == 0);
    CHECK(bl[0].b == 0);
    CHECK(bl[0].size == 300);  // whole run matched, nothing pruned as "popular"
    CHECK(bl[1].size == 0);    // sentinel
}

TEST_CASE("get_matching_blocks is memoized + stable", "[seqmatch]") {
    SequenceMatcher sm(toks({"a", "b", "c"}), toks({"a", "x", "c"}));
    const auto first = blocks(sm);
    const auto second = blocks(sm);
    REQUIRE(first == second);
}
