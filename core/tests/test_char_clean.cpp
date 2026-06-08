// Native unit tests for the align driver's char-cleaner (core/align/char_clean):
// the alignment.py:252-280 preprocessing — utf8proc lowercase, ' '->'|', leading/
// trailing whitespace skip, dict-key / OOV-wildcard keep, codepoint indexing.
// Byte-parity vs the committed goldens lives in bindings/test/test_char_clean_parity;
// these pin the algorithm + edge cases in isolation under ASan/UBSan.
#include <catch2/catch_test_macros.hpp>

#include <map>
#include <string>
#include <vector>

#include "align/char_clean.hpp"

using namespace whisperx::align;

namespace {
// A small lowercase English dict (a-z subset + the wav2vec2 separator).
std::map<std::string, int> en_dict() {
    std::map<std::string, int> d{{"|", 0}};
    int i = 1;
    for (char c = 'a'; c <= 'z'; ++c) d[std::string(1, c)] = i++;
    return d;
}
}  // namespace

TEST_CASE("lower_utf8: Latin / Cyrillic / Greek simple lowercase") {
    CHECK(lower_utf8(U'A') == "a");
    CHECK(lower_utf8(U'z') == "z");          // already lower — identity
    CHECK(lower_utf8(U'Я') == "\xD1\x8F");  // Я -> я (U+044F)
    CHECK(lower_utf8(U'Α') == "\xCE\xB1");  // Α -> α (U+03B1)
    CHECK(lower_utf8(U'1') == "1");          // non-letter — identity
}

TEST_CASE("clean_segment: basic lowercase + leading/trailing skip") {
    auto r = clean_segment(" Hello ", "en", en_dict());
    CHECK(r.clean_char == "hello");
    CHECK(r.clean_cdx == std::vector<int>{1, 2, 3, 4, 5});
}

TEST_CASE("clean_segment: space -> '|' for spaced languages") {
    std::map<std::string, int> d{{"a", 1}, {"b", 2}, {"|", 0}};
    auto r = clean_segment("a b", "en", d);
    CHECK(r.clean_char == "a|b");
    CHECK(r.clean_cdx == std::vector<int>{0, 1, 2});
}

TEST_CASE("clean_segment: OOV digits/symbols kept (wildcard), spaces dropped") {
    std::map<std::string, int> d{{"a", 1}, {"|", 0}};
    auto r = clean_segment("a1.", "en", d);  // '1' and '.' are OOV but kept
    CHECK(r.clean_char == "a1.");
    CHECK(r.clean_cdx == std::vector<int>{0, 1, 2});

    // A bare-space OOV (dict without '|') is still dropped, not wildcarded.
    auto r2 = clean_segment("a b", "en", std::map<std::string, int>{{"a", 1}});
    CHECK(r2.clean_char == "ab");  // 'a', '|'(dropped: not dict & is pipe), 'b'
    CHECK(r2.clean_cdx == std::vector<int>{0, 2});
}

TEST_CASE("clean_segment: no-spaces language keeps chars, drops spaces, no '|'") {
    // zh: each non-space char kept verbatim (OOV -> wildcard), spaces removed.
    auto r = clean_segment("\xE4\xBD\xA0 \xE5\xA5\xBD", "zh", {});  // "你 好"
    CHECK(r.clean_char == "\xE4\xBD\xA0\xE5\xA5\xBD");             // "你好"
    CHECK(r.clean_cdx == std::vector<int>{0, 2});  // codepoint idx skips the space
}

TEST_CASE("clean_segment: codepoint (not byte) indices with multibyte chars") {
    // "яeя x": я is 2 bytes; clean_cdx must count codepoints. dict: e, x, |.
    std::map<std::string, int> d{{"e", 1}, {"x", 2}, {"|", 0}};
    // Split literals so the letter after \x8F isn't swallowed into the hex escape.
    auto r = clean_segment("\xD1\x8F" "e" "\xD1\x8F" " x", "en", d);
    CHECK(r.clean_char == "\xD1\x8F" "e" "\xD1\x8F" "|x");  // я kept as OOV wildcard
    CHECK(r.clean_cdx == std::vector<int>{0, 1, 2, 3, 4});
}

TEST_CASE("clean_segment: empty / all-whitespace / all-OOV-dropped") {
    CHECK(clean_segment("", "en", en_dict()).clean_char.empty());
    CHECK(clean_segment("   ", "en", en_dict()).clean_char.empty());
    // Interior whitespace run skipped only at the ends; a middle tab -> not space
    // char ' ', so '\t'.lower() == '\t', OOV, kept as wildcard.
    auto r = clean_segment("a\tb", "en", en_dict());
    CHECK(r.clean_char == "a\tb");
}
