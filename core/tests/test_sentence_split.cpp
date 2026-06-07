// Extensive native tests for the rule-based sentence splitter (the punkt
// replacement). Cross-impl agreement with nltk punkt on a broad corpus is pinned
// in bindings/test/test_align_parity.py via the committed baseline; these assert
// the contract invariants + the documented behaviour on terminators,
// abbreviations (Moses prefixes), numbers, Unicode/Cyrillic, quotes, ASR-shaped
// and degenerate inputs, under ASan/UBSan.
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "text/sentence_split.hpp"
#include "text/utf8.hpp"

using whisperx::text::sentence_spans;
using Spans = std::vector<std::pair<std::size_t, std::size_t>>;

namespace {
std::size_t ncp(const std::string& s) {
    return whisperx::text::utf8_chars(s).size();
}
// Assert the universal span contract on any input.
void check_invariants(const Spans& spans, std::size_t n) {
    std::size_t prev_end = 0;
    for (const auto& [s, e] : spans) {
        REQUIRE(s < e);
        REQUIRE(e <= n);
        REQUIRE(s >= prev_end);  // non-overlapping, ascending
        prev_end = e;
    }
}
}  // namespace

TEST_CASE("contract invariants on a sweep", "[split]") {
    const std::vector<std::string> texts = {
        "", " ", "   ", "x", "Hi.", "...", "?!", "One.", "a.b.c",
        "Hello world. Goodbye now.", "Mr. Smith left. He ran.",
        "No. 5 is here. Done.", "Pi is 3.14 today. Yes.",
        "Wait... what? Go home.", "Привет мир. Как дела?",
        "the cat sat the dog ran", "Tab\tand\nnewline. Next.",
        "Multiple   spaces.   Here.", "\"Run!\" She left."};
    for (const auto& t : texts) check_invariants(sentence_spans(t, "en"), ncp(t));
    for (const auto& t : texts) check_invariants(sentence_spans(t, "ru"), ncp(t));
}

TEST_CASE("basic terminators split before a capital", "[split]") {
    REQUIRE(sentence_spans("Hello world. Goodbye now.", "en") ==
            Spans{{0, 12}, {13, 25}});
    REQUIRE(sentence_spans("Stop! Go? Wait.", "en").size() == 3);
    // terminator at EOF → no trailing empty span
    REQUIRE(sentence_spans("Done.", "en") == Spans{{0, 5}});
    // no whitespace after terminator → no split
    REQUIRE(sentence_spans("a.b", "en") == Spans{{0, 3}});
    // lowercase start after period → no split (ASR-friendly; punkt-divergent)
    REQUIRE(sentence_spans("first here. and more", "en") == Spans{{0, 20}});
}

TEST_CASE("Moses abbreviations suppress the boundary", "[split][abbrev]") {
    REQUIRE(sentence_spans("Mr. Smith left. He ran.", "en").size() == 2);
    REQUIRE(sentence_spans("Dr. Jones and Prof. Lee met. Yes.", "en").size() == 2);
    REQUIRE(sentence_spans("J. R. R. Tolkien wrote. Done.", "en").size() == 2);
    REQUIRE(sentence_spans("See e.g. the end here. Now go.", "en").size() == 2);

    SECTION("NUMERIC_ONLY suppresses only before a digit") {
        // "No." before a number → no split
        REQUIRE(sentence_spans("No. 5 is here. Done.", "en").size() == 2);
        // "No." before a capital word → DOES split (numeric-only)
        REQUIRE(sentence_spans("No. The answer is five.", "en").size() == 2);
    }
}

TEST_CASE("numbers, versions, urls do not split", "[split]") {
    REQUIRE(sentence_spans("Pi is 3.14 today. Yes.", "en").size() == 2);
    REQUIRE(sentence_spans("Version 1.2.3 shipped. Ok.", "en").size() == 2);
    REQUIRE(sentence_spans("Email a.b@c.com please.", "en") ==
            Spans{{0, 23}});
    REQUIRE(sentence_spans("Go to example.com/path now.", "en") ==
            Spans{{0, 27}});
}

TEST_CASE("ellipsis and multi-terminator runs", "[split]") {
    // "..." then lowercase → no split; "what?" then capital → split
    REQUIRE(sentence_spans("Wait... what? Go home.", "en").size() == 2);
    // mixed run "?!" consumed together
    REQUIRE(sentence_spans("Really?! Yes indeed.", "en").size() == 2);
    // only punctuation → single span over the trimmed extent
    REQUIRE(sentence_spans("...", "en") == Spans{{0, 3}});
    REQUIRE(sentence_spans("?!", "en") == Spans{{0, 2}});
}

TEST_CASE("Unicode / Cyrillic byte-offset correctness", "[split][unicode]") {
    // "Дом. Сад." — codepoint spans, not byte spans (each Cyrillic char is 2
    // bytes, so byte offsets would be doubled). [0,4) and [5,9) are codepoints.
    REQUIRE(sentence_spans("Дом. Сад.", "ru") == Spans{{0, 4}, {5, 9}});
    // single-letter "Я." is a non-breaking initial (Cyrillic letters are prefixes).
    REQUIRE(sentence_spans("Я. Б.", "ru") == Spans{{0, 5}});
    REQUIRE(sentence_spans("Привет мир. Как дела?", "ru").size() == 2);
    // capital detection over Cyrillic (П, К, Я are uppercase)
    REQUIRE(sentence_spans("Стоп. Иди. Жди.", "ru").size() == 3);
    // lowercase Cyrillic start → no split
    REQUIRE(sentence_spans("привет. мир здесь", "ru") == Spans{{0, 17}});
}

TEST_CASE("quotes and closing punctuation", "[split]") {
    // closing quote rides with the sentence; next capital → split
    REQUIRE(sentence_spans("\"Run!\" She left.", "en").size() == 2);
    REQUIRE(sentence_spans("(Wait.) Then go now.", "en").size() == 2);
}

TEST_CASE("ASR-shaped: lowercase, sparse punctuation -> one span", "[split]") {
    REQUIRE(sentence_spans("the cat sat the dog ran here", "en") ==
            Spans{{0, 28}});
    // leading/trailing whitespace trimmed
    REQUIRE(sentence_spans("  hi there  ", "en") == Spans{{2, 10}});
}

TEST_CASE("degenerate inputs", "[split][edge]") {
    REQUIRE(sentence_spans("", "en").empty());
    REQUIRE(sentence_spans("   ", "en").empty());
    REQUIRE(sentence_spans("\t\n ", "en").empty());
    REQUIRE(sentence_spans("x", "en") == Spans{{0, 1}});
    REQUIRE(sentence_spans("Single sentence no period", "en") == Spans{{0, 25}});
}

TEST_CASE("unknown language falls back to English prefixes", "[split]") {
    // "Mr." must still be recognised under an unknown lang (en fallback).
    REQUIRE(sentence_spans("Mr. Smith left. He ran.", "zz").size() == 2);
    check_invariants(sentence_spans("Hello. World.", "klingon"), 13);
}
