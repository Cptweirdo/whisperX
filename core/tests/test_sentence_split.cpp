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
        "Multiple   spaces.   Here.", "\"Run!\" She left.",
        "Dr. Müller kam an. Er sprach.", "Am 1. Januar kam er.",
        "Es war 1990. Danach kam Ruhe.", "Schön. Ärgerlich.",
        "M. Dupont est arrivé. Il parle.", "Ça suffit. Écoute-moi.",
        "Voir art. 5 ici. Fin."};
    for (const auto& t : texts) check_invariants(sentence_spans(t, "en"), ncp(t));
    for (const auto& t : texts) check_invariants(sentence_spans(t, "ru"), ncp(t));
    for (const auto& t : texts) check_invariants(sentence_spans(t, "de"), ncp(t));
    for (const auto& t : texts) check_invariants(sentence_spans(t, "fr"), ncp(t));
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

TEST_CASE("English decimals/fractions do not split", "[split][en][number]") {
    // Internal decimal period has no following whitespace -> never a boundary,
    // regardless of locale. The trailing terminator is a real sentence end.
    REQUIRE(sentence_spans("The ratio is 1.23 exactly. Done.", "en").size() == 2);
    REQUIRE(sentence_spans("It is 1.23. Next.", "en").size() == 2);
    REQUIRE(sentence_spans("Value 0.5 and 1.25 here. Ok.", "en").size() == 2);
}

TEST_CASE("German: terminators and accented capitals", "[split][de]") {
    // Ä/Ö/Ü are Latin-1 uppercase starters -> split before them.
    REQUIRE(sentence_spans("Schön. Ärgerlich.", "de").size() == 2);
    REQUIRE(sentence_spans("Die Sonne geht unter. Es wird dunkel.", "de").size() == 2);
    REQUIRE(sentence_spans("Wie geht es dir? Mir gut. Gehen wir!", "de").size() == 3);
}

TEST_CASE("German: Moses abbreviations suppress the boundary", "[split][de][abbrev]") {
    REQUIRE(sentence_spans("Dr. Müller kam an. Er sprach.", "de").size() == 2);
    REQUIRE(sentence_spans("Prof. Bauer und St. Gallen. Ja.", "de").size() == 2);
    REQUIRE(sentence_spans("Siehe Nr. 5 unten. Dort.", "de").size() == 2);
    REQUIRE(sentence_spans("Kauf Brot usw. Das reicht. Ende.", "de").size() == 2);
    // internal-period abbreviations: suppressed via the single-letter prefixes.
    REQUIRE(sentence_spans("Das war z.B. wichtig. Merk dir.", "de").size() == 2);
    REQUIRE(sentence_spans("Es ist d.h. fertig. Gut.", "de").size() == 2);
}

TEST_CASE("German: ordinal abbreviations 1. 2.", "[split][de][ordinal]") {
    // "<digit(s)>." before a capital is a German ordinal (1. = 1st) -> no split.
    REQUIRE(sentence_spans("Am 1. Januar kam er.", "de") == Spans{{0, 20}});
    REQUIRE(sentence_spans("Treffen am 1. und 2. Mai.", "de").size() == 1);
    // a real boundary after the ordinal's clause still splits.
    REQUIRE(sentence_spans("Der 3. Platz reicht. Gut.", "de").size() == 2);
    // 4-digit year is NOT an ordinal -> the sentence boundary stays.
    REQUIRE(sentence_spans("Es war 1990. Danach kam Ruhe.", "de").size() == 2);
    // the rule is de-gated: English is unaffected (period after a lone digit splits).
    REQUIRE(sentence_spans("It was 1. Then came 2.", "en").size() == 2);
}

TEST_CASE("French: terminators and accented capitals", "[split][fr]") {
    REQUIRE(sentence_spans("Le soleil se couche. La nuit tombe.", "fr").size() == 2);
    REQUIRE(sentence_spans("Ça suffit. Écoute-moi.", "fr").size() == 2);
    REQUIRE(sentence_spans("Comment ça va? Très bien. Allons-y!", "fr").size() == 3);
}

TEST_CASE("French: Moses abbreviations suppress the boundary", "[split][fr][abbrev]") {
    REQUIRE(sentence_spans("M. Dupont est arrivé. Il parle.", "fr").size() == 2);
    REQUIRE(sentence_spans("Voir art. 5 et fig. 2 ici. Fin.", "fr").size() == 2);
    REQUIRE(sentence_spans("Cf. chap. 3 ici. Voilà.", "fr").size() == 2);
    // French ordinals are written "1er/1re", not "1." -> no de-style ordinal rule:
    // a lone digit then period before a capital splits like a normal number.
    REQUIRE(sentence_spans("Au 1. Mai ici.", "fr").size() == 2);
}

TEST_CASE("unknown language falls back to English prefixes", "[split]") {
    // "Mr." must still be recognised under an unknown lang (en fallback).
    REQUIRE(sentence_spans("Mr. Smith left. He ran.", "zz").size() == 2);
    check_invariants(sentence_spans("Hello. World.", "klingon"), 13);
}
