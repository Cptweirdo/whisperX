// Catch2 tests for the translation overlay join + the input guards on the Google
// backend. The overlay is pure (no network); verify_api_key("") and
// google_translate(..., "") fail fast before any HTTP, so they're offline too.
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "translate/google.hpp"
#include "translate/overlay.hpp"

using nlohmann::json;
namespace tr = whisperx::server::translate;

TEST_CASE("start_key formats to ms precision, skips start-less", "[translate]") {
    REQUIRE(*tr::start_key(json(1.5)) == "1.500");
    REQUIRE(*tr::start_key(json(0)) == "0.000");
    REQUIRE_FALSE(tr::start_key(json(nullptr)).has_value());
}

TEST_CASE("build_entries pairs source + translation by start", "[translate]") {
    json segs = json::array({{{"start", 0.0}, {"text", "hello"}},
                             {{"start", 1.0}, {"text", "world"}},
                             {{"text", "no-start-skipped"}}});
    json entries = tr::build_entries(segs, {"hola", "mundo", "ignored"});
    REQUIRE(entries.size() == 2);
    REQUIRE(entries["0.000"]["src"] == "hello");
    REQUIRE(entries["0.000"]["tr"] == "hola");
    REQUIRE(entries["1.000"]["tr"] == "mundo");
}

TEST_CASE("apply_overlay joins translation, marks edited stale", "[translate]") {
    json orig = json::array({{{"start", 0.0}, {"end", 1.0}, {"speaker", "SPEAKER_00"},
                              {"text", "hello"}},
                             {{"start", 1.0}, {"end", 2.0}, {"text", "edited now"}}});
    json overlay = {
        {"version", 2},
        {"entries",
         {{"0.000", {{"src", "hello"}, {"tr", "hola"}}},
          {"1.000", {{"src", "world"}, {"tr", "mundo"}}}}}};

    json out = tr::apply_overlay(orig, overlay);
    REQUIRE(out.size() == 2);
    // Fresh: source text still matches -> translated, not stale, speaker kept.
    REQUIRE(out[0]["text"] == "hola");
    REQUIRE(out[0]["stale"] == false);
    REQUIRE(out[0]["speaker"] == "SPEAKER_00");
    // Stale: the original text changed since translation -> falls back to source.
    REQUIRE(out[1]["text"] == "edited now");
    REQUIRE(out[1]["stale"] == true);
}

TEST_CASE("apply_overlay reads legacy v1 (frozen segments)", "[translate]") {
    json orig = json::array({{{"start", 0.0}, {"text", "hello"}}});
    json overlay = {{"segments", json::array({{{"start", 0.0}, {"text", "hola"}}})}};
    json out = tr::apply_overlay(orig, overlay);
    REQUIRE(out[0]["text"] == "hola");  // v1: exact start match, treated fresh
    REQUIRE(out[0]["stale"] == false);
}

TEST_CASE("verify_api_key rejects an empty key offline", "[translate]") {
    auto [ok, detail] = tr::verify_api_key("   ");
    REQUIRE_FALSE(ok);
    REQUIRE(detail == "Enter an API key to continue.");
}

TEST_CASE("google_translate throws without a key, no-ops on empty input",
          "[translate]") {
    REQUIRE_THROWS_AS(tr::google_translate({"hi"}, "es", ""),
                      tr::TranslationError);
    // Empty input returns empty before any key/network check.
    REQUIRE(tr::google_translate({}, "es", "anything").empty());
}
