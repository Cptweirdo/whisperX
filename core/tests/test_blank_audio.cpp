// Unit tests for strip_blank_audio (core/asr/blank_audio.cpp) — the removal of the
// "[BLANK_AUDIO]" text sherpa-onnx Whisper emits for silent input. Pure string util,
// so it links whisperx_core_lib alone (no sherpa/ffmpeg) and runs under ASan/UBSan.
#include <catch2/catch_test_macros.hpp>

#include <string>

#include "asr/whisper_sherpa.hpp"

using whisperx::asr::strip_blank_audio;

TEST_CASE("strip_blank_audio: marker mid-text removed, double space collapsed",
          "[blank_audio]") {
    std::string s = "ресурс [BLANK_AUDIO] Угу";
    CHECK(strip_blank_audio(s) == 1);
    CHECK(s == "ресурс Угу");
}

TEST_CASE("strip_blank_audio: multiple markers counted", "[blank_audio]") {
    std::string s = "a [BLANK_AUDIO] b [BLANK_AUDIO] c";
    CHECK(strip_blank_audio(s) == 2);
    CHECK(s == "a b c");
}

TEST_CASE("strip_blank_audio: case-insensitive", "[blank_audio]") {
    std::string s = "hello [blank_audio] world";
    CHECK(strip_blank_audio(s) == 1);
    CHECK(s == "hello world");
}

TEST_CASE("strip_blank_audio: tolerates inner whitespace", "[blank_audio]") {
    std::string s = "hello [ BLANK_AUDIO ] world";
    CHECK(strip_blank_audio(s) == 1);
    CHECK(s == "hello world");
}

TEST_CASE("strip_blank_audio: whole-text marker -> empty, trimmed",
          "[blank_audio]") {
    std::string s = "[BLANK_AUDIO]";
    CHECK(strip_blank_audio(s) == 1);
    CHECK(s.empty());
}

TEST_CASE("strip_blank_audio: leading/trailing marker trimmed", "[blank_audio]") {
    std::string s = "[BLANK_AUDIO] foo bar [BLANK_AUDIO]";
    CHECK(strip_blank_audio(s) == 2);
    CHECK(s == "foo bar");
}

TEST_CASE("strip_blank_audio: no marker -> unchanged, count 0", "[blank_audio]") {
    std::string s = "a normal [bracketed] transcript";
    CHECK(strip_blank_audio(s) == 0);
    CHECK(s == "a normal [bracketed] transcript");
}

TEST_CASE("strip_blank_audio: empty string -> count 0", "[blank_audio]") {
    std::string s;
    CHECK(strip_blank_audio(s) == 0);
    CHECK(s.empty());
}
