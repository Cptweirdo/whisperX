// Catch2 port of oatpp/test/oatpp/encoding/UrlTest.cpp — round-trips random
// binary data (both spaceToPlus modes) and pins the space + Cyrillic escapes.
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <string>

#include "encoding/url.hpp"

using whisperx::server::encoding::Url;

namespace {

std::string random_bytes(std::size_t n) {
    static std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    std::string s(n, '\0');
    for (auto& c : s) c = static_cast<char>(dist(rng));
    return s;
}

}  // namespace

TEST_CASE("encode/decode round-trips random binary (spaceToPlus=false)",
          "[url]") {
    Url::Config config;
    config.spaceToPlus = false;
    for (int i = 0; i < 100; ++i) {
        auto buff = random_bytes(100);
        REQUIRE(Url::decode(Url::encode(buff, config)) == buff);
    }
}

TEST_CASE("encode/decode round-trips random binary (spaceToPlus=true)",
          "[url]") {
    Url::Config config;
    config.spaceToPlus = true;
    for (int i = 0; i < 100; ++i) {
        auto buff = random_bytes(100);
        REQUIRE(Url::decode(Url::encode(buff, config)) == buff);
    }
}

TEST_CASE("space encodes to %20 when spaceToPlus=false", "[url]") {
    Url::Config config;
    config.spaceToPlus = false;
    REQUIRE(Url::encode(" ", config) == "%20");
}

TEST_CASE("space encodes to + when spaceToPlus=true", "[url]") {
    Url::Config config;
    config.spaceToPlus = true;
    REQUIRE(Url::encode(" ", config) == "+");
}

TEST_CASE("Cyrillic string encodes with %20 when spaceToPlus=false", "[url]") {
    Url::Config config;
    config.spaceToPlus = false;
    REQUIRE(Url::encode("Смачна Овсяночка!", config) ==
            "%D0%A1%D0%BC%D0%B0%D1%87%D0%BD%D0%B0%20%D0%9E%D0%B2%D1%81%D1%8F%D0"
            "%BD%D0%BE%D1%87%D0%BA%D0%B0%21");
}

TEST_CASE("Cyrillic string encodes with + when spaceToPlus=true", "[url]") {
    Url::Config config;
    config.spaceToPlus = true;
    REQUIRE(Url::encode("Смачна Овсяночка!", config) ==
            "%D0%A1%D0%BC%D0%B0%D1%87%D0%BD%D0%B0+%D0%9E%D0%B2%D1%81%D1%8F%D0"
            "%BD%D0%BE%D1%87%D0%BA%D0%B0%21");
}

TEST_CASE("decode is lenient with a truncated trailing percent escape",
          "[url]") {
    REQUIRE(Url::decode("abc%") == "abc");
    REQUIRE(Url::decode("abc%4") == "abc");
    REQUIRE(Url::decode("a%41b") == "aAb");  // trailing %XX still decodes
}
