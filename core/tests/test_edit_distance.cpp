#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "build_info.hpp"
#include "text/edit_distance.hpp"

using whisperx::text::edit_distance;

TEST_CASE("token edit distance matches the Python reference", "[text]") {
    REQUIRE(edit_distance(std::vector<std::string>{}, {}) == 0);
    REQUIRE(edit_distance(std::vector<std::string>{"a", "b", "c"},
                          std::vector<std::string>{"a", "b", "c"}) == 0);
    // one substitution
    REQUIRE(edit_distance(std::vector<std::string>{"the", "cat"},
                          std::vector<std::string>{"the", "dog"}) == 1);
    // one insertion + one deletion
    REQUIRE(edit_distance(std::vector<std::string>{"a", "b", "c"},
                          std::vector<std::string>{"a", "x", "c", "d"}) == 2);
    // empty hypothesis -> distance == reference length (WER denominator path)
    REQUIRE(edit_distance(std::vector<std::string>{"a", "b", "c"}, {}) == 3);
}

TEST_CASE("character edit distance (CER path)", "[text]") {
    REQUIRE(edit_distance(std::string("kitten"), std::string("sitting")) == 3);
    REQUIRE(edit_distance(std::string(""), std::string("abc")) == 3);
    // UTF-8 bytes: Cyrillic differs byte-wise; we only assert determinism here
    REQUIRE(edit_distance(std::string("да"), std::string("да")) == 0);
}

TEST_CASE("build_info exposes version + linked deps", "[provenance]") {
    const std::string info = whisperx::build_info_json();
    REQUIRE(info.find("\"version\"") != std::string::npos);
    REQUIRE(info.find("\"nlohmann_json\"") != std::string::npos);
}
