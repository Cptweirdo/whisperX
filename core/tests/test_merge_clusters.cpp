// Catch2 tests for merge_speaker_clusters — the centroid agglomeration that
// re-joins clusters sherpa's FastClustering fragmented (merge_clusters.hpp).
#include <catch2/catch_test_macros.hpp>

#include <map>
#include <vector>

#include "diarize/merge_clusters.hpp"

using whisperx::diarize::merge_speaker_clusters;

namespace {
using Centroids = std::map<int, std::vector<float>>;
using Talk = std::map<int, double>;
}  // namespace

TEST_CASE("far centroids stay separate", "[merge_clusters]") {
    Centroids c = {{0, {1.f, 0.f}}, {1, {0.f, 1.f}}};  // cosine distance 1.0
    auto m = merge_speaker_clusters(c, {{0, 10.0}, {1, 10.0}}, 0.3f);
    CHECK(m.at(0) == 0);
    CHECK(m.at(1) == 1);
}

TEST_CASE("near centroids merge into the smaller id", "[merge_clusters]") {
    Centroids c = {{0, {1.f, 0.f}}, {3, {0.99f, 0.05f}}};  // distance ~0.001
    auto m = merge_speaker_clusters(c, {{0, 10.0}, {3, 2.0}}, 0.3f);
    CHECK(m.at(0) == 0);
    CHECK(m.at(3) == 0);
}

TEST_CASE("fragments chain-merge while a distinct speaker survives",
          "[merge_clusters]") {
    // 0 and 2 are fragments of one voice, 1 is orthogonal to both.
    Centroids c = {
        {0, {1.f, 0.f, 0.f}}, {1, {0.f, 1.f, 0.f}}, {2, {0.97f, 0.f, 0.05f}}};
    Talk t = {{0, 100.0}, {1, 50.0}, {2, 5.0}};
    auto m = merge_speaker_clusters(c, t, 0.3f);
    CHECK(m.at(0) == 0);
    CHECK(m.at(2) == 0);
    CHECK(m.at(1) == 1);
}

TEST_CASE("closest pair merges first; weighted mean repels the rest",
          "[merge_clusters]") {
    // d(1,2)~0.044 < d(0,1)~0.106 < threshold-adjacent d(0,2)~0.27. The merge
    // takes (1,2) first; their talk-time-weighted mean lands at d(0,·)~0.16,
    // beyond the 0.12 threshold — so 0 must NOT chain in afterwards.
    Centroids c = {
        {0, {1.f, 0.00f}},
        {1, {1.f, 0.50f}},
        {2, {1.f, 0.95f}}};
    Talk t = {{0, 1.0}, {1, 99.0}, {2, 50.0}};
    auto m = merge_speaker_clusters(c, t, 0.12f);
    CHECK(m.at(1) == 1);
    CHECK(m.at(2) == 1);  // absorbed into the smaller id of the pair
    CHECK(m.at(0) == 0);  // weighted mean moved away; no transitive merge
}

TEST_CASE("zero-norm centroid never merges", "[merge_clusters]") {
    Centroids c = {{0, {1.f, 0.f}}, {1, {0.f, 0.f}}};  // 1: failed extraction
    auto m = merge_speaker_clusters(c, {{0, 10.0}, {1, 10.0}}, 0.99f);
    CHECK(m.at(0) == 0);
    CHECK(m.at(1) == 1);
}

TEST_CASE("threshold 0 disables all merging", "[merge_clusters]") {
    Centroids c = {{0, {1.f, 0.f}}, {1, {1.f, 0.f}}};  // identical
    auto m = merge_speaker_clusters(c, {{0, 1.0}, {1, 1.0}}, 0.0f);
    CHECK(m.at(0) == 0);
    CHECK(m.at(1) == 1);
}

TEST_CASE("single cluster is identity", "[merge_clusters]") {
    Centroids c = {{5, {1.f, 0.f}}};
    auto m = merge_speaker_clusters(c, {{5, 3.0}}, 0.5f);
    REQUIRE(m.size() == 1);
    CHECK(m.at(5) == 5);
}
