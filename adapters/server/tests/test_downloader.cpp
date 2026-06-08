// Catch2 tests for the model-asset downloader. The URL builder + model-name map
// are pure. The cache-hit path is exercised offline: seed a fully-populated model
// dir under a temp WHISPERX_SHERPA_CACHE and assert ensure_whisper_dir() returns
// it without any network call (a present file is a cache hit).
#include <catch2/catch_test_macros.hpp>

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "assets/downloader.hpp"

namespace fs = std::filesystem;
namespace dl = whisperx::server::assets;

namespace {

void write_file(const fs::path& p, const std::string& body) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << body;
}

}  // namespace

TEST_CASE("hf_url builds the resolve URL", "[downloader]") {
    REQUIRE(dl::hf_url("Owner/repo", "tiny/meta.json") ==
            "https://huggingface.co/Owner/repo/resolve/main/tiny/meta.json");
    REQUIRE(dl::hf_url("R", "f", "abc123") ==
            "https://huggingface.co/R/resolve/abc123/f");
}

TEST_CASE("map_model mirrors SHERPA_MODEL_MAP", "[downloader]") {
    REQUIRE(dl::map_model("tiny") == "tiny");
    REQUIRE(dl::map_model("large") == "large-v3");
    REQUIRE(dl::map_model("turbo") == "large-v3-turbo");
    REQUIRE(dl::map_model("unknown-xyz") == "unknown-xyz");  // verbatim passthrough
}

TEST_CASE("ensure_whisper_dir is a cache hit when fully seeded", "[downloader]") {
    fs::path tmp = fs::temp_directory_path() /
                   ("wx-dl-" + std::to_string(::getpid()));
    fs::remove_all(tmp);
    setenv("WHISPERX_SHERPA_CACHE", tmp.string().c_str(), 1);

    // Seed the mirror-repo layout for "tiny": meta.json naming the three assets,
    // plus the assets themselves — all in the same dir.
    fs::path dir = tmp / "KonstantK/whisper-onnx-sherpa" / "tiny";
    write_file(dir / "meta.json",
               R"({"encoder":"tiny-encoder.onnx","decoder":"tiny-decoder.onnx",)"
               R"("tokens":"tiny-tokens.txt","feature_dim":80})");
    write_file(dir / "tiny-encoder.onnx", "x");
    write_file(dir / "tiny-decoder.onnx", "x");
    write_file(dir / "tiny-tokens.txt", "x");

    auto got = dl::ensure_whisper_dir("tiny");
    REQUIRE(got.has_value());
    REQUIRE(fs::equivalent(*got, dir));

    fs::remove_all(tmp);
    unsetenv("WHISPERX_SHERPA_CACHE");
}
