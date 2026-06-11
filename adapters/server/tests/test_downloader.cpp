// Catch2 tests for the model-asset downloader. The URL builder + model-name map
// are pure. The cache-hit path is exercised offline: seed a fully-populated model
// dir under a temp WHISPERX_SHERPA_CACHE and assert ensure_whisper_dir() returns
// it without any network call (a present file is a cache hit).
#include <catch2/catch_test_macros.hpp>

#include "posix_compat.hpp"

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

TEST_CASE("map_align_model mirrors DEFAULT_ALIGN_MODELS + '--' folding",
          "[downloader]") {
    REQUIRE(dl::map_align_model("en") == "WAV2VEC2_ASR_BASE_960H");
    REQUIRE(dl::map_align_model("de") == "VOXPOPULI_ASR_BASE_10K_DE");
    // HF ids fold '/' to "--" (the mirror's folder naming, alignment.py:116)
    REQUIRE(dl::map_align_model("ru") ==
            "jonatasgrosman--wav2vec2-large-xlsr-53-russian");
    REQUIRE(dl::map_align_model("xx") == "");  // no default align model
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

    // A v1 meta has no variants — a fp16 request takes the same legacy path
    // (cache hit on the flat-key files; no extra fetches to miss on).
    auto fp16 = dl::ensure_whisper_dir("tiny", whisperx::server::Precision::Fp16);
    REQUIRE(fp16.has_value());
    REQUIRE(fs::equivalent(*fp16, dir));

    fs::remove_all(tmp);
    unsetenv("WHISPERX_SHERPA_CACHE");
}

TEST_CASE("ensure_whisper_dir fetches the requested v2 variant", "[downloader]") {
    using whisperx::server::Precision;
    fs::path tmp = fs::temp_directory_path() /
                   ("wx-dl-v2-" + std::to_string(::getpid()));
    fs::remove_all(tmp);
    setenv("WHISPERX_SHERPA_CACHE", tmp.string().c_str(), 1);

    // Contract-v2 meta (golden/mirror_whisper_onnx.py): flat keys = int8,
    // variants carry fp32 (with an external .weights extra) and fp16. Only the
    // fp16 set + tokens are seeded — an fp16 request must be a pure cache hit,
    // while the int8/fp32 requests would need files that aren't there (and the
    // offline fetch of a missing file fails) → nullopt-or-tarball-miss, which
    // proves the right per-variant file list was assembled.
    fs::path dir = tmp / "KonstantK/whisper-onnx-sherpa" / "large-v3-turbo";
    write_file(dir / "meta.json", R"({
        "contract_version": 2,
        "encoder": "turbo-encoder.int8.onnx",
        "decoder": "turbo-decoder.int8.onnx",
        "tokens": "turbo-tokens.txt",
        "variants": {
            "int8": {"encoder": "turbo-encoder.int8.onnx",
                     "decoder": "turbo-decoder.int8.onnx", "files": []},
            "fp32": {"encoder": "turbo-encoder.onnx",
                     "decoder": "turbo-decoder.onnx",
                     "files": ["turbo-encoder.weights"]},
            "fp16": {"encoder": "turbo-encoder.fp16.onnx",
                     "decoder": "turbo-decoder.fp16.onnx", "files": []}
        }})");
    write_file(dir / "turbo-encoder.fp16.onnx", "x");
    write_file(dir / "turbo-decoder.fp16.onnx", "x");
    write_file(dir / "turbo-tokens.txt", "x");

    auto got = dl::ensure_whisper_dir("large-v3-turbo", Precision::Fp16);
    REQUIRE(got.has_value());
    REQUIRE(fs::equivalent(*got, dir));

    // fp32 variant: the .weights extra must be part of the fetched set — seed
    // everything but it offline-misses; with it, cache hit.
    write_file(dir / "turbo-encoder.onnx", "x");
    write_file(dir / "turbo-decoder.onnx", "x");
    write_file(dir / "turbo-encoder.weights", "x");
    auto fp32 = dl::ensure_whisper_dir("large-v3-turbo", Precision::Fp32);
    REQUIRE(fp32.has_value());
    REQUIRE(fs::equivalent(*fp32, dir));

    fs::remove_all(tmp);
    unsetenv("WHISPERX_SHERPA_CACHE");
}
