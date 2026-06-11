// Lazy model-asset downloader — the C++ port of the HF-mirror / sherpa-release
// resolvers in whisperx/asr_sherpa.py, whisperx/diarize_sherpa.py, and
// golden/export_align_onnx.py. Pulls the sherpa-onnx ONNX assets from our public
// sha-pinned HF mirrors over libcurl (no huggingface_hub), falling back to
// sherpa-onnx's official .tar.bz2 release tarballs (extracted with libarchive) for
// Whisper / diarization models not on the mirror. Assets are cached under
// WHISPERX_SHERPA_CACHE (default ~/.cache/whisperx-sherpa); a present file is a
// cache hit (no network). models/assets.cpp calls these only after its local-dir
// env-var resolution misses.
#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "models/assets.hpp"  // DiarizeAssets

namespace whisperx::server::assets {

namespace fs = std::filesystem;

// The HF "resolve" download URL for repo/rel_path at a revision (302s to a CDN).
std::string hf_url(const std::string& repo, const std::string& rel_path,
                   const std::string& rev = "main");

// The asset cache root (WHISPERX_SHERPA_CACHE or ~/.cache/whisperx-sherpa).
fs::path cache_root();

// Map a WhisperX model name to its sherpa folder/release key (SHERPA_MODEL_MAP).
std::string map_model(const std::string& whisper_arch);

// Ensure a sherpa Whisper model directory (encoder/decoder/tokens) exists in the
// cache — mirror first, then the sherpa-onnx release tarball. Returns the dir, or
// nullopt if neither source yields the model. `precision` selects the mirror
// variant on contract-v2 metas ("variants" block, see golden/mirror_whisper_onnx.py)
// with the same fallback order models/assets.cpp uses to pick from the dir
// afterwards; v1 metas (no variants) keep the legacy flat-keys path.
std::optional<fs::path> ensure_whisper_dir(
    const std::string& model_name,
    whisperx::server::Precision precision = whisperx::server::Precision::Fp32);

// Map a language code to the mirror folder of its default align model
// (DEFAULT_ALIGN_MODELS_TORCH/_HF with '/' folded to "--", alignment.py:116).
// "" when the language has no default align model.
std::string map_align_model(const std::string& language);

// Ensure the wav2vec2 align directory (model.onnx + meta.json) for a language.
// Mirror-only (align has no sherpa release). Returns the dir, or nullopt.
std::optional<fs::path> ensure_align_dir(const std::string& language);

// Ensure the diarization segmentation + embedding ONNX — mirror first, then the
// sherpa-onnx release (segmentation tarball + embedding file). Returns the paths.
std::optional<whisperx::server::models::DiarizeAssets> ensure_diarize();

}  // namespace whisperx::server::assets
