// Model-asset resolution for the native server — the C++ analog of the local-dir
// branch of asr_sherpa._resolve_sherpa_assets / diarize_sherpa._resolve_diarize_assets
// / the align mirror loader. v1 resolves from local directories (env vars below);
// the libcurl HF-mirror downloader (task 7) plugs in behind this same surface as
// the fallback when a local dir is absent.
//
// Env (all optional; an explicit per-asset dir wins over a root):
//   WHISPERX_SHERPA_MODELS_ROOT   root holding <model>/ dirs (encoder/decoder/tokens)
//   WHISPERX_SHERPA_WHISPER_DIR   a single sherpa-Whisper dir (the active model, dev)
//   WHISPERX_ALIGN_ONNX_ROOT      root holding <lang>/ dirs (model.onnx + meta.json)
//   WHISPERX_ALIGN_ONNX_DIR       a single align dir (one language, dev)
//   WHISPERX_DIARIZE_ONNX_DIR     dir holding the segmentation + embedding ONNX
//   WHISPERX_DIARIZE_SEG_ONNX     explicit segmentation ONNX path
//   WHISPERX_DIARIZE_EMBED_ONNX   explicit embedding ONNX path
//   WHISPERX_SILERO_ONNX          silero VAD ONNX (default ./models/silero_vad.onnx)
#pragma once

#include <map>
#include <optional>
#include <string>

namespace whisperx::server::models {

struct WhisperAssets {
    std::string encoder;
    std::string decoder;
    std::string tokens;
    int feature_dim = 80;  // 128 for large-v3 / large-v3-turbo, else 80
};

struct AlignAssets {
    std::string onnx_path;                  // <dir>/model.onnx
    std::map<std::string, int> dictionary;  // meta.json["dictionary"]
    bool batchable = false;                 // meta.json["batchable"]
    std::string language;                   // meta.json["language"] (fallback)
};

struct DiarizeAssets {
    std::string segmentation;
    std::string embedding;
};

// Mel bin count by Whisper family (FEATURE_DIM in asr_sherpa.py).
int feature_dim_for(const std::string& model_name);

// Resolve a Whisper checkpoint to its three sherpa ONNX assets, or nullopt if no
// local directory holds it (the downloader would handle that case in v2).
std::optional<WhisperAssets> resolve_whisper(const std::string& model_name);

// Resolve the wav2vec2 align ONNX + dictionary for a language, or nullopt.
std::optional<AlignAssets> resolve_align(const std::string& language);

// Resolve the diarization segmentation + embedding ONNX, or nullopt.
std::optional<DiarizeAssets> resolve_diarize();

// The silero VAD ONNX path (env override or ./models/silero_vad.onnx).
std::string silero_path();

}  // namespace whisperx::server::models
