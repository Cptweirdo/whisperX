#include "models/assets.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "assets/downloader.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

namespace whisperx::server::models {

namespace {

std::optional<std::string> env(const char* key) {
    const char* v = std::getenv(key);
    if (v && v[0]) return std::string(v);
    return std::nullopt;
}

// Precision variant a filename carries: "int8"/"fp16" in the name, else fp32
// (sherpa's plain unsuffixed exports are fp32).
Precision variant_of(const std::string& name) {
    if (name.find("int8") != std::string::npos) return Precision::Int8;
    if (name.find("fp16") != std::string::npos) return Precision::Fp16;
    return Precision::Fp32;
}

// Preference rank of a file's variant under the requested precision (lower
// wins). Fallback keeps models without the requested variant loadable: fp16 →
// [fp16, fp32, int8]; fp32 → [fp32, fp16, int8]; int8 → [int8, fp32, fp16].
// int8 ranks last on the GPU precisions deliberately — it was the original
// "slow CUDA" bug (no CUDA int8 kernels; see CUDA_DECODE_FINDINGS.md).
int variant_rank(Precision file, Precision want) {
    if (file == want) return 0;
    if (file == Precision::Int8) return 2;
    return 1;
}

// Pick the asset in `dir` whose filename contains `needle` and has one of the
// given extensions, preferring the `precision` variant (asr_sherpa._pick is
// the fp32-preferring ancestor of this).
std::optional<std::string> pick(const fs::path& dir, const std::string& needle,
                                const std::vector<std::string>& exts,
                                Precision precision = Precision::Fp32) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return std::nullopt;
    std::vector<fs::path> matches;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        const auto name = e.path().filename().string();
        const auto ext = e.path().extension().string();
        if (name.find(needle) == std::string::npos) continue;
        if (std::find(exts.begin(), exts.end(), ext) == exts.end()) continue;
        matches.push_back(e.path());
    }
    if (matches.empty()) return std::nullopt;
    std::sort(matches.begin(), matches.end(),
              [precision](const auto& a, const auto& b) {
        const auto an = a.filename().string(), bn = b.filename().string();
        int ar = variant_rank(variant_of(an), precision);
        int br = variant_rank(variant_of(bn), precision);
        if (ar != br) return ar < br;
        return an.size() < bn.size();
    });
    return matches.front().string();
}

std::optional<WhisperAssets> from_whisper_dir(const fs::path& dir,
                                              Precision precision) {
    auto enc = pick(dir, "encoder", {".onnx"}, precision);
    auto dec = pick(dir, "decoder", {".onnx"}, precision);
    auto tok = pick(dir, "tokens", {".txt"}, precision);
    if (!enc || !dec || !tok) return std::nullopt;
    return WhisperAssets{*enc, *dec, *tok, 80};
}

// Parse an align dir holding model.onnx + meta.json{dictionary,batchable,language}.
std::optional<AlignAssets> from_align_dir(const fs::path& dir,
                                          const std::string& language) {
    std::error_code ec;
    fs::path onnx = dir / "model.onnx";
    fs::path meta = dir / "meta.json";
    if (!fs::exists(onnx, ec) || !fs::exists(meta, ec)) return std::nullopt;
    AlignAssets out;
    out.onnx_path = onnx.string();
    try {
        json m = json::parse(std::ifstream(meta));
        out.dictionary = m.at("dictionary").get<std::map<std::string, int>>();
        out.batchable = m.value("batchable", false);
        out.language = m.value("language", language);
    } catch (...) {
        return std::nullopt;
    }
    return out;
}

}  // namespace

int feature_dim_for(const std::string& model_name) {
    if (model_name == "large-v3" || model_name == "large-v3-turbo") return 128;
    return 80;
}

std::optional<WhisperAssets> resolve_whisper(const std::string& model_name,
                                             Precision precision) {
    std::optional<WhisperAssets> a;
    // 1. a per-model subdir under the models root
    if (auto root = env("WHISPERX_SHERPA_MODELS_ROOT")) {
        a = from_whisper_dir(fs::path(*root) / model_name, precision);
    }
    // 2. a single explicit dir (dev: holds the active model)
    if (!a) {
        if (auto dir = env("WHISPERX_SHERPA_WHISPER_DIR"))
            a = from_whisper_dir(fs::path(*dir), precision);
    }
    // 3. lazy-download from the HF mirror / sherpa release into the cache
    if (!a) {
        if (auto dir = whisperx::server::assets::ensure_whisper_dir(model_name,
                                                                    precision))
            a = from_whisper_dir(*dir, precision);
    }
    if (a) a->feature_dim = feature_dim_for(model_name);
    return a;
}

namespace {
// ggml-<sherpa-key>[-quant].bin — the official ggerganov/whisper.cpp file naming.
std::string ggml_filename(const std::string& model_name,
                          const std::string& quant) {
    std::string key =
        whisperx::server::assets::map_model(model_name);  // turbo→large-v3-turbo
    return "ggml-" + key + (quant.empty() ? "" : "-" + quant) + ".bin";
}
}  // namespace

std::optional<GgmlWhisperAssets> resolve_whisper_ggml(
    const std::string& model_name, const std::string& quant) {
    std::error_code ec;
    // 1. an explicit single .bin (dev: the active model)
    if (auto f = env("WHISPERX_GGML_MODEL")) {
        if (fs::exists(*f, ec)) return GgmlWhisperAssets{*f};
    }
    // 2. a per-model file under a models root
    if (auto root = env("WHISPERX_GGML_MODELS_ROOT")) {
        fs::path p = fs::path(*root) / ggml_filename(model_name, quant);
        if (fs::exists(p, ec)) return GgmlWhisperAssets{p.string()};
    }
    // 3. lazy-download from the official ggerganov/whisper.cpp HF repo
    if (auto p = whisperx::server::assets::ensure_ggml_whisper(model_name, quant))
        return GgmlWhisperAssets{p->string()};
    return std::nullopt;
}

std::optional<AlignAssets> resolve_align(const std::string& language) {
    // 1. a local dir (env): a per-language root, or a single explicit dir.
    if (auto root = env("WHISPERX_ALIGN_ONNX_ROOT")) {
        if (auto a = from_align_dir(fs::path(*root) / language, language))
            return a;
    } else if (auto single = env("WHISPERX_ALIGN_ONNX_DIR")) {
        if (auto a = from_align_dir(fs::path(*single), language)) return a;
    }
    // 2. lazy-download from the HF mirror (align is mirror-only).
    if (auto dir = whisperx::server::assets::ensure_align_dir(language))
        return from_align_dir(*dir, language);
    return std::nullopt;
}

std::optional<DiarizeAssets> resolve_diarize() {
    auto seg = env("WHISPERX_DIARIZE_SEG_ONNX");
    auto emb = env("WHISPERX_DIARIZE_EMBED_ONNX");
    if (seg && emb) return DiarizeAssets{*seg, *emb};

    if (auto dir = env("WHISPERX_DIARIZE_ONNX_DIR")) {
        std::error_code ec;
        fs::path root(*dir);
        std::optional<std::string> seg_p, emb_p;
        // Walk the dir tree: segmentation = an .onnx with "seg" in its path;
        // embedding = any other .onnx (mirrors diarize_sherpa._resolve_local).
        for (const auto& e : fs::recursive_directory_iterator(root, ec)) {
            if (!e.is_regular_file() || e.path().extension() != ".onnx") continue;
            auto p = e.path().string();
            auto lower = p;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (lower.find("int8") != std::string::npos) continue;  // prefer fp32
            if (lower.find("seg") != std::string::npos) {
                if (!seg_p) seg_p = p;
            } else if (lower.find("camp") != std::string::npos ||
                       lower.find("embed") != std::string::npos ||
                       lower.find("wespeaker") != std::string::npos) {
                if (!emb_p) emb_p = p;
            }
        }
        if (seg_p && emb_p) return DiarizeAssets{*seg_p, *emb_p};
    }
    // lazy-download from the HF mirror / sherpa release into the cache
    return whisperx::server::assets::ensure_diarize();
}

std::string silero_path() {
    if (auto p = env("WHISPERX_SILERO_ONNX")) return *p;
    return (fs::current_path() / "models" / "silero_vad.onnx").string();
}

}  // namespace whisperx::server::models
