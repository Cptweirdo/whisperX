#include "models/assets.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using nlohmann::json;

namespace whisperx::server::models {

namespace {

std::optional<std::string> env(const char* key) {
    const char* v = std::getenv(key);
    if (v && v[0]) return std::string(v);
    return std::nullopt;
}

// Pick the asset in `dir` whose filename contains `needle` and has one of the
// given extensions, preferring the fp32 (non-int8) one (asr_sherpa._pick).
std::optional<std::string> pick(const fs::path& dir, const std::string& needle,
                                const std::vector<std::string>& exts) {
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
    std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
        bool ai = a.filename().string().find("int8") != std::string::npos;
        bool bi = b.filename().string().find("int8") != std::string::npos;
        if (ai != bi) return !ai;  // non-int8 first
        return a.filename().string().size() < b.filename().string().size();
    });
    return matches.front().string();
}

std::optional<WhisperAssets> from_whisper_dir(const fs::path& dir) {
    auto enc = pick(dir, "encoder", {".onnx"});
    auto dec = pick(dir, "decoder", {".onnx"});
    auto tok = pick(dir, "tokens", {".txt"});
    if (!enc || !dec || !tok) return std::nullopt;
    return WhisperAssets{*enc, *dec, *tok, 80};
}

}  // namespace

int feature_dim_for(const std::string& model_name) {
    if (model_name == "large-v3" || model_name == "large-v3-turbo") return 128;
    return 80;
}

std::optional<WhisperAssets> resolve_whisper(const std::string& model_name) {
    std::optional<WhisperAssets> a;
    // 1. a per-model subdir under the models root
    if (auto root = env("WHISPERX_SHERPA_MODELS_ROOT")) {
        a = from_whisper_dir(fs::path(*root) / model_name);
    }
    // 2. a single explicit dir (dev: holds the active model)
    if (!a) {
        if (auto dir = env("WHISPERX_SHERPA_WHISPER_DIR"))
            a = from_whisper_dir(fs::path(*dir));
    }
    if (a) a->feature_dim = feature_dim_for(model_name);
    return a;
}

std::optional<AlignAssets> resolve_align(const std::string& language) {
    fs::path dir;
    if (auto root = env("WHISPERX_ALIGN_ONNX_ROOT"))
        dir = fs::path(*root) / language;
    else if (auto single = env("WHISPERX_ALIGN_ONNX_DIR"))
        dir = fs::path(*single);
    else
        return std::nullopt;

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
    return std::nullopt;
}

std::string silero_path() {
    if (auto p = env("WHISPERX_SILERO_ONNX")) return *p;
    return (fs::current_path() / "models" / "silero_vad.onnx").string();
}

}  // namespace whisperx::server::models
