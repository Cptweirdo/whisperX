#include "assets/downloader.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <system_error>

#include <archive.h>
#include <archive_entry.h>
#include <nlohmann/json.hpp>

#include "http/curl_client.hpp"
#include "log/log.hpp"
#include "secrets/keyring.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

namespace whisperx::server::assets {

namespace {

// Mirror repos — identical to the Python backends (asr_sherpa / diarize_sherpa /
// export_align_onnx). Public, so a token is optional (sent as a bearer when set
// for rate-limits / future gated assets).
constexpr const char* WHISPER_REPO = "KonstantK/whisper-onnx-sherpa";
constexpr const char* ALIGN_REPO = "KonstantK/wav2vec2-align-onnx";
constexpr const char* DIARIZE_REPO = "KonstantK/diarize-onnx-sherpa";

// sherpa-onnx official release fallbacks (asr_sherpa.py:45 / diarize_sherpa.py:46).
constexpr const char* SHERPA_WHISPER_RELEASE =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-whisper-";  // + <model>.tar.bz2
constexpr const char* SEG_RELEASE =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2";
constexpr const char* EMBED_RELEASE =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx";

std::optional<std::string> env(const char* key) {
    const char* v = std::getenv(key);
    if (v && v[0]) return std::string(v);
    return std::nullopt;
}

std::shared_ptr<spdlog::logger> logger() { return log::get("downloader"); }

// libcurl options carrying the HF bearer token when one is resolvable.
net::Options auth_opts() {
    net::Options o;
    if (auto b = net::bearer_header(secrets::resolve_hf_token()))
        o.headers.push_back(*b);
    return o;
}

// Download repo/rel_path into <cache>/<repo>/<rel_path>; a present file is a cache
// hit (no network). Returns the local path, or nullopt on 404 / network failure.
std::optional<fs::path> fetch_file(const std::string& repo,
                                   const std::string& rel_path) {
    fs::path dest = cache_root() / repo / rel_path;
    std::error_code ec;
    if (fs::exists(dest, ec)) return dest;
    long st = net::download_to_file(hf_url(repo, rel_path), dest.string(),
                                    auth_opts());
    if (st >= 200 && st < 300) return dest;
    return std::nullopt;
}

// Extract a (bzip2/gzip) tar at `archive_path` into `dest_dir` with libarchive.
bool extract_tar(const fs::path& archive_path, const fs::path& dest_dir) {
    struct archive* a = archive_read_new();
    archive_read_support_filter_bzip2(a);
    archive_read_support_filter_gzip(a);
    archive_read_support_format_tar(a);

    struct archive* ext = archive_write_disk_new();
    archive_write_disk_set_options(
        ext, ARCHIVE_EXTRACT_TIME | ARCHIVE_EXTRACT_PERM |
                 ARCHIVE_EXTRACT_SECURE_NODOTDOT |
                 ARCHIVE_EXTRACT_SECURE_SYMLINKS);

    bool ok = true;
    if (archive_read_open_filename(a, archive_path.string().c_str(), 10240) !=
        ARCHIVE_OK) {
        logger()->error("tar open failed: {}", archive_error_string(a));
        archive_read_free(a);
        archive_write_free(ext);
        return false;
    }

    struct archive_entry* entry;
    int r;
    while ((r = archive_read_next_header(a, &entry)) == ARCHIVE_OK) {
        const char* name = archive_entry_pathname(entry);
        if (!name) continue;
        fs::path target = dest_dir / name;  // SECURE_NODOTDOT guards traversal
        archive_entry_set_pathname(entry, target.string().c_str());
        if (archive_write_header(ext, entry) != ARCHIVE_OK) {
            logger()->error("tar write header failed: {}",
                            archive_error_string(ext));
            ok = false;
            break;
        }
        const void* buff;
        size_t size;
        la_int64_t offset;
        while ((r = archive_read_data_block(a, &buff, &size, &offset)) ==
               ARCHIVE_OK) {
            if (archive_write_data_block(ext, buff, size, offset) != ARCHIVE_OK) {
                ok = false;
                break;
            }
        }
        if (r != ARCHIVE_EOF && r != ARCHIVE_OK) ok = false;
        archive_write_finish_entry(ext);
    }
    if (r != ARCHIVE_EOF && r != ARCHIVE_OK) ok = false;

    archive_read_close(a);
    archive_read_free(a);
    archive_write_close(ext);
    archive_write_free(ext);
    return ok;
}

bool dir_has_onnx(const fs::path& dir, const std::string& needle = "") {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return false;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        if (e.path().extension() != ".onnx") continue;
        if (needle.empty() || e.path().filename().string().find(needle) !=
                                  std::string::npos)
            return true;
    }
    return false;
}

// Download a .tar.bz2 release into the cache and extract it; returns the extracted
// top-level dir (cache/<subdir>). Skips the download if already extracted.
std::optional<fs::path> fetch_release_tarball(const std::string& url,
                                              const std::string& subdir) {
    fs::path cache = cache_root();
    fs::path out = cache / subdir;
    if (dir_has_onnx(out)) return out;

    fs::path tmp = cache / (subdir + ".tar.bz2");
    std::error_code ec;
    fs::create_directories(cache, ec);
    logger()->info("Downloading sherpa release tarball: {}", url);
    long st = net::download_to_file(url, tmp.string());
    if (st < 200 || st >= 300) {
        logger()->warn("tarball download failed (HTTP {}): {}", st, url);
        std::remove(tmp.string().c_str());
        return std::nullopt;
    }
    bool ok = extract_tar(tmp, cache);
    std::remove(tmp.string().c_str());
    if (ok && fs::is_directory(out, ec)) return out;
    logger()->error("tarball extracted but {} is missing", out.string());
    return std::nullopt;
}

}  // namespace

std::string hf_url(const std::string& repo, const std::string& rel_path,
                   const std::string& rev) {
    return "https://huggingface.co/" + repo + "/resolve/" + rev + "/" + rel_path;
}

fs::path cache_root() {
    if (auto c = env("WHISPERX_SHERPA_CACHE")) return fs::path(*c);
    if (auto h = env("HOME")) return fs::path(*h) / ".cache" / "whisperx-sherpa";
    return fs::temp_directory_path() / "whisperx-sherpa";
}

std::string map_model(const std::string& whisper_arch) {
    // SHERPA_MODEL_MAP (asr_sherpa.py:56). Unknown names pass through verbatim.
    static const std::map<std::string, std::string> kMap = {
        {"tiny", "tiny"},        {"tiny.en", "tiny.en"},
        {"base", "base"},        {"base.en", "base.en"},
        {"small", "small"},      {"small.en", "small.en"},
        {"medium", "medium"},    {"medium.en", "medium.en"},
        {"large", "large-v3"},   {"large-v1", "large-v1"},
        {"large-v2", "large-v2"},{"large-v3", "large-v3"},
        {"large-v3-turbo", "large-v3-turbo"}, {"turbo", "large-v3-turbo"}};
    auto it = kMap.find(whisper_arch);
    return it == kMap.end() ? whisper_arch : it->second;
}

std::optional<fs::path> ensure_whisper_dir(const std::string& model_name) {
    const std::string name = map_model(model_name);

    // 1. Our sha-pinned mirror: meta.json names the three assets, fetched into the
    //    repo-cache dir alongside it (so the dir holds encoder/decoder/tokens).
    if (auto meta_p = fetch_file(WHISPER_REPO, name + "/meta.json")) {
        try {
            json m = json::parse(std::ifstream(*meta_p));
            bool all = true;
            for (const char* k : {"encoder", "decoder", "tokens"}) {
                std::string f = m.at(k).get<std::string>();
                if (!fetch_file(WHISPER_REPO, name + "/" + f)) {
                    all = false;
                    break;
                }
            }
            if (all) {
                logger()->info("Whisper '{}' resolved from mirror {}", name,
                               WHISPER_REPO);
                return meta_p->parent_path();
            }
        } catch (const std::exception& e) {
            logger()->warn("mirror meta.json parse failed for '{}': {}", name,
                           e.what());
        }
    }

    // 2. sherpa-onnx official release tarball.
    logger()->info("Whisper '{}' not on mirror; trying sherpa release", name);
    std::string url = std::string(SHERPA_WHISPER_RELEASE) + name + ".tar.bz2";
    return fetch_release_tarball(url, "sherpa-onnx-whisper-" + name);
}

std::optional<fs::path> ensure_align_dir(const std::string& language) {
    if (!fetch_file(ALIGN_REPO, language + "/meta.json")) return std::nullopt;
    auto onnx = fetch_file(ALIGN_REPO, language + "/model.onnx");
    if (!onnx) return std::nullopt;
    logger()->info("Align '{}' resolved from mirror {}", language, ALIGN_REPO);
    return onnx->parent_path();
}

std::optional<whisperx::server::models::DiarizeAssets> ensure_diarize() {
    using whisperx::server::models::DiarizeAssets;

    // 1. Mirror: meta.json names segmentation + embedding.
    if (auto meta_p = fetch_file(DIARIZE_REPO, "meta.json")) {
        try {
            json m = json::parse(std::ifstream(*meta_p));
            auto seg = fetch_file(DIARIZE_REPO, m.at("segmentation"));
            auto emb = fetch_file(DIARIZE_REPO, m.at("embedding"));
            if (seg && emb) {
                logger()->info("Diarize assets resolved from mirror {}",
                               DIARIZE_REPO);
                return DiarizeAssets{seg->string(), emb->string()};
            }
        } catch (const std::exception& e) {
            logger()->warn("diarize mirror meta.json parse failed: {}", e.what());
        }
    }

    // 2. sherpa-onnx release: segmentation tarball + embedding file.
    logger()->info("Diarize not on mirror; trying sherpa release");
    auto segdir = fetch_release_tarball(
        SEG_RELEASE, "sherpa-onnx-pyannote-segmentation-3-0");
    if (!segdir) return std::nullopt;

    std::error_code ec;
    fs::path seg = *segdir / "model.onnx";
    if (!fs::exists(seg, ec)) {  // some tarballs ship a different inner name
        for (const auto& e : fs::directory_iterator(*segdir, ec)) {
            if (e.is_regular_file() && e.path().extension() == ".onnx") {
                seg = e.path();
                break;
            }
        }
    }
    if (!fs::exists(seg, ec)) return std::nullopt;

    fs::path embed = cache_root() / "wespeaker_en_voxceleb_CAM++.onnx";
    if (!fs::exists(embed, ec)) {
        logger()->info("Downloading sherpa embedding: {}", EMBED_RELEASE);
        long st = net::download_to_file(EMBED_RELEASE, embed.string());
        if (st < 200 || st >= 300) return std::nullopt;
    }
    return DiarizeAssets{seg.string(), embed.string()};
}

}  // namespace whisperx::server::assets
