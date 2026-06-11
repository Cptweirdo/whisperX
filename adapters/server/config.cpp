#include "config.hpp"

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>

namespace fs = std::filesystem;

namespace whisperx::server {

namespace {

// Port of server.py::_load_dotenv: KEY=VALUE pairs, real env wins (we never
// overwrite an already-set key). Strips an unquoted inline `# comment`, and
// surrounding single/double quotes.
void load_dotenv(const fs::path& path) {
    std::error_code ec;
    if (!fs::exists(path, ec)) return;
    std::ifstream in(path);
    if (!in) return;
    std::string raw;
    while (std::getline(in, raw)) {
        // trim leading/trailing whitespace
        auto b = raw.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        auto e = raw.find_last_not_of(" \t\r\n");
        std::string line = raw.substr(b, e - b + 1);
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        // trim key
        auto kb = key.find_first_not_of(" \t");
        auto ke = key.find_last_not_of(" \t");
        if (kb == std::string::npos) continue;
        key = key.substr(kb, ke - kb + 1);
        std::string value = line.substr(eq + 1);
        auto vb = value.find_first_not_of(" \t");
        value = (vb == std::string::npos) ? "" : value.substr(vb);
        if (!value.empty() && value[0] != '"' && value[0] != '\'') {
            // strip unquoted inline comment (whitespace + #)
            static const std::regex inline_comment(R"(\s+#.*$)");
            value = std::regex_replace(value, inline_comment, "");
        }
        // trim trailing ws then surrounding quotes
        auto ve = value.find_last_not_of(" \t");
        if (ve != std::string::npos) value = value.substr(0, ve + 1);
        if (value.size() >= 2 &&
            ((value.front() == '"' && value.back() == '"') ||
             (value.front() == '\'' && value.back() == '\'')))
            value = value.substr(1, value.size() - 2);
        if (key.empty()) continue;
        // real env wins: only set if unset (setenv overwrite=0)
#if defined(_WIN32)
        if (std::getenv(key.c_str()) == nullptr)
            ::_putenv_s(key.c_str(), value.c_str());
#else
        ::setenv(key.c_str(), value.c_str(), /*overwrite=*/0);
#endif
    }
}

}  // namespace

void load_dotenv_chain(const std::string& exe_dir) {
    fs::path exe(exe_dir);
    load_dotenv(exe / ".env");
    load_dotenv(fs::path(data_dir()) / ".env");
    load_dotenv(exe / "defaults.env");
}

std::string data_dir() {
    if (const char* override = std::getenv("WHISPERX_DATA_DIR"))
        if (override[0]) return std::string(override);
#if defined(__APPLE__)
    if (const char* home = std::getenv("HOME"))
        return std::string(home) +
               "/Library/Application Support/WhisperX";
#endif
    // Historical default: ./data relative to the working dir (dev). The packaged
    // app always sets WHISPERX_DATA_DIR, so this is the dev fallback only.
    return (fs::current_path() / "data").string();
}

std::string env_str(const char* key, const std::string& def) {
    const char* v = std::getenv(key);
    return (v && v[0]) ? std::string(v) : def;
}

std::optional<std::string> env_opt(const char* key) {
    const char* v = std::getenv(key);
    if (v && v[0]) return std::string(v);
    return std::nullopt;
}

long env_long(const char* key, long def) {
    const char* v = std::getenv(key);
    if (!v || !v[0]) return def;
    try {
        return std::stol(v);
    } catch (...) {
        return def;
    }
}

double env_double(const char* key, double def) {
    const char* v = std::getenv(key);
    if (!v || !v[0]) return def;
    try {
        return std::stod(v);
    } catch (...) {
        return def;
    }
}

std::optional<Device> parse_device(const std::string& s) {
    std::string v;
    v.reserve(s.size());
    for (char ch : s) v.push_back(static_cast<char>(std::tolower(ch)));
    if (v == "cpu") return Device::Cpu;
    if (v == "cuda") return Device::Cuda;
    if (v == "coreml") return Device::CoreML;
    return std::nullopt;
}

const char* to_string(Device d) {
    switch (d) {
        case Device::Cuda:   return "cuda";
        case Device::CoreML: return "coreml";
        case Device::Cpu:    return "cpu";
    }
    return "cpu";
}

std::optional<Precision> parse_precision(const std::string& s) {
    std::string v;
    v.reserve(s.size());
    for (char ch : s) v.push_back(static_cast<char>(std::tolower(ch)));
    if (v == "fp16") return Precision::Fp16;
    if (v == "fp32") return Precision::Fp32;
    if (v == "int8") return Precision::Int8;
    return std::nullopt;
}

const char* to_string(Precision p) {
    switch (p) {
        case Precision::Fp16: return "fp16";
        case Precision::Fp32: return "fp32";
        case Precision::Int8: return "int8";
    }
    return "fp16";
}

std::optional<AsrBackend> parse_asr_backend(const std::string& s) {
    std::string v;
    v.reserve(s.size());
    for (char ch : s) v.push_back(static_cast<char>(std::tolower(ch)));
    if (v == "sherpa" || v == "sherpa-onnx") return AsrBackend::Sherpa;
    if (v == "whispercpp" || v == "whisper.cpp" || v == "whisper-cpp")
        return AsrBackend::WhisperCpp;
    return std::nullopt;
}

const char* to_string(AsrBackend b) {
    switch (b) {
        case AsrBackend::Sherpa:     return "sherpa";
        case AsrBackend::WhisperCpp: return "whispercpp";
    }
    return "sherpa";
}

// "1" | "true" | "yes" | "on" (case-insensitive) → true; everything else false.
static bool parse_bool(const std::string& s, bool def) {
    if (s.empty()) return def;
    std::string v;
    for (char ch : s) v.push_back(static_cast<char>(std::tolower(ch)));
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

Config load_config() {
    Config c;
    c.data_dir = data_dir();
    c.host = env_str("WHISPERX_HOST", "127.0.0.1");
    c.port = static_cast<int>(env_long("WHISPERX_PORT", 8000));
    c.max_upload_mb = env_long("WHISPERX_MAX_UPLOAD_MB", 5000);
    c.max_audio_hours = env_double("WHISPERX_MAX_AUDIO_HOURS", 4);
    c.batch_size = env_long("WHISPERX_BATCH_SIZE", 8);
    c.asr_batch_size = env_long("WHISPERX_ASR_BATCH_SIZE", 1);
    c.asr_precision = parse_precision(env_str("WHISPERX_ASR_PRECISION", "fp16"))
                          .value_or(Precision::Fp16);
    // Platform defaults for the Stage-1 engine + compute device.
    //
    // Apple Silicon: the measured-best Stage 1 is whisper.cpp/Metal with q8_0 +
    // flash-attention (SPEEDUP_FINDINGS.md lever 1 / docs/MACOS_COREML.md Phase 2b:
    // 1.26× Stage 1, 1.1% Russian WER drift). whisper.cpp runs on Metal, so the
    // Device knob is moot — default device is cpu.
    //
    // Windows / Linux: sherpa-onnx with the CUDA execution provider when a GPU is
    // present — so default device is cuda. A box without a usable CUDA GPU degrades
    // to cpu in the ModelManager ctor (device_available gate), mirroring how the
    // whisper.cpp backend degrades to sherpa when the build lacks it.
    //
    // An explicit env var, or a persisted /api/{device,asr_backend} choice
    // (main.cpp), still wins over these defaults.
#if defined(__APPLE__)
    constexpr const char* kDefaultBackend = "whispercpp";
    constexpr const char* kDefaultDevice = "cpu";
    constexpr const char* kDefaultQuant = "q8_0";
    constexpr const char* kDefaultFlash = "1";
#else
    constexpr const char* kDefaultBackend = "sherpa";
    constexpr const char* kDefaultDevice = "cuda";
    constexpr const char* kDefaultQuant = "";
    constexpr const char* kDefaultFlash = "";
#endif
    c.asr_backend = parse_asr_backend(env_str("WHISPERX_ASR_BACKEND", kDefaultBackend))
                        .value_or(AsrBackend::Sherpa);
    c.ggml_quant = env_str("WHISPERX_GGML_QUANT", kDefaultQuant);
    c.whispercpp_flash_attn =
        parse_bool(env_str("WHISPERX_WHISPERCPP_FLASH_ATTN", kDefaultFlash), false);
    c.long_audio_warn_s = env_long("WHISPERX_LONG_AUDIO_WARN_S", 2 * 3600);
    c.log_level = env_str("WHISPERX_LOG_LEVEL", "info");
    c.active_model = env_str("WHISPERX_MODEL", "small");
    c.device = parse_device(env_str("WHISPERX_DEVICE", kDefaultDevice)).value_or(Device::Cpu);
    c.diarize_threshold = env_double("WHISPERX_DIARIZE_THRESHOLD", c.diarize_threshold);
    c.diarize_min_on = env_double("WHISPERX_DIARIZE_MIN_ON", c.diarize_min_on);
    c.diarize_min_off = env_double("WHISPERX_DIARIZE_MIN_OFF", c.diarize_min_off);
    c.diarize_merge_threshold =
        env_double("WHISPERX_DIARIZE_MERGE_THRESHOLD", c.diarize_merge_threshold);
    // Built SPA default: <data_dir>/../? No — the SPA ships beside the binary or
    // at app/static/spa in dev. Allow an explicit override; default resolved by
    // the caller (main) relative to the exe.
    c.spa_dir = env_str("WHISPERX_SPA_DIR", "");
    c.static_dir = env_str("WHISPERX_STATIC_DIR", "");
    c.backup_backend = env_str("WHISPERX_BACKUP_BACKEND", "");
    c.backup_dir = env_str("WHISPERX_BACKUP_DIR", "");
    c.backup_interval = env_long("WHISPERX_BACKUP_INTERVAL", 900);
    return c;
}

}  // namespace whisperx::server
