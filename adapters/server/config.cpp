#include "config.hpp"

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
        ::setenv(key.c_str(), value.c_str(), /*overwrite=*/0);
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

Config load_config() {
    Config c;
    c.data_dir = data_dir();
    c.host = env_str("WHISPERX_HOST", "127.0.0.1");
    c.port = static_cast<int>(env_long("WHISPERX_PORT", 8000));
    c.max_upload_mb = env_long("WHISPERX_MAX_UPLOAD_MB", 5000);
    c.max_audio_hours = env_double("WHISPERX_MAX_AUDIO_HOURS", 4);
    c.batch_size = env_long("WHISPERX_BATCH_SIZE", 8);
    c.long_audio_warn_s = env_long("WHISPERX_LONG_AUDIO_WARN_S", 2 * 3600);
    c.log_level = env_str("WHISPERX_LOG_LEVEL", "info");
    c.active_model = env_str("WHISPERX_MODEL", "small");
    // Built SPA default: <data_dir>/../? No — the SPA ships beside the binary or
    // at app/static/spa in dev. Allow an explicit override; default resolved by
    // the caller (main) relative to the exe.
    c.spa_dir = env_str("WHISPERX_SPA_DIR", "");
    c.static_dir = env_str("WHISPERX_STATIC_DIR", "");
    return c;
}

}  // namespace whisperx::server
