// Runtime configuration for the native server — the C++ port of the env/.env
// handling in app/server.py (the _load_dotenv precedence chain) + app/paths.py
// (data_dir) + the WHISPERX_* knobs the MVP host needs.
//
// Precedence (highest first), matching server.py:42-53 — a .env load never
// overrides an already-set key:
//   1. real environment variables
//   2. <exe_dir>/.env           (dev / back-compat; analog of app/.env)
//   3. <data_dir>/.env          (user / per-machine overrides)
//   4. <exe_dir>/defaults.env   (ship-with-the-app defaults; lowest priority)
#pragma once

#include <optional>
#include <string>

namespace whisperx::server {

// Load the .env chain into the process environment (real env always wins). Call
// once at startup before reading any config. `exe_dir` is where the binary lives
// (for the dev/defaults .env files); pass argv[0]'s directory.
void load_dotenv_chain(const std::string& exe_dir);

// Platform-aware writable data root (port of app/paths.py::data_dir):
//   WHISPERX_DATA_DIR > macOS ~/Library/Application Support/WhisperX > ./data
std::string data_dir();

// --- typed env readers (read after load_dotenv_chain) --------------------
std::string env_str(const char* key, const std::string& def = "");
std::optional<std::string> env_opt(const char* key);
long env_long(const char* key, long def);
double env_double(const char* key, double def);

struct Config {
    std::string data_dir;        // writable root for db + sessions + models
    std::string host = "127.0.0.1";
    int port = 8000;             // WHISPERX_PORT
    long max_upload_mb = 5000;   // WHISPERX_MAX_UPLOAD_MB (server.py:87)
    double max_audio_hours = 4;  // WHISPERX_MAX_AUDIO_HOURS (server.py:93)
    long batch_size = 8;         // WHISPERX_BATCH_SIZE (pipeline.py:58)
    long long_audio_warn_s = 2 * 3600;  // WHISPERX_LONG_AUDIO_WARN_S (pipeline.py:211)
    std::string log_level = "info";     // WHISPERX_LOG_LEVEL
    std::string spa_dir;         // app/static/spa (built SPA); WHISPERX_SPA_DIR override
    std::string static_dir;      // app/static (parent of spa); WHISPERX_STATIC_DIR
    std::string active_model;    // seed (persisted by the store); WHISPERX_MODEL
};

// Build the Config from the (already-loaded) environment.
Config load_config();

}  // namespace whisperx::server
