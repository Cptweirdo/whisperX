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

// Execution device for the inference engines. Strings live only at the I/O
// boundaries (WHISPERX_DEVICE env, /api/device JSON, the persisted setting, and
// sherpa/ORT's provider C-string); internally the device is this enum so invalid
// states are unrepresentable and validation has a single home (parse_device).
// Extensible to Mlx/WhisperCpp later (status() already reports those flags).
// CoreML is the Apple-Silicon provider swap (METAL_INTEGRATION.md route A) —
// selectable only when ort_coreml_available() says the EP is linked in.
enum class Device { Cpu, Cuda, CoreML };

// Parse a boundary string ("cpu" | "cuda" | "coreml", case-insensitive) into a
// Device. Returns nullopt for anything else — the one place device validation
// happens.
std::optional<Device> parse_device(const std::string& s);
// Canonical lowercase name — used for the /models status JSON, the persisted
// setting, AND as the sherpa/ORT provider string passed to engine ctors (they
// coincide: "cpu"/"cuda"/"coreml").
const char* to_string(Device d);

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
    Device device = Device::Cpu; // WHISPERX_DEVICE (cpu|cuda|coreml); persisted setting wins
    // Diarization clustering (sherpa FastClustering, core/diarize/diarize_sherpa.hpp).
    // threshold is a cosine *distance*: larger merges more aggressively -> fewer
    // speakers. sherpa's 0.5 default over-segments real recordings (12+ labels on a
    // 2-speaker call); 0.7 + the centroid merge below is the optimum of a sweep over
    // the golden 4-speaker dialogs + a real 2-speaker call (commit message has the
    // table). min_on drops speech bursts shorter than it; min_off merges pauses
    // shorter than it (seconds).
    double diarize_threshold = 0.7;  // WHISPERX_DIARIZE_THRESHOLD
    double diarize_min_on = 0.3;     // WHISPERX_DIARIZE_MIN_ON
    double diarize_min_off = 0.5;    // WHISPERX_DIARIZE_MIN_OFF
    // Centroid merge post-pass (core/diarize/merge_clusters.hpp): pooled
    // per-cluster embeddings closer than this re-join. Pooled-embedding distance
    // scale — true-speaker centroid pairs sit >=0.29 in the sweep, so 0.25 merges
    // fragments while keeping real speakers apart. NOT the chunk scale above.
    // 0 disables the pass.
    double diarize_merge_threshold = 0.25;  // WHISPERX_DIARIZE_MERGE_THRESHOLD
    // Cloud backup (port of app/backup/__init__.py::build_service):
    std::string backup_backend;  // WHISPERX_BACKUP_BACKEND: "gdrive" | "local" | ""
    std::string backup_dir;      // WHISPERX_BACKUP_DIR (local backend root)
    long backup_interval = 900;  // WHISPERX_BACKUP_INTERVAL secs (0 = no periodic)
};

// Build the Config from the (already-loaded) environment.
Config load_config();

}  // namespace whisperx::server
