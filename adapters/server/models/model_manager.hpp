// ModelManager — the C++ port of app/pipeline.py::ModelManager, CPU + sherpa-only.
//
// Holds the resident native engines: a WhisperSherpa per selected checkpoint
// (cached, no eviction — switching back is instant), one shared SherpaDiarizer,
// and a per-language wav2vec2 align cache. The Python ModelManager juggled four
// backends + device switching; here the runtime is sherpa-onnx on CPU only
// (faster-whisper / torch / mlx / pyannote are gone with Python), so device is
// fixed to "cpu" and the status() shape reports the others unavailable.
//
// Server-only (links whisperx_core_audio). Asset paths come from models/assets.hpp.
#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "align/wav2vec2_onnx.hpp"
#include "asr/whisper_sherpa.hpp"
#include "diarize/diarize_sherpa.hpp"

namespace whisperx::server::models {

using nlohmann::json;

// The selectable Whisper checkpoints (WhisperModel enum in pipeline.py).
const std::vector<std::string>& whisper_model_names();
bool is_known_model(const std::string& name);

// A resolved align model + its tokenizer dictionary (align_run inputs).
struct AlignHandle {
    whisperx::align::Wav2Vec2Onnx* model;
    const std::map<std::string, int>* dictionary;
    bool batchable;
};

class ModelManager {
public:
    // on_change fires with status() after any load/active/diarize transition, so
    // the server can push live model state over /models/events.
    using OnChange = std::function<void(const json&)>;

    explicit ModelManager(std::string active, OnChange on_change = nullptr);

    std::string active();
    json status();

    // Load (or return cached) the WhisperSherpa for `name`; blocks while loading.
    // Throws std::runtime_error if the assets can't be resolved / the model fails.
    whisperx::asr::WhisperSherpa& load_asr(const std::string& name);
    // Load `name` in a background thread (non-blocking) — boot/active warm.
    void warm(const std::string& name);
    // Set the active model and warm it. Returns status().
    json set_active(const std::string& name);

    // The shared diarizer, loaded once from local assets, or nullptr if none are
    // available (transcribe + align only — mirrors ensure_diarize returning None).
    whisperx::diarize::SherpaDiarizer* ensure_diarize();

    // Resolve+cache the align model for a language. Throws if unavailable.
    AlignHandle align_for(const std::string& language);

    std::string silero_path() const;

private:
    void notify_change();
    json status_locked();

    struct AlignEntry {
        std::unique_ptr<whisperx::align::Wav2Vec2Onnx> model;
        std::map<std::string, int> dictionary;
        bool batchable = false;
    };

    std::mutex lock_;       // guards the maps / active / status
    std::mutex load_lock_;  // held only during a heavy model load
    std::map<std::string, std::unique_ptr<whisperx::asr::WhisperSherpa>> asr_;
    std::set<std::string> loading_;
    std::map<std::string, std::string> errors_;
    std::unique_ptr<whisperx::diarize::SherpaDiarizer> diarize_;
    bool diarize_loaded_ = false;
    std::optional<std::string> diarize_error_;
    std::map<std::string, AlignEntry> align_;
    std::string active_;
    OnChange on_change_;
};

}  // namespace whisperx::server::models
