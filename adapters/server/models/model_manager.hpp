// ModelManager — the C++ port of app/pipeline.py::ModelManager, sherpa-onnx based.
//
// Holds the resident native engines: a WhisperSherpa per selected checkpoint, one
// shared SherpaDiarizer, and a per-language wav2vec2 align cache. All engines are
// built for the manager's current Device (cpu | cuda); the provider is threaded into
// every constructor. A runtime device switch (set_device, Slice 4) evicts and
// rebuilds the caches on the new device, which is why the borrows handed to a job are
// shared_ptrs — an in-flight job keeps its engine alive past the eviction.
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
#include "config.hpp"  // Device
#include "diarize/diarize_sherpa.hpp"

namespace whisperx::server::models {

using nlohmann::json;

// The selectable Whisper checkpoints (WhisperModel enum in pipeline.py).
const std::vector<std::string>& whisper_model_names();
bool is_known_model(const std::string& name);

// A resolved align model + its tokenizer dictionary (align_run inputs).
// Self-owning: holds shared_ptrs so an in-flight job keeps its engine + dictionary
// alive past a device switch that evicts the manager's cache (UAF guard).
struct AlignHandle {
    std::shared_ptr<whisperx::align::Wav2Vec2Onnx> model;
    std::shared_ptr<const std::map<std::string, int>> dictionary;
    bool batchable;
};

class ModelManager {
public:
    // on_change fires with status() after any load/active/diarize transition, so
    // the server can push live model state over /models/events.
    using OnChange = std::function<void(const json&)>;

    ModelManager(std::string active, Device device, OnChange on_change = nullptr);

    std::string active();
    json status();

    // Load (or return cached) the WhisperSherpa for `name`; blocks while loading.
    // Throws std::runtime_error if the assets can't be resolved / the model fails.
    // Returns a shared_ptr so a job holding it survives a device-switch eviction.
    std::shared_ptr<whisperx::asr::WhisperSherpa> load_asr(const std::string& name);
    // Load `name` in a background thread (non-blocking) — boot/active warm.
    void warm(const std::string& name);
    // Set the active model and warm it. Returns status().
    json set_active(const std::string& name);

    // Switch the execution device at runtime. Atomically rebuilds the active model
    // on `dev` BEFORE evicting the old caches, so a GPU init failure rolls back
    // cleanly (the manager stays usable on the previous device) — throws on failure.
    // align/diarize caches are evicted and rebuilt lazily on next use. Callers must
    // ensure no job is running (the /api/device 409 busy-gate); in-flight borrows are
    // shared_ptrs so a racing job stays safe regardless. Returns status().
    json set_device(Device dev);

    // The shared diarizer, loaded once from local assets, or nullptr if none are
    // available (transcribe + align only — mirrors ensure_diarize returning None).
    std::shared_ptr<whisperx::diarize::SherpaDiarizer> ensure_diarize();

    // Resolve+cache the align model for a language. Throws if unavailable.
    AlignHandle align_for(const std::string& language);

    std::string silero_path() const;

private:
    void notify_change();
    json status_locked();
    // Whether the CUDA device can be selected (GPU build + a usable CUDA device).
    // Cached probe via whisperx::align::ort_cuda_available(); false in a CPU build.
    static bool cuda_available();
    // Resolve assets + construct a WhisperSherpa for `name` on `dev`. No cache
    // mutation — shared by load_asr and set_device's atomic rebuild. Throws on failure.
    std::shared_ptr<whisperx::asr::WhisperSherpa> build_asr_engine(
        const std::string& name, Device dev);

    struct AlignEntry {
        std::shared_ptr<whisperx::align::Wav2Vec2Onnx> model;
        std::shared_ptr<const std::map<std::string, int>> dictionary;
        bool batchable = false;
    };

    std::mutex lock_;       // guards the maps / active / status
    std::mutex load_lock_;  // held only during a heavy model load
    std::map<std::string, std::shared_ptr<whisperx::asr::WhisperSherpa>> asr_;
    std::set<std::string> loading_;
    std::map<std::string, std::string> errors_;
    std::shared_ptr<whisperx::diarize::SherpaDiarizer> diarize_;
    bool diarize_loaded_ = false;
    std::optional<std::string> diarize_error_;
    std::map<std::string, AlignEntry> align_;
    std::string active_;
    Device device_ = Device::Cpu;  // guarded by lock_; mutated only by set_device
    OnChange on_change_;
};

}  // namespace whisperx::server::models
