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
#include "asr/asr_engine.hpp"     // AsrEngine (backend-neutral handle)
#include "asr/whisper_cpp.hpp"    // WhisperCpp (impl gated by WHISPERX_WHISPERCPP_BUILD)
#include "asr/whisper_sherpa.hpp"
#include "config.hpp"  // Device, AsrBackend
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

// Diarization clustering tuning, forwarded verbatim to the SherpaDiarizer ctor.
// Defaults mirror Config (config.hpp) so tests constructing a bare ModelManager
// get the production values.
struct DiarizeTuning {
    float threshold = 0.7f;     // WHISPERX_DIARIZE_THRESHOLD
    float min_duration_on = 0.3f;   // WHISPERX_DIARIZE_MIN_ON
    float min_duration_off = 0.5f;  // WHISPERX_DIARIZE_MIN_OFF
    float merge_threshold = 0.25f;  // WHISPERX_DIARIZE_MERGE_THRESHOLD
};

class ModelManager {
public:
    // on_change fires with status() after any load/active/diarize transition, so
    // the server can push live model state over /models/events.
    using OnChange = std::function<void(const json&)>;

    // asr_batch_size: VAD chunks decoded per sherpa call on the Cuda device
    // (WHISPERX_BATCH_SIZE; Cpu/CoreML always decode serially — the win is
    // GPU throughput, the cost is VRAM per row).
    // asr_precision: Whisper variant on GPU devices (WHISPERX_ASR_PRECISION);
    // Cpu always loads int8-preferred (config.hpp Precision comment).
    // asr_backend: Stage-1 engine — Sherpa (ONNX/ORT, default) or WhisperCpp (Metal).
    // ggml_quant / flash_attn apply only to the WhisperCpp backend.
    ModelManager(std::string active, Device device, OnChange on_change = nullptr,
                 DiarizeTuning diarize_tuning = {}, int asr_batch_size = 1,
                 Precision asr_precision = Precision::Fp16,
                 AsrBackend asr_backend = AsrBackend::Sherpa,
                 std::string ggml_quant = "", bool whispercpp_flash_attn = false);

    std::string active();
    json status();

    // Load (or return cached) the ASR engine for `name`; blocks while loading.
    // Throws std::runtime_error if the assets can't be resolved / the model fails.
    // Returns a shared_ptr (to the AsrEngine base — sherpa or whisper.cpp) so a job
    // holding it survives a device/backend-switch eviction.
    std::shared_ptr<whisperx::asr::AsrEngine> load_asr(const std::string& name);
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

    // Switch the Stage-1 ASR backend at runtime (sherpa ↔ whisper.cpp). Same
    // atomic rebuild-then-evict contract as set_device: the active model is rebuilt
    // on the new backend before the old caches drop, rolling back on failure. Returns
    // status(). Throws if the target backend isn't available in this build.
    json set_asr_backend(AsrBackend backend);

    // Whether `dev` can be selected in this build/on this machine: Cpu always,
    // Cuda/CoreML per the cached ORT provider probes. The single gate behind
    // /api/device and onboarding (replaces per-endpoint cuda-only checks).
    static bool device_available(Device dev);

    // Whether `backend` can be selected: Sherpa always; WhisperCpp only in a
    // WHISPERX_WHISPERCPP_BUILD. The gate behind /api/asr_backend.
    static bool asr_backend_available(AsrBackend backend);

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
    // Whether the CoreML EP is linked in (Apple builds only). Cached probe via
    // whisperx::align::ort_coreml_available(); false everywhere else.
    static bool coreml_available();
    // Whether the whisper.cpp backend is compiled in (WHISPERX_WHISPERCPP_BUILD).
    static bool whispercpp_available();
    // Resolve assets + construct the ASR engine for `name` on `dev` with the current
    // asr_backend_. No cache mutation — shared by load_asr and the set_device /
    // set_asr_backend atomic rebuilds. Throws on failure.
    std::shared_ptr<whisperx::asr::AsrEngine> build_asr_engine(
        const std::string& name, Device dev);

    struct AlignEntry {
        std::shared_ptr<whisperx::align::Wav2Vec2Onnx> model;
        std::shared_ptr<const std::map<std::string, int>> dictionary;
        bool batchable = false;
    };

    std::mutex lock_;       // guards the maps / active / status
    std::mutex load_lock_;  // held only during a heavy model load
    std::map<std::string, std::shared_ptr<whisperx::asr::AsrEngine>> asr_;
    std::set<std::string> loading_;
    std::map<std::string, std::string> errors_;
    std::shared_ptr<whisperx::diarize::SherpaDiarizer> diarize_;
    bool diarize_loaded_ = false;
    std::optional<std::string> diarize_error_;
    std::map<std::string, AlignEntry> align_;
    std::string active_;
    Device device_ = Device::Cpu;  // guarded by lock_; mutated only by set_device
    AsrBackend asr_backend_ = AsrBackend::Sherpa;  // guarded by lock_; set_asr_backend
    OnChange on_change_;
    DiarizeTuning diarize_tuning_;  // immutable after construction
    int asr_batch_size_ = 1;        // immutable after construction
    Precision asr_precision_ = Precision::Fp16;  // immutable after construction
    std::string ggml_quant_;        // immutable; whisper.cpp quant suffix
    bool whispercpp_flash_attn_ = false;  // immutable; whisper.cpp flash-attn
};

}  // namespace whisperx::server::models
