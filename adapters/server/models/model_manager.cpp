#include "models/model_manager.hpp"

#include <stdexcept>
#include <thread>

#include "log/log.hpp"
#include "models/assets.hpp"

namespace wa = whisperx::asr;
namespace wal = whisperx::align;
namespace wd = whisperx::diarize;

namespace whisperx::server::models {

const std::vector<std::string>& whisper_model_names() {
    static const std::vector<std::string> names = {
        "tiny",   "tiny.en",  "base",     "base.en",
        "small",  "small.en", "medium",   "medium.en",
        "large-v2", "large-v3", "large-v3-turbo", "distil-large-v3"};
    return names;
}

bool is_known_model(const std::string& name) {
    const auto& n = whisper_model_names();
    return std::find(n.begin(), n.end(), name) != n.end();
}

namespace {
// Intra-op thread count per device. CPU: half the cores (≥1) — single-thread CPU
// left perf on the table; the old hardcoded 1 was a CPU-only-runtime artifact. GPU:
// 1 — intra-op CPU threads don't help once the forward runs on the CUDA EP.
int threads_for(Device d) {
    if (d != Device::Cpu) return 1;
    unsigned hc = std::thread::hardware_concurrency();
    return hc > 1 ? static_cast<int>(hc / 2) : 1;
}
}  // namespace

ModelManager::ModelManager(std::string active, Device device, OnChange on_change,
                           DiarizeTuning diarize_tuning)
    : active_(is_known_model(active) ? std::move(active) : std::string("small")),
      device_(device),
      on_change_(std::move(on_change)),
      diarize_tuning_(diarize_tuning) {}

std::string ModelManager::active() {
    std::lock_guard<std::mutex> lk(lock_);
    return active_;
}

void ModelManager::notify_change() {
    if (!on_change_) return;
    try {
        on_change_(status());
    } catch (...) {
        // a listener error must never break model loading (pipeline.py)
        whisperx::server::log::get("models")
            ->warn("model on_change callback failed");
    }
}

json ModelManager::status() {
    std::lock_guard<std::mutex> lk(lock_);
    return status_locked();
}

bool ModelManager::cuda_available() {
    static const bool avail = wal::ort_cuda_available();
    return avail;
}

bool ModelManager::coreml_available() {
    static const bool avail = wal::ort_coreml_available();
    return avail;
}

bool ModelManager::device_available(Device dev) {
    switch (dev) {
        case Device::Cpu:    return true;
        case Device::Cuda:   return cuda_available();
        case Device::CoreML: return coreml_available();
    }
    return false;
}

json ModelManager::status_locked() {
    bool diarize_available = resolve_diarize().has_value();
    json models = json::array();
    for (const auto& name : whisper_model_names()) {
        models.push_back({
            {"name", name},
            {"loaded", asr_.count(name) > 0},
            {"loading", loading_.count(name) > 0},
            {"error", errors_.count(name) ? json(errors_.at(name)) : json(nullptr)},
        });
    }
    return {
        {"active", active_},
        {"device", to_string(device_)},
        {"cuda_available", cuda_available()},
        {"coreml_available", coreml_available()},
        {"mlx_available", false},
        {"whispercpp_available", false},
        {"diarize", diarize_ != nullptr},
        {"diarize_error", diarize_error_ ? json(*diarize_error_) : json(nullptr)},
        {"diarize_available", diarize_available},
        {"diarize_source", diarize_available ? "local" : "none"},
        {"diarize_version", nullptr},
        {"diarize_token", false},
        {"models", models},
    };
}

std::shared_ptr<wa::WhisperSherpa> ModelManager::load_asr(const std::string& name) {
    if (!is_known_model(name))
        throw std::runtime_error("Unknown model: " + name);
    {
        std::lock_guard<std::mutex> lk(lock_);
        if (auto it = asr_.find(name); it != asr_.end()) return it->second;
    }
    std::lock_guard<std::mutex> load(load_lock_);
    {
        std::lock_guard<std::mutex> lk(lock_);
        if (auto it = asr_.find(name); it != asr_.end()) return it->second;
        loading_.insert(name);
    }
    notify_change();  // broadcast "loading"

    auto logger = whisperx::server::log::get("models");
    try {
        auto pipe = build_asr_engine(name, device_);
        {
            std::lock_guard<std::mutex> lk(lock_);
            asr_[name] = pipe;
            loading_.erase(name);
            errors_.erase(name);
        }
        notify_change();  // broadcast "ready"
        return pipe;
    } catch (const std::exception& exc) {
        {
            std::lock_guard<std::mutex> lk(lock_);
            loading_.erase(name);
            errors_[name] = exc.what();
        }
        notify_change();  // broadcast the failure
        logger->error("Whisper model={} load failed: {}", name, exc.what());
        throw;
    }
}

void ModelManager::warm(const std::string& name) {
    std::thread([this, name] {
        try {
            load_asr(name);
        } catch (...) {
            // already logged + recorded in errors_
        }
    }).detach();
}

json ModelManager::set_active(const std::string& name) {
    if (!is_known_model(name))
        throw std::runtime_error("Unknown model: " + name);
    {
        std::lock_guard<std::mutex> lk(lock_);
        active_ = name;
    }
    warm(name);
    notify_change();
    return status();
}

std::shared_ptr<wa::WhisperSherpa> ModelManager::build_asr_engine(
    const std::string& name, Device dev) {
    auto assets = resolve_whisper(name);
    if (!assets)
        throw std::runtime_error(
            "Whisper assets for '" + name +
            "' not found locally (set WHISPERX_SHERPA_MODELS_ROOT / "
            "WHISPERX_SHERPA_WHISPER_DIR; the downloader lands in task 7).");
    whisperx::server::log::get("models")->info(
        "Loading whisper model={} on {} (feature_dim={})", name, to_string(dev),
        assets->feature_dim);
    return std::make_shared<wa::WhisperSherpa>(
        assets->encoder, assets->decoder, assets->tokens, threads_for(dev),
        assets->feature_dim, /*language=*/"", /*task=*/"transcribe",
        /*provider=*/to_string(dev));
}

json ModelManager::set_device(Device dev) {
    // load_lock_ serializes against every heavy load (load_asr/align_for/ensure_
    // diarize all take it), so device_ can't be read mid-construction by another load.
    std::lock_guard<std::mutex> load(load_lock_);
    Device prev;
    std::string active_snapshot;
    {
        std::lock_guard<std::mutex> lk(lock_);
        if (dev == device_) return status_locked();  // no-op
        prev = device_;
        active_snapshot = active_;
    }
    auto logger = whisperx::server::log::get("models");
    logger->info("Switching device {} -> {}", to_string(prev), to_string(dev));

    // Atomic rebuild: construct the active model on the NEW device BEFORE evicting,
    // so a failure (GPU OOM / missing cuDNN / EP load error) leaves device_ and the
    // caches untouched — the manager stays fully usable on the previous device.
    std::shared_ptr<wa::WhisperSherpa> new_active;
    try {
        new_active = build_asr_engine(active_snapshot, dev);
    } catch (const std::exception& exc) {
        logger->error("Device switch to {} failed, kept {}: {}", to_string(dev),
                      to_string(prev), exc.what());
        throw;  // nothing mutated yet — clean rollback
    }

    {
        std::lock_guard<std::mutex> lk(lock_);
        // Every resident engine is bound to the old provider — drop them all. align +
        // diarize rebuild lazily on next use; the active ASR is pre-warmed below.
        asr_.clear();
        align_.clear();
        diarize_.reset();
        diarize_loaded_ = false;
        diarize_error_.reset();
        loading_.clear();
        errors_.clear();
        device_ = dev;
        asr_[active_snapshot] = std::move(new_active);
    }
    notify_change();  // push the new device/cuda_available to /models/events + SPA
    return status();
}

std::shared_ptr<wd::SherpaDiarizer> ModelManager::ensure_diarize() {
    {
        std::lock_guard<std::mutex> lk(lock_);
        if (diarize_loaded_) return diarize_;
    }
    std::lock_guard<std::mutex> load(load_lock_);
    {
        std::lock_guard<std::mutex> lk(lock_);
        if (diarize_loaded_) return diarize_;
    }
    auto logger = whisperx::server::log::get("models");
    auto assets = resolve_diarize();
    if (!assets) {
        logger->warn(
            "No local diarization assets (set WHISPERX_DIARIZE_ONNX_DIR): "
            "diarization disabled (transcribe + align only).");
        std::lock_guard<std::mutex> lk(lock_);
        diarize_loaded_ = true;
        notify_change();
        return nullptr;
    }
    try {
        logger->info("Loading diarization (seg={}, embed={}, threshold={}, "
                     "min_on={}, min_off={}, merge={}, provider=cpu, threads={})",
                     assets->segmentation, assets->embedding,
                     diarize_tuning_.threshold, diarize_tuning_.min_duration_on,
                     diarize_tuning_.min_duration_off,
                     diarize_tuning_.merge_threshold, threads_for(Device::Cpu));
        // Diarization always runs on CPU with CPU-count threads, regardless of
        // device_: the workload is hundreds of tiny forwards (10s segmentation
        // windows + per-chunk embeddings), where per-call CUDA launch/copy
        // overhead serializes everything onto one core. Measured on a 23-min
        // file: cuda+1thread 737s, cuda+8threads no better (single core pegged),
        // cpu+8threads 98s. Whisper keeps the device provider — its 30s windows
        // are big enough for the GPU to win.
        auto d = std::make_shared<wd::SherpaDiarizer>(
            assets->segmentation, assets->embedding, threads_for(Device::Cpu),
            /*provider=*/"cpu", diarize_tuning_.threshold,
            diarize_tuning_.min_duration_on, diarize_tuning_.min_duration_off,
            diarize_tuning_.merge_threshold);
        std::lock_guard<std::mutex> lk(lock_);
        diarize_ = std::move(d);
        diarize_loaded_ = true;
    } catch (const std::exception& exc) {
        std::lock_guard<std::mutex> lk(lock_);
        diarize_error_ = exc.what();
        diarize_loaded_ = true;
        logger->error("Diarization model load failed: {}", exc.what());
    }
    notify_change();
    std::lock_guard<std::mutex> lk(lock_);
    return diarize_;
}

AlignHandle ModelManager::align_for(const std::string& language) {
    {
        std::lock_guard<std::mutex> lk(lock_);
        if (auto it = align_.find(language); it != align_.end())
            return {it->second.model, it->second.dictionary,
                    it->second.batchable};
    }
    std::lock_guard<std::mutex> load(load_lock_);
    {
        std::lock_guard<std::mutex> lk(lock_);
        if (auto it = align_.find(language); it != align_.end())
            return {it->second.model, it->second.dictionary,
                    it->second.batchable};
    }
    auto assets = resolve_align(language);
    if (!assets)
        throw std::runtime_error(
            "Align model for language '" + language +
            "' not found locally (set WHISPERX_ALIGN_ONNX_ROOT/<lang> or "
            "WHISPERX_ALIGN_ONNX_DIR).");
    whisperx::server::log::get("models")->info(
        "Loading align model for language={}", language);
    AlignEntry entry;
    entry.model = std::make_shared<wal::Wav2Vec2Onnx>(
        assets->onnx_path, threads_for(device_), /*provider=*/to_string(device_));
    entry.dictionary = std::make_shared<const std::map<std::string, int>>(
        std::move(assets->dictionary));
    entry.batchable = assets->batchable;
    std::lock_guard<std::mutex> lk(lock_);
    auto& slot = align_[language];
    slot = std::move(entry);
    return {slot.model, slot.dictionary, slot.batchable};
}

std::string ModelManager::silero_path() const {
    return whisperx::server::models::silero_path();
}

}  // namespace whisperx::server::models
