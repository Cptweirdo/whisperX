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

ModelManager::ModelManager(std::string active, OnChange on_change)
    : active_(is_known_model(active) ? std::move(active) : std::string("small")),
      on_change_(std::move(on_change)) {}

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
        {"device", "cpu"},
        {"cuda_available", false},
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

wa::WhisperSherpa& ModelManager::load_asr(const std::string& name) {
    if (!is_known_model(name))
        throw std::runtime_error("Unknown model: " + name);
    {
        std::lock_guard<std::mutex> lk(lock_);
        if (auto it = asr_.find(name); it != asr_.end()) return *it->second;
    }
    std::lock_guard<std::mutex> load(load_lock_);
    {
        std::lock_guard<std::mutex> lk(lock_);
        if (auto it = asr_.find(name); it != asr_.end()) return *it->second;
        loading_.insert(name);
    }
    notify_change();  // broadcast "loading"

    auto logger = whisperx::server::log::get("models");
    try {
        auto assets = resolve_whisper(name);
        if (!assets)
            throw std::runtime_error(
                "Whisper assets for '" + name +
                "' not found locally (set WHISPERX_SHERPA_MODELS_ROOT / "
                "WHISPERX_SHERPA_WHISPER_DIR; the downloader lands in task 7).");
        logger->info("Loading whisper model={} on cpu (feature_dim={})", name,
                     assets->feature_dim);
        auto pipe = std::make_unique<wa::WhisperSherpa>(
            assets->encoder, assets->decoder, assets->tokens, /*num_threads=*/1,
            assets->feature_dim);
        wa::WhisperSherpa& ref = *pipe;
        {
            std::lock_guard<std::mutex> lk(lock_);
            asr_[name] = std::move(pipe);
            loading_.erase(name);
            errors_.erase(name);
        }
        notify_change();  // broadcast "ready"
        return ref;
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

wd::SherpaDiarizer* ModelManager::ensure_diarize() {
    {
        std::lock_guard<std::mutex> lk(lock_);
        if (diarize_loaded_) return diarize_.get();
    }
    std::lock_guard<std::mutex> load(load_lock_);
    {
        std::lock_guard<std::mutex> lk(lock_);
        if (diarize_loaded_) return diarize_.get();
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
        logger->info("Loading diarization (seg={}, embed={})", assets->segmentation,
                     assets->embedding);
        auto d = std::make_unique<wd::SherpaDiarizer>(assets->segmentation,
                                                      assets->embedding);
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
    return diarize_.get();
}

AlignHandle ModelManager::align_for(const std::string& language) {
    {
        std::lock_guard<std::mutex> lk(lock_);
        if (auto it = align_.find(language); it != align_.end())
            return {it->second.model.get(), &it->second.dictionary,
                    it->second.batchable};
    }
    std::lock_guard<std::mutex> load(load_lock_);
    {
        std::lock_guard<std::mutex> lk(lock_);
        if (auto it = align_.find(language); it != align_.end())
            return {it->second.model.get(), &it->second.dictionary,
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
    entry.model = std::make_unique<wal::Wav2Vec2Onnx>(assets->onnx_path);
    entry.dictionary = std::move(assets->dictionary);
    entry.batchable = assets->batchable;
    std::lock_guard<std::mutex> lk(lock_);
    auto& slot = align_[language];
    slot = std::move(entry);
    return {slot.model.get(), &slot.dictionary, slot.batchable};
}

std::string ModelManager::silero_path() const {
    return whisperx::server::models::silero_path();
}

}  // namespace whisperx::server::models
