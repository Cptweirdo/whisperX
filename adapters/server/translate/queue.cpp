#include "translate/queue.hpp"

#include <nlohmann/json.hpp>

#include "db/session_store.hpp"
#include "log/log.hpp"
#include "secrets/keyring.hpp"
#include "sse/broker.hpp"
#include "time_iso.hpp"
#include "translate/google.hpp"
#include "translate/overlay.hpp"

namespace whisperx::server::translate {

using nlohmann::json;

std::string channel(const std::string& session_id) {
    return "translate:" + session_id;
}

namespace {

json event(const std::string& lang, const std::string& status,
           const std::string& error = "") {
    json e = {{"lang", lang}, {"status", status}};
    if (!error.empty()) e["error"] = error;
    return e;
}

}  // namespace

TranslationQueue::TranslationQueue(whisperx::db::SessionStore& store,
                                   whisperx::server::sse::Broker* broker)
    : store_(store), broker_(broker) {
    worker_ = std::thread([this] { worker_loop(); });
}

TranslationQueue::~TranslationQueue() { shutdown(); }

void TranslationQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stop_) return;
        stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void TranslationQueue::submit(const std::string& session_id,
                              const std::string& target_lang,
                              const std::string& service) {
    Job job{session_id, target_lang, service,
            secrets::resolve_google_api_key().value_or("")};
    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back(std::move(job));
    }
    cv_.notify_one();
}

void TranslationQueue::worker_loop() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty()) return;
            job = std::move(queue_.front());
            queue_.pop_front();
        }
        run_one(job);
    }
}

void TranslationQueue::run_one(const Job& job) {
    auto logger = log::get("translate");
    auto publish = [&](const json& e) {
        if (broker_) broker_->publish(channel(job.session_id), e);
    };

    store_.set_translation_status(job.session_id, job.target_lang,
                                  whisperx::db::Status::Running, job.service,
                                  std::nullopt);
    publish(event(job.target_lang, "running"));
    try {
        json result = store_.load_result(job.session_id);
        json orig = (result.is_object() && result.contains("segments"))
                        ? result["segments"]
                        : json::array();
        json segments = store_.current_segments(job.session_id, orig);

        std::vector<std::string> texts;
        texts.reserve(segments.size());
        for (const auto& s : segments)
            texts.push_back(s.value("text", std::string()));

        // Only "google" is registered today; an unknown service is a hard error
        // (mirrors get_translator raising TranslationError).
        if (job.service != "google")
            throw TranslationError("Unknown translation service: " + job.service);
        std::vector<std::string> translated =
            google_translate(texts, job.target_lang, job.api_key);

        json payload = {
            {"version", 2},
            {"target_language", job.target_lang},
            {"service", job.service},
            {"created_at", whisperx::now_iso()},
            {"entries", build_entries(segments, translated)},
        };
        store_.save_translation(job.session_id, job.target_lang, payload);
    } catch (const std::exception& exc) {
        logger->error("Translation of {} -> {} failed: {}", job.session_id,
                      job.target_lang, exc.what());
        store_.set_translation_status(job.session_id, job.target_lang,
                                      whisperx::db::Status::Error, std::nullopt,
                                      std::string(exc.what()));
        publish(event(job.target_lang, "error", exc.what()));
        return;
    }
    store_.set_translation_status(job.session_id, job.target_lang,
                                  whisperx::db::Status::Done, std::nullopt,
                                  std::nullopt);
    publish(event(job.target_lang, "done"));
}

}  // namespace whisperx::server::translate
