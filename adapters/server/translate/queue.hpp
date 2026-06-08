// On-demand transcript translation — the C++ port of app/translate_job.py.
//
// Translation is network-bound (an external API call), so it runs on its own
// single-worker executor rather than the CPU transcription JobQueue, and won't
// block transcription. Each run is non-destructive: it reads the session's
// *current* (possibly edited) segments and writes a per-language overlay
// (transcript.translation.<lang>.json) via SessionStore; the original transcript
// is never mutated. Status is recorded on the session row's `translations` column
// (durable, for reconnects) and published to SSE subscribers on the
// `translate:<session_id>` channel.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

namespace whisperx::db {
class SessionStore;
}
namespace whisperx::server::sse {
class Broker;
}

namespace whisperx::server::translate {

// SSE channel name carrying a session's translation progress.
std::string channel(const std::string& session_id);

class TranslationQueue {
public:
    TranslationQueue(whisperx::db::SessionStore& store,
                     whisperx::server::sse::Broker* broker = nullptr);
    ~TranslationQueue();

    TranslationQueue(const TranslationQueue&) = delete;
    TranslationQueue& operator=(const TranslationQueue&) = delete;

    // Enqueue a translation of `session_id` into `target_lang` using `service`.
    // The API key is resolved here (submit time), matching the Python oracle.
    void submit(const std::string& session_id, const std::string& target_lang,
                const std::string& service);
    void shutdown();

private:
    struct Job {
        std::string session_id;
        std::string target_lang;
        std::string service;
        std::string api_key;
    };

    void worker_loop();
    void run_one(const Job& job);

    whisperx::db::SessionStore& store_;
    whisperx::server::sse::Broker* broker_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Job> queue_;
    bool stop_ = false;
    std::thread worker_;
};

}  // namespace whisperx::server::translate
