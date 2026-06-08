#include "jobs/jobs.hpp"

#include <nlohmann/json.hpp>

#include "db/session_store.hpp"
#include "log/log.hpp"
#include "sse/broker.hpp"

using nlohmann::json;

namespace whisperx::server::jobs {

JobQueue::JobQueue(whisperx::db::SessionStore& store, RunSession run_session,
                   whisperx::server::sse::Broker* broker)
    : store_(store),
      run_session_(std::move(run_session)),
      broker_(broker),
      worker_([this] { worker_loop(); }) {}

JobQueue::~JobQueue() { shutdown(); }

void JobQueue::submit(const std::string& session_id) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stop_) return;
        cancel_flags_[session_id] = std::make_shared<std::atomic<bool>>(false);
        queue_.push_back(session_id);
    }
    cv_.notify_one();
}

void JobQueue::cancel(const std::string& session_id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = cancel_flags_.find(session_id);
    if (it != cancel_flags_.end()) it->second->store(true);
}

void JobQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stop_) return;
        stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void JobQueue::worker_loop() {
    for (;;) {
        std::string id;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty()) return;
            id = std::move(queue_.front());
            queue_.pop_front();
        }
        run_one(id);
    }
}

void JobQueue::run_one(const std::string& session_id) {
    auto logger = whisperx::server::log::get("jobs");
    CancelFlag cancel;
    {
        std::lock_guard<std::mutex> lk(mu_);
        cancel = cancel_flags_[session_id];
    }

    store_.mark_running(session_id);
    bool ok = false;
    bool cancelled = false;
    try {
        run_session_(session_id, cancel);
        ok = true;
    } catch (const Cancelled&) {
        // The delete endpoint already removed the row — don't mark_error.
        cancelled = true;
        logger->info("Session {} cancelled", session_id);
    } catch (const std::exception& exc) {
        logger->error("Session {} failed: {}", session_id, exc.what());
        store_.mark_error(session_id, exc.what());
    } catch (...) {
        logger->error("Session {} failed: unknown error", session_id);
        store_.mark_error(session_id, "unknown error");
    }

    // Terminal SSE *after* the store row is updated (consistent read for a
    // client reacting to the event). Cancelled jobs publish nothing.
    if (broker_ != nullptr && !cancelled)
        broker_->publish(session_id,
                         json{{"status", ok ? "done" : "error"}});

    std::lock_guard<std::mutex> lk(mu_);
    cancel_flags_.erase(session_id);
}

}  // namespace whisperx::server::jobs
