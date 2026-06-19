// Single-worker background queue driving the transcription pipeline — the C++
// port of app/jobs.py::JobQueue.
//
// The SessionStore is the source of truth for status/results; this queue only
// runs work on a serialized worker (one at a time, so CPU isn't oversubscribed
// and uploads queue) and records the lifecycle transitions. When a broker is
// given, terminal status transitions are published to SSE subscribers *after*
// the store row is updated (so a client reacting to the event reads a consistent
// row — api-reference.md §SSE: "DB row is the durable source of truth").
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace whisperx::db {
class SessionStore;
}

namespace whisperx::server::sse {
class Broker;
}

namespace whisperx::server::jobs {

// Raised inside the worker (via the stage callback) when a cancel() lands — the
// job unwinds at the next stage boundary. Mirrors app.jobs.Cancelled.
class Cancelled : public std::exception {
public:
    const char* what() const noexcept override { return "job cancelled"; }
};

// A shared cancel flag handed to the run_session callback; the pipeline's stage
// callback throws Cancelled when it is set (stage-boundary cancellation — jobs
// run to completion otherwise, per the migration memory plan).
using CancelFlag = std::shared_ptr<std::atomic<bool>>;

// run_session(session_id, cancel): execute the pipeline and persist artifacts.
// It must call store.mark_done(...) itself (it has the result metadata); the
// queue handles mark_running and error capture.
using RunSession = std::function<void(const std::string&, const CancelFlag&)>;

class JobQueue {
public:
    JobQueue(whisperx::db::SessionStore& store, RunSession run_session,
             whisperx::server::sse::Broker* broker = nullptr);
    ~JobQueue();

    JobQueue(const JobQueue&) = delete;
    JobQueue& operator=(const JobQueue&) = delete;

    // Enqueue a session for processing.
    void submit(const std::string& session_id);
    // Signal a queued/running job to stop at the next stage boundary.
    void cancel(const std::string& session_id);
    // Stop the worker (called on shutdown); a job in flight is left to the next
    // boot's reconcile_startup to flip to error.
    void shutdown();

private:
    void worker_loop();
    void run_one(const std::string& session_id);

    whisperx::db::SessionStore& store_;
    RunSession run_session_;
    whisperx::server::sse::Broker* broker_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::string> queue_;
    std::map<std::string, CancelFlag> cancel_flags_;
    bool stop_ = false;
    std::thread worker_;
};

}  // namespace whisperx::server::jobs
