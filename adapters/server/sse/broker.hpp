// Server-Sent Events broker — the C++ port of app/sse.py::Broker.
//
// The pipeline runs on background threads; browsers watch a channel via an
// EventSource. The broker bridges the two: a worker publish()es JSON events
// keyed by channel, each open SSE request drains a subscribe()d queue. Purely
// in-memory — the SQLite row stays the durable source of truth (used for the
// initial state on connect and after a reload), the broker only carries live
// deltas (CLAUDE.md, api-reference.md §SSE).
//
// Transport-agnostic (no oatpp dep) so it is unit-testable in isolation; the
// oatpp streaming response that consumes a Subscription lives in sse_response.*.
#pragma once

#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <string>

#include <nlohmann/json.hpp>

namespace whisperx::server::sse {

using nlohmann::json;

// One subscriber's bounded event queue. push() drops on full (a slow/dead client
// must never block the worker — sse.py's queue.Full: pass). pop() blocks up to a
// timeout so the reader can emit a keepalive frame and re-check.
class Subscription {
public:
    explicit Subscription(std::size_t max_size = 64) : max_size_(max_size) {}

    // Worker side: enqueue an event, dropping it if the queue is full.
    void push(json event);

    // Reader side: wait up to `timeout` for an event. Returns the event, or
    // std::nullopt on timeout (-> caller emits a keepalive) or when closed+empty.
    std::optional<json> pop(std::chrono::milliseconds timeout);

    // Wake any blocked pop() and stop accepting (used on broker shutdown).
    void close();

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<json> q_;
    std::size_t max_size_;
    bool closed_ = false;
};

class Broker {
public:
    // Subscribe to a channel; the returned Subscription must be passed back to
    // unsubscribe() when the reader leaves (the SSE response does this in its
    // finally/RAII path).
    std::shared_ptr<Subscription> subscribe(const std::string& channel);
    void unsubscribe(const std::string& channel,
                     const std::shared_ptr<Subscription>& sub);

    // Publish an event to every current subscriber of a channel.
    void publish(const std::string& channel, json event);

    // Count of live subscribers on a channel (test/introspection).
    std::size_t subscriber_count(const std::string& channel);

private:
    std::mutex mu_;
    std::map<std::string, std::set<std::shared_ptr<Subscription>>> subs_;
};

}  // namespace whisperx::server::sse
