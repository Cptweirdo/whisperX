// Catch2 tests for the JobQueue (port of app/jobs.py) against a real temp
// SessionStore — serialization, terminal-publish-after-store, cancellation.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "db/session_store.hpp"
#include "jobs/jobs.hpp"
#include "sse/broker.hpp"

using namespace std::chrono_literals;
namespace fs = std::filesystem;
using nlohmann::json;
using whisperx::db::SessionStore;
using whisperx::server::jobs::Cancelled;
using whisperx::server::jobs::CancelFlag;
using whisperx::server::jobs::JobQueue;
using whisperx::server::sse::Broker;

namespace {
fs::path temp_data_dir() {
    auto dir = fs::temp_directory_path() /
               ("wxjobs-" + std::to_string(::getpid()) + "-" +
                std::to_string(std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count()));
    fs::create_directories(dir);
    return dir;
}
}  // namespace

TEST_CASE("a completed job marks done and publishes done after the store update",
          "[jobs]") {
    auto dir = temp_data_dir();
    SessionStore store(dir.string());
    store.create("s1", "clip.mp3", "audio.mp3", json::object(),
                 std::optional<std::string>("tiny"));
    Broker broker;
    auto sub = broker.subscribe("s1");

    std::string seen_status;  // store status observed at terminal-publish time
    JobQueue q(
        store,
        [&](const std::string& id, const CancelFlag&) {
            store.mark_done(id, std::optional<std::string>("en"),
                            /*diarized=*/false, "tiny", /*num_segments=*/3,
                            /*duration=*/1.0);
        },
        &broker);

    q.submit("s1");
    auto ev = sub->pop(5s);
    REQUIRE(ev.has_value());
    REQUIRE((*ev)["status"] == "done");
    // The event fired after mark_done, so the row is already terminal.
    REQUIRE(store.get("s1")->status == whisperx::db::Status::Done);
    fs::remove_all(dir);
}

TEST_CASE("a throwing job marks error and publishes error", "[jobs]") {
    auto dir = temp_data_dir();
    SessionStore store(dir.string());
    store.create("s1", "clip.mp3", "audio.mp3", json::object(),
                 std::optional<std::string>("tiny"));
    Broker broker;
    auto sub = broker.subscribe("s1");

    JobQueue q(
        store,
        [&](const std::string&, const CancelFlag&) {
            throw std::runtime_error("boom");
        },
        &broker);
    q.submit("s1");

    auto ev = sub->pop(5s);
    REQUIRE(ev.has_value());
    REQUIRE((*ev)["status"] == "error");
    auto row = store.get("s1");
    REQUIRE(row.has_value());
    REQUIRE(row->status == whisperx::db::Status::Error);
    REQUIRE(row->error == "boom");
    fs::remove_all(dir);
}

TEST_CASE("a cancelled job publishes nothing and is not marked error",
          "[jobs]") {
    auto dir = temp_data_dir();
    SessionStore store(dir.string());
    store.create("s1", "clip.mp3", "audio.mp3", json::object(),
                 std::optional<std::string>("tiny"));
    Broker broker;
    auto sub = broker.subscribe("s1");

    JobQueue q(
        store,
        [&](const std::string&, const CancelFlag& cancel) {
            // Simulate the stage-boundary cancel check.
            for (int i = 0; i < 100; ++i) {
                if (cancel->load()) throw Cancelled();
                std::this_thread::sleep_for(2ms);
            }
        },
        &broker);
    q.submit("s1");
    std::this_thread::sleep_for(10ms);
    q.cancel("s1");

    REQUIRE_FALSE(sub->pop(500ms).has_value());  // no terminal event
    REQUIRE(store.get("s1")->status != whisperx::db::Status::Error);
    fs::remove_all(dir);
}

TEST_CASE("jobs run serially (one worker)", "[jobs]") {
    auto dir = temp_data_dir();
    SessionStore store(dir.string());
    for (auto id : {"a", "b", "c"})
        store.create(id, "clip.mp3", "audio.mp3", json::object(),
                     std::optional<std::string>("tiny"));
    Broker broker;

    std::atomic<int> concurrent{0};
    std::atomic<int> max_concurrent{0};
    std::atomic<int> done{0};
    JobQueue q(
        store,
        [&](const std::string& id, const CancelFlag&) {
            int c = ++concurrent;
            int prev = max_concurrent.load();
            while (c > prev && !max_concurrent.compare_exchange_weak(prev, c)) {
            }
            std::this_thread::sleep_for(20ms);
            --concurrent;
            store.mark_done(id, std::optional<std::string>("en"), false, "tiny",
                            1, 1.0);
            ++done;
        },
        &broker);
    q.submit("a");
    q.submit("b");
    q.submit("c");

    for (int i = 0; i < 100 && done.load() < 3; ++i)
        std::this_thread::sleep_for(20ms);
    REQUIRE(done.load() == 3);
    REQUIRE(max_concurrent.load() == 1);
    fs::remove_all(dir);
}
