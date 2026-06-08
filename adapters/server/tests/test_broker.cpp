// Catch2 tests for the SSE broker (port of app/sse.py) — pub/sub fan-out,
// drop-on-full, pop timeout (keepalive trigger), close, and unsubscribe.
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <thread>

#include "sse/broker.hpp"

using namespace std::chrono_literals;
using whisperx::server::sse::Broker;
using whisperx::server::sse::Subscription;
using nlohmann::json;

TEST_CASE("publish fans out to all subscribers", "[broker]") {
    Broker b;
    auto a = b.subscribe("s1");
    auto c = b.subscribe("s1");
    REQUIRE(b.subscriber_count("s1") == 2);

    b.publish("s1", json{{"stage", "decoding"}});
    auto ea = a->pop(10ms);
    auto ec = c->pop(10ms);
    REQUIRE(ea.has_value());
    REQUIRE(ec.has_value());
    REQUIRE((*ea)["stage"] == "decoding");
    REQUIRE((*ec)["stage"] == "decoding");
}

TEST_CASE("publish to a channel with no subscribers is a no-op", "[broker]") {
    Broker b;
    REQUIRE_NOTHROW(b.publish("empty", json{{"x", 1}}));
    REQUIRE(b.subscriber_count("empty") == 0);
}

TEST_CASE("unsubscribe stops delivery and cleans up the channel", "[broker]") {
    Broker b;
    auto a = b.subscribe("s1");
    b.unsubscribe("s1", a);
    REQUIRE(b.subscriber_count("s1") == 0);
    b.publish("s1", json{{"x", 1}});
    REQUIRE_FALSE(a->pop(10ms).has_value());
}

TEST_CASE("pop times out when no event (keepalive trigger)", "[broker]") {
    Subscription sub;
    auto t0 = std::chrono::steady_clock::now();
    auto ev = sub.pop(30ms);
    auto dt = std::chrono::steady_clock::now() - t0;
    REQUIRE_FALSE(ev.has_value());
    REQUIRE(dt >= 25ms);
}

TEST_CASE("queue drops events when full (slow client never blocks worker)",
          "[broker]") {
    Subscription sub(/*max_size=*/4);
    for (int i = 0; i < 10; ++i) sub.push(json{{"i", i}});
    int drained = 0;
    while (sub.pop(0ms).has_value()) ++drained;
    REQUIRE(drained == 4);  // capped at max_size, extras dropped
}

TEST_CASE("close wakes a blocked pop and drains remaining", "[broker]") {
    auto sub = std::make_shared<Subscription>();
    sub->push(json{{"a", 1}});
    sub->close();
    REQUIRE(sub->pop(0ms).has_value());        // drains the queued event
    REQUIRE_FALSE(sub->pop(0ms).has_value());  // then empty+closed

    auto sub2 = std::make_shared<Subscription>();
    std::thread closer([&] {
        std::this_thread::sleep_for(20ms);
        sub2->close();
    });
    auto ev = sub2->pop(5s);  // returns promptly on close, not after 5s
    closer.join();
    REQUIRE_FALSE(ev.has_value());
}

TEST_CASE("FIFO ordering within a subscription", "[broker]") {
    Broker b;
    auto a = b.subscribe("s1");
    b.publish("s1", json{{"n", 1}});
    b.publish("s1", json{{"n", 2}});
    b.publish("s1", json{{"n", 3}});
    REQUIRE((*a->pop(10ms))["n"] == 1);
    REQUIRE((*a->pop(10ms))["n"] == 2);
    REQUIRE((*a->pop(10ms))["n"] == 3);
}
