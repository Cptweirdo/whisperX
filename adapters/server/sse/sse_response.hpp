// oat++ streaming response for an SSE broker channel — the C++ port of
// app/sse.py::sse_response. Centralises the parts every SSE endpoint must get
// right (subscribe, finally-unsubscribe, the keepalive frame, no-buffering
// headers) so a route only supplies what varies:
//
//   initial  : () -> optional<json>  — state emitted right after subscribe (so it
//              can't miss an event racing the subscribe). nullopt skips.
//   terminal : (json) -> bool        — close after this event. nullptr = persistent.
//   pending  : () -> optional<json>  — late-subscriber replay; if it returns an
//              event, emit it and close immediately (work finished pre-connect).
//
// Built on the simple (threaded) HttpConnectionHandler: each connection runs on
// its own thread, so the ReadCallback may block on the subscription queue.
#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "oatpp/web/protocol/http/outgoing/Response.hpp"

#include "sse/broker.hpp"

namespace whisperx::server::sse {

using nlohmann::json;
using InitialFn = std::function<std::optional<json>()>;
using TerminalFn = std::function<bool(const json&)>;
using PendingFn = std::function<std::optional<json>()>;

std::shared_ptr<oatpp::web::protocol::http::outgoing::Response> sse_response(
    Broker& broker, const std::string& channel, InitialFn initial = nullptr,
    TerminalFn terminal = nullptr, PendingFn pending = nullptr,
    std::chrono::milliseconds keepalive = std::chrono::milliseconds(15000));

}  // namespace whisperx::server::sse
