#include "sse/sse_response.hpp"

#include <cstring>

#include "oatpp/core/IODefinitions.hpp"
#include "oatpp/core/async/Coroutine.hpp"
#include "oatpp/core/data/stream/Stream.hpp"
#include "oatpp/web/protocol/http/outgoing/StreamingBody.hpp"

namespace http = oatpp::web::protocol::http;

namespace whisperx::server::sse {

namespace {

std::string sse_format(const json& event) {
    return "data: " + event.dump() + "\n\n";
}

// A ReadCallback that drains a broker Subscription into the SSE wire format,
// mirroring sse.py::sse_response's stream() generator (pending -> initial ->
// loop with keepalive). Unsubscribes on destruction (the finally: clause).
class SseReadCallback : public oatpp::data::stream::ReadCallback {
public:
    SseReadCallback(Broker& broker, std::string channel, InitialFn initial,
                    TerminalFn terminal, PendingFn pending,
                    std::chrono::milliseconds keepalive)
        : broker_(broker),
          channel_(std::move(channel)),
          sub_(broker.subscribe(channel_)),
          initial_(std::move(initial)),
          terminal_(std::move(terminal)),
          pending_(std::move(pending)),
          keepalive_(keepalive) {}

    ~SseReadCallback() override { broker_.unsubscribe(channel_, sub_); }

    oatpp::v_io_size read(void* buffer, v_buff_size count,
                          oatpp::async::Action& /*action*/) override {
        for (;;) {
            if (pos_ < buf_.size()) {
                auto n = std::min<std::size_t>(
                    static_cast<std::size_t>(count), buf_.size() - pos_);
                std::memcpy(buffer, buf_.data() + pos_, n);
                pos_ += n;
                return static_cast<oatpp::v_io_size>(n);
            }
            if (done_) return 0;  // EOF — connection closes
            if (last_frame_) {    // last frame fully served
                done_ = true;
                return 0;
            }
            produce_next();  // refills buf_ (may be empty -> loop again)
        }
    }

private:
    void set_frame(std::string s, bool last) {
        buf_ = std::move(s);
        pos_ = 0;
        last_frame_ = last;
    }

    void produce_next() {
        switch (phase_) {
            case Phase::Pending:
                phase_ = Phase::Initial;
                if (pending_) {
                    if (auto p = pending_(); p) {
                        set_frame(sse_format(*p), /*last=*/true);
                        return;
                    }
                }
                buf_.clear();
                pos_ = 0;
                return;
            case Phase::Initial:
                phase_ = Phase::Loop;
                if (initial_) {
                    if (auto f = initial_(); f) {
                        bool last = terminal_ && terminal_(*f);
                        set_frame(sse_format(*f), last);
                        return;
                    }
                }
                buf_.clear();
                pos_ = 0;
                return;
            case Phase::Loop:
            default: {
                auto ev = sub_->pop(keepalive_);
                if (!ev) {
                    set_frame(": keepalive\n\n", /*last=*/false);
                    return;
                }
                bool last = terminal_ && terminal_(*ev);
                set_frame(sse_format(*ev), last);
                return;
            }
        }
    }

    enum class Phase { Pending, Initial, Loop };

    Broker& broker_;
    std::string channel_;
    std::shared_ptr<Subscription> sub_;
    InitialFn initial_;
    TerminalFn terminal_;
    PendingFn pending_;
    std::chrono::milliseconds keepalive_;

    Phase phase_ = Phase::Pending;
    std::string buf_;
    std::size_t pos_ = 0;
    bool last_frame_ = false;
    bool done_ = false;
};

}  // namespace

std::shared_ptr<http::outgoing::Response> sse_response(
    Broker& broker, const std::string& channel, InitialFn initial,
    TerminalFn terminal, PendingFn pending,
    std::chrono::milliseconds keepalive) {
    auto cb = std::make_shared<SseReadCallback>(broker, channel,
                                                std::move(initial),
                                                std::move(terminal),
                                                std::move(pending), keepalive);
    auto body = std::make_shared<http::outgoing::StreamingBody>(cb);
    auto response = http::outgoing::Response::createShared(
        http::Status::CODE_200, body);
    response->putHeader("Content-Type", "text/event-stream");
    response->putHeader("Cache-Control", "no-cache");
    response->putHeader("X-Accel-Buffering", "no");
    response->putHeader("Connection", "close");
    return response;
}

}  // namespace whisperx::server::sse
