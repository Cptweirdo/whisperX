// LinkFlow — the interactive consent half of the OAuth link, the native analog of
// app/backup/oauth.py::_run_consent. It runs on its own detached thread (never
// the transcription JobQueue worker): generate PKCE verifier/challenge + state,
// open the user's browser at the authorize URL, then block on a promise the
// HTTP /oauth/callback route fulfils (state-validated), up to a 300 s timeout.
// On the returned code it calls OAuthClient::exchange_code and reports the
// outcome via a completion callback (the BackupService publishes it over SSE).
//
// The loopback redirect is the *main* oat++ server (redirect_uri points at its
// /oauth/callback) — no second server — so the route just calls on_callback().
#pragma once

#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>

#include "oauth/client.hpp"

namespace whisperx::server::oauth {

class LinkFlow {
public:
    // ok=false carries a human-readable reason.
    using DoneFn = std::function<void(bool ok, const std::string& error)>;

    explicit LinkFlow(OAuthClient& client) : client_(client) {}

    // Begin consent on a detached thread. No-op (returns false) if already
    // running. `on_done` fires once, from the flow thread, when it settles.
    bool start(DoneFn on_done);

    bool in_progress() const;

    // Called by the HTTP callback route with the query params. Validates state
    // and wakes the waiting flow thread. Safe to call when no flow is pending.
    void on_callback(const std::string& code, const std::string& state,
                     const std::string& error);

    // Static success/error pages served by the callback route.
    static const char* success_html();
    static const char* error_html();

private:
    struct CallbackData {
        std::string code;
        std::string error;
    };

    void run(DoneFn on_done);

    OAuthClient& client_;
    mutable std::mutex mu_;
    bool running_ = false;
    std::string expected_state_;
    std::shared_ptr<std::promise<CallbackData>> promise_;
};

// Open `url` in the user's default browser (best-effort, non-blocking).
void open_in_browser(const std::string& url);

}  // namespace whisperx::server::oauth
