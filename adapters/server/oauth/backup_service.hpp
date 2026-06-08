// BackupService — the server-facing facade over the OAuth link, owning the
// OAuthClient + LinkFlow and publishing link progress on the backup SSE channel.
// It is the collaborator the ApiController talks to for /api/backup/* and
// /oauth/callback. Construction tolerates missing Google client credentials: if
// GOOGLE_CLIENT_ID/SECRET aren't set the service is "unconfigured" and the
// endpoints degrade gracefully (settings still render), matching how the rest of
// the MVP treats deferred/optional features.
//
// Scope is the token lifecycle only — a live bearer token for a future Drive
// (files.*) client. No upload/manifest logic lives here.
#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "config.hpp"
#include "drive/drive_client.hpp"
#include "oauth/client.hpp"
#include "oauth/flow.hpp"
#include "oauth/token_store.hpp"

namespace whisperx::server::sse {
class Broker;
}

namespace whisperx::server {

// SSE channel for backup link progress (analog of the Python /backup/events).
inline constexpr const char* kBackupChannel = "__backup__";

class BackupService {
public:
    BackupService(const Config& cfg, sse::Broker& broker);

    bool configured() const { return static_cast<bool>(client_); }
    bool is_linked();

    // backup_stub()-shaped status the SPA expects, with linked/state filled.
    nlohmann::json status_json();

    // Kick off the consent flow (no-op if unconfigured or already running).
    // Returns false if it couldn't start.
    bool connect();
    void disconnect();

    // /oauth/callback handler — forward the redirect query params to the flow.
    void handle_callback(const std::string& code, const std::string& state,
                         const std::string& error);

    // A live bearer header for outbound Drive calls, or nullopt.
    std::optional<std::string> bearer_header();

    // A Drive files.* client bound to this link's live token, or nullopt when
    // backup isn't configured. The returned client refreshes the token lazily
    // via bearer_header() on each call.
    std::optional<drive::DriveClient> drive();

private:
    void publish_status();

    sse::Broker& broker_;
    oauth::KeyringTokenStore store_;
    std::unique_ptr<oauth::OAuthClient> client_;
    std::unique_ptr<oauth::LinkFlow> flow_;
    std::mutex mu_;
    std::string last_error_;
};

}  // namespace whisperx::server
