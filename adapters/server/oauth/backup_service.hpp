// BackupService — the server-facing facade over the cloud-backup subsystem. It
// owns the OAuth link (OAuthClient + LinkFlow) AND the backup engine
// (backup::BackupEngine over a StorageBackend), and is the collaborator the
// ApiController talks to for /api/backup/* + /oauth/callback.
//
// Construction tolerates a missing/optional backend: with no Google client creds
// (GOOGLE_CLIENT_ID/SECRET) and no local backend configured, the subsystem is
// "unconfigured" and the endpoints degrade gracefully (settings still render).
// When configured it builds a GDriveBackend bound to the live OAuth token (or a
// LocalFsBackend when WHISPERX_BACKUP_BACKEND=local) and an engine on top.
//
// Two SSE channels (matching the Python host): kBackupChannel is the one-shot
// OAuth consent result (/backup/events), kBackupStatusChannel is the persistent
// sync-status stream (/backup/status/events).
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
namespace whisperx::db {
class SessionStore;
}
namespace whisperx::server::backup {
class BackupEngine;
class GDriveBackend;
}

namespace whisperx::server {

// One-shot OAuth consent result channel (analog of the Python /backup/events).
inline constexpr const char* kBackupChannel = "__backup__";
// Persistent backup sync-status channel (analog of /backup/status/events).
inline constexpr const char* kBackupStatusChannel = "__backup_status__";

class BackupService {
public:
    BackupService(const Config& cfg, sse::Broker& broker,
                  whisperx::db::SessionStore& store);
    ~BackupService();

    // Whether the backup subsystem is active (a backend + engine exist).
    bool configured() const;
    // Whether the backend currently has usable credentials / a reachable target.
    bool is_linked();

    // The full backup-card status the SPA renders (engine state + link facts +
    // provider_label/folder/configured/...).
    nlohmann::json status_json();

    // --- OAuth link (Google Drive) --------------------------------------
    // Re-target the Drive backup folder (persists to the keyring; live on the
    // running backend). No-op for the local backend / empty name.
    void set_folder(const std::string& name);
    // Kick off the consent flow (no-op/false if unconfigured or already running).
    bool connect();
    void disconnect();
    // /oauth/callback handler — forward the redirect query params to the flow.
    void handle_callback(const std::string& code, const std::string& state,
                         const std::string& error);

    // --- backup engine actions (delegate; throw std::runtime_error on misuse) -
    bool start_backup_now();          // async push; false if not linked
    int restore();                    // returns files restored
    int adopt();                      // bootstrap "load existing"
    nlohmann::json overwrite();       // bootstrap "start fresh" -> {uploaded,skipped}
    nlohmann::json remote_info();     // {remote: {...}}

    // A live bearer header ("Authorization: Bearer …") for outbound calls.
    std::optional<std::string> bearer_header();
    // A Drive files.* client bound to this link's live token, or nullopt when not
    // linked. The client refreshes the token lazily on each call.
    std::optional<drive::DriveClient> drive();

private:
    void build_backend(const Config& cfg, whisperx::db::SessionStore& store);
    void on_link_done(bool ok, const std::string& error);
    void publish_connect(const std::string& phase,
                         const std::string& message = "");
    void publish_status();

    const Config& cfg_;
    sse::Broker& broker_;
    oauth::KeyringTokenStore token_store_;
    std::unique_ptr<oauth::OAuthClient> client_;
    std::unique_ptr<oauth::LinkFlow> flow_;
    std::unique_ptr<backup::BackupEngine> engine_;
    backup::GDriveBackend* gdrive_ = nullptr;  // non-owning (owned by engine_)
    std::string folder_;  // gdrive backup folder name (empty for local/none)
    std::mutex mu_;
    std::string last_error_;
};

}  // namespace whisperx::server
