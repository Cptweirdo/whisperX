#include "oauth/backup_service.hpp"

#include "log/log.hpp"
#include "oauth/provider.hpp"
#include "sse/broker.hpp"

namespace whisperx::server {

using nlohmann::json;

BackupService::BackupService(const Config& cfg, sse::Broker& broker)
    : broker_(broker) {
    if (!oauth::google_client_configured()) {
        log::get("oauth")->info(
            "Google client credentials not set — backup link disabled.");
        return;
    }
    try {
        client_ = std::make_unique<oauth::OAuthClient>(
            oauth::google_drive_provider(cfg.port), store_);
        flow_ = std::make_unique<oauth::LinkFlow>(*client_);
    } catch (const std::exception& e) {
        log::get("oauth")->warn("Backup link unavailable: {}", e.what());
        client_.reset();
        flow_.reset();
    }
}

bool BackupService::is_linked() { return client_ && client_->is_linked(); }

std::optional<std::string> BackupService::bearer_header() {
    if (!client_) return std::nullopt;
    return client_->bearer_header();
}

std::optional<drive::DriveClient> BackupService::drive() {
    if (!client_) return std::nullopt;
    return drive::DriveClient([this] { return bearer_header(); });
}

json BackupService::status_json() {
    bool linked = is_linked();
    bool connecting = flow_ && flow_->in_progress();
    std::string state = connecting ? "connecting" : (linked ? "linked" : "idle");
    std::string err;
    {
        std::lock_guard<std::mutex> lk(mu_);
        err = last_error_;
    }
    return {
        {"state", state},
        {"linked", linked},
        {"backend", linked ? json("gdrive") : json(nullptr)},
        {"dirty", nullptr},
        {"last_root", nullptr},
        {"last_backup_at", nullptr},
        {"last_error", err.empty() ? json(nullptr) : json(err)},
        {"interval", nullptr},
        {"provider_label", "Google Drive"},
        {"last_human", nullptr},
        {"folder", nullptr},
        {"remote", nullptr},
        {"configured", configured()},
    };
}

void BackupService::publish_status() {
    broker_.publish(kBackupChannel, status_json());
}

bool BackupService::connect() {
    if (!flow_) return false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        last_error_.clear();
    }
    bool started = flow_->start([this](bool ok, const std::string& error) {
        if (!ok) {
            std::lock_guard<std::mutex> lk(mu_);
            last_error_ = error;
        }
        publish_status();  // terminal: linked or idle (with last_error)
    });
    if (started) publish_status();  // connecting
    return started;
}

void BackupService::disconnect() {
    if (client_) client_->unlink();
    {
        std::lock_guard<std::mutex> lk(mu_);
        last_error_.clear();
    }
    publish_status();
}

void BackupService::handle_callback(const std::string& code,
                                    const std::string& state,
                                    const std::string& error) {
    if (flow_) flow_->on_callback(code, state, error);
}

}  // namespace whisperx::server
