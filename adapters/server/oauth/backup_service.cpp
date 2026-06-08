#include "oauth/backup_service.hpp"

#include <thread>
#include <utility>

#include "backup/engine.hpp"
#include "backup/gdrive_backend.hpp"
#include "backup/local_backend.hpp"
#include "db/session_store.hpp"
#include "log/log.hpp"
#include "oauth/provider.hpp"
#include "secrets/keyring.hpp"
#include "sse/broker.hpp"

namespace whisperx::server {

using nlohmann::json;

namespace {
auto logger() { return log::get("backup"); }
}  // namespace

BackupService::BackupService(const Config& cfg, sse::Broker& broker,
                             whisperx::db::SessionStore& store)
    : cfg_(cfg), broker_(broker) {
    // OAuth link (Google) — only when client credentials are configured.
    if (oauth::google_client_configured()) {
        try {
            client_ = std::make_unique<oauth::OAuthClient>(
                oauth::google_drive_provider(cfg.port), token_store_);
            flow_ = std::make_unique<oauth::LinkFlow>(*client_);
        } catch (const std::exception& e) {
            logger()->warn("Backup link unavailable: {}", e.what());
            client_.reset();
            flow_.reset();
        }
    } else {
        logger()->info(
            "Google client credentials not set — Drive backup link disabled.");
    }
    build_backend(cfg, store);
}

BackupService::~BackupService() = default;

void BackupService::build_backend(const Config& cfg,
                                  whisperx::db::SessionStore& store) {
    std::unique_ptr<backup::StorageBackend> be;
    if (cfg.backup_backend == "local" && !cfg.backup_dir.empty()) {
        be = std::make_unique<backup::LocalFsBackend>(cfg.backup_dir);
    } else if (client_) {
        folder_ = secrets::get(secrets::keys::GDRIVE_FOLDER)
                      .value_or("Manuscript Backup");
        auto g = std::make_unique<backup::GDriveBackend>(
            [this] { return drive(); }, folder_);
        gdrive_ = g.get();  // non-owning; engine_ owns it after the move
        be = std::move(g);
    }
    if (!be) return;  // unconfigured — endpoints degrade gracefully
    engine_ = std::make_unique<backup::BackupEngine>(
        store, cfg.data_dir, std::move(be), static_cast<int>(cfg.backup_interval),
        /*gc=*/true, [this] { publish_status(); });
    engine_->start_periodic();  // no-op when interval <= 0; loop guards on linked
}

bool BackupService::configured() const { return static_cast<bool>(engine_); }

bool BackupService::is_linked() { return engine_ && engine_->is_linked(); }

std::optional<std::string> BackupService::bearer_header() {
    if (!client_) return std::nullopt;
    return client_->bearer_header();
}

std::optional<drive::DriveClient> BackupService::drive() {
    if (!client_ || !client_->is_linked()) return std::nullopt;
    return drive::DriveClient([this] {
        auto t = client_->access_token();
        if (!t || t->empty()) return std::optional<std::string>();
        return std::optional<std::string>("Bearer " + *t);
    });
}

json BackupService::status_json() {
    json card;
    if (engine_) {
        card = engine_->status();
    } else {
        card = {{"state", "disabled"},  {"linked", false},
                {"backend", nullptr},   {"dirty", nullptr},
                {"last_root", nullptr}, {"last_backup_at", nullptr},
                {"last_error", nullptr}, {"interval", nullptr}};
    }
    bool linked = engine_ && engine_->is_linked();
    bool connecting = !linked && flow_ && flow_->in_progress();
    if (connecting) card["state"] = "connecting";
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!last_error_.empty()) card["last_error"] = last_error_;
    }
    // SPA-card extras (mirror app/server.py::_backup_json).
    card["configured"] = configured();
    card["provider_label"] = "Google Drive";
    card["folder"] = folder_.empty() ? json(nullptr) : json(folder_);
    card["last_human"] = nullptr;
    card["remote"] = nullptr;
    return card;
}

void BackupService::publish_connect(const std::string& phase,
                                    const std::string& message) {
    json e = {{"status", phase}, {"backup", status_json()}};
    if (!message.empty()) e["message"] = message;
    broker_.publish(kBackupChannel, e);
}

void BackupService::publish_status() {
    broker_.publish(kBackupStatusChannel, json{{"status", status_json()}});
}

void BackupService::set_folder(const std::string& name) {
    std::string n = name;
    size_t b = n.find_first_not_of(" \t\r\n");
    size_t e = n.find_last_not_of(" \t\r\n");
    n = (b == std::string::npos) ? std::string() : n.substr(b, e - b + 1);
    if (n.empty() || !gdrive_) return;
    folder_ = n;
    try {
        secrets::set(secrets::keys::GDRIVE_FOLDER, n);
    } catch (const std::exception& ex) {
        logger()->warn("could not persist backup folder: {}", ex.what());
    }
    gdrive_->set_folder(n);
}

bool BackupService::connect() {
    if (!flow_) return false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        last_error_.clear();
    }
    bool started = flow_->start(
        [this](bool ok, const std::string& error) { on_link_done(ok, error); });
    if (started) publish_connect("connecting");
    return started;
}

void BackupService::on_link_done(bool ok, const std::string& error) {
    if (!ok) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            last_error_ = error;
        }
        publish_connect("error", error);
        return;
    }
    // Linked: classify the remote and act (seed empty / adopt empty-local), then
    // start the periodic loop. Runs on the detached consent thread.
    if (engine_) {
        try {
            auto a = engine_->assess_link();
            if (a.outcome == backup::LinkOutcome::Fresh)
                engine_->backup_now();
            else if (a.outcome == backup::LinkOutcome::RemoteOnly)
                engine_->restore();
            // InSync / Diverged: nothing automatic (Diverged awaits the user).
            engine_->start_periodic();
        } catch (const std::exception& e) {
            logger()->warn("post-link bootstrap failed: {}", e.what());
            std::lock_guard<std::mutex> lk(mu_);
            last_error_ = e.what();
        }
    }
    publish_connect("linked");
    publish_status();
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

bool BackupService::start_backup_now() {
    if (!engine_ || !engine_->is_linked()) return false;
    backup::BackupEngine* e = engine_.get();
    std::thread([e] {
        try {
            e->backup_now();
        } catch (const std::exception& ex) {
            logger()->warn("manual backup failed: {}", ex.what());
        }
    }).detach();
    return true;
}

int BackupService::restore() {
    if (!engine_) throw std::runtime_error("Cloud backup isn't configured.");
    return engine_->restore();
}

int BackupService::adopt() {
    if (!engine_) throw std::runtime_error("Cloud backup isn't configured.");
    return engine_->adopt_remote();
}

json BackupService::overwrite() {
    if (!engine_) throw std::runtime_error("Cloud backup isn't configured.");
    backup::BackupResult r = engine_->overwrite_remote();
    return {{"uploaded", r.uploaded}, {"skipped", r.skipped}};
}

json BackupService::remote_info() {
    if (!engine_) throw std::runtime_error("Cloud backup isn't configured.");
    backup::RemoteState r = engine_->bootstrap();
    return {{"remote",
             {{"exists", r.exists},
              {"generation", r.generation},
              {"entries", r.entries},
              {"total_size", r.total_size},
              {"created_at", r.created_at}}}};
}

}  // namespace whisperx::server
