#include "backup/engine.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "log/log.hpp"

namespace fs = std::filesystem;

namespace whisperx::server::backup {

using nlohmann::json;

namespace {

std::string now_iso_utc() {
    std::time_t t = std::time(nullptr);
    std::tm tm {};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S+00:00", &tm);
    return buf;
}

auto logger() { return log::get("backup"); }

}  // namespace

std::string to_string(BackupState s) {
    switch (s) {
        case BackupState::Disabled: return "disabled";
        case BackupState::Unlinked: return "unlinked";
        case BackupState::Idle: return "idle";
        case BackupState::Dirty: return "dirty";
        case BackupState::Conflict: return "conflict";
        case BackupState::BackingUp: return "backing_up";
        case BackupState::Restoring: return "restoring";
        case BackupState::Error: return "error";
    }
    return "idle";
}

BackupEngine::BackupEngine(whisperx::db::SessionStore& store,
                           std::string data_dir,
                           std::unique_ptr<StorageBackend> backend, int interval,
                           bool gc, OnChange on_change)
    : store_(store),
      data_dir_(std::move(data_dir)),
      backend_(std::move(backend)),
      interval_(interval),
      gc_(gc),
      on_change_(std::move(on_change)) {
    snapshot_dir_ = (fs::path(data_dir_) / ".backup").string();
    state_path_ = (fs::path(snapshot_dir_) / "state.json").string();
    load_state();
}

BackupEngine::~BackupEngine() { stop(); }

// --- notify + per-device state persistence ---------------------------------
void BackupEngine::notify() {
    if (!on_change_) return;
    try {
        on_change_();
    } catch (...) {
        logger()->warn("backup on_change callback failed");
    }
}

void BackupEngine::load_state() {
    std::ifstream f(state_path_, std::ios::binary);
    if (!f) return;
    try {
        json d = json::parse(f);
        if (d.contains("last_signature") && d["last_signature"].is_string())
            last_signature_ = d["last_signature"].get<std::string>();
        if (d.contains("last_root") && d["last_root"].is_string())
            last_root_ = d["last_root"].get<std::string>();
        if (d.contains("last_backup_at") && d["last_backup_at"].is_string())
            last_backup_at_ = d["last_backup_at"].get<std::string>();
    } catch (...) {
        // tolerate a missing/corrupt sidecar
    }
}

void BackupEngine::save_state() {
    std::error_code ec;
    fs::create_directories(snapshot_dir_, ec);
    json payload = {
        {"last_signature",
         last_signature_ ? json(*last_signature_) : json(nullptr)},
        {"last_root", last_root_ ? json(*last_root_) : json(nullptr)},
        {"last_backup_at",
         last_backup_at_ ? json(*last_backup_at_) : json(nullptr)},
    };
    std::string tmp = state_path_ + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            logger()->warn("could not persist backup state to {}", state_path_);
            return;
        }
        f << payload.dump();
    }
    fs::rename(tmp, state_path_, ec);
}

// --- status ----------------------------------------------------------------
bool BackupEngine::is_linked() { return backend_ && backend_->is_linked(); }

bool BackupEngine::is_dirty() {
    if (!last_signature_) return true;
    return cheap_signature(data_dir_) != *last_signature_;
}

BackupState BackupEngine::state() {
    if (!backend_) return BackupState::Disabled;
    if (!is_linked()) return BackupState::Unlinked;
    BackupState a;
    {
        std::lock_guard<std::mutex> lk(activity_lock_);
        a = activity_;
    }
    if (a == BackupState::BackingUp || a == BackupState::Restoring ||
        a == BackupState::Error)
        return a;
    if (awaiting_decision_) return BackupState::Conflict;
    return is_dirty() ? BackupState::Dirty : BackupState::Idle;
}

void BackupEngine::set_activity(BackupState a, const std::string& error) {
    {
        std::lock_guard<std::mutex> lk(activity_lock_);
        activity_ = a;
        last_error_ = (a == BackupState::Error) ? error : std::string();
    }
    notify();
}

void BackupEngine::set_awaiting_decision(bool v) {
    if (awaiting_decision_.exchange(v) != v) notify();
}

json BackupEngine::status() {
    bool linked = is_linked();
    std::string err;
    {
        std::lock_guard<std::mutex> lk(activity_lock_);
        err = last_error_;
    }
    return {
        {"state", to_string(state())},
        {"linked", linked},
        {"backend", backend_ ? json(backend_->name()) : json(nullptr)},
        {"dirty", linked ? json(is_dirty()) : json(nullptr)},
        {"last_root", last_root_ ? json(*last_root_) : json(nullptr)},
        {"last_backup_at",
         last_backup_at_ ? json(*last_backup_at_) : json(nullptr)},
        {"last_error", err.empty() ? json(nullptr) : json(err)},
        {"interval", interval_},
    };
}

// --- core: push ------------------------------------------------------------
BackupResult BackupEngine::backup_now(std::optional<bool> gc) {
    if (!is_linked()) throw std::runtime_error("backup backend is not linked");
    bool do_gc = gc ? *gc : gc_;
    std::lock_guard<std::mutex> lk(sync_lock_);
    set_activity(BackupState::BackingUp);
    try {
        BackupResult r = do_backup(do_gc);
        set_activity(BackupState::Idle);
        return r;
    } catch (const std::exception& e) {
        set_activity(BackupState::Error, e.what());
        throw;
    }
}

BackupResult BackupEngine::do_backup(bool gc) {
    std::error_code ec;
    fs::create_directories(snapshot_dir_, ec);
    std::string snapshot_path = (fs::path(snapshot_dir_) / "snapshot.db").string();

    // 1. consistent DB copy (holds the store write lock)
    store_.snapshot_db(snapshot_path);

    // 2. capture a cheap signature BEFORE hashing so a change that lands during
    //    this run leaves us dirty for the next pass (no missed updates).
    std::string signature = cheap_signature(data_dir_);

    std::optional<Manifest> remote = backend_->read_manifest();
    int generation = remote ? remote->generation + 1 : 1;

    // 3. build local manifest from the snapshot + artifact tree
    Manifest local = build_local_manifest(snapshot_path, data_dir_, generation);

    // 4. stream up blobs whose content is new remotely
    int uploaded = 0, skipped = 0;
    for (const std::string& path : changed_paths(local, remote)) {
        const std::string& key = local.entries[path].hash;
        if (backend_->has_object(key)) {
            skipped++;
            continue;
        }
        std::string src = (path == "sessions.db")
                              ? snapshot_path
                              : (fs::path(data_dir_) / path).string();
        if (!fs::exists(src, ec)) {
            // File deleted mid-run (e.g. session removed); drop it so we don't
            // commit a dangling reference. Next backup reconciles the rest.
            logger()->warn("skip vanished file during backup: {}", path);
            local.entries.erase(path);
            continue;
        }
        backend_->put_object(key, src);
        uploaded++;
    }

    // 5. commit: write the manifest last (atomic) — this is when it "lands"
    backend_->write_manifest(local);

    // 6. optional GC of blobs no longer referenced
    if (gc) gc_orphans(local);

    std::string root = merkle_root(local);
    last_signature_ = signature;
    last_root_ = root;
    last_backup_at_ = now_iso_utc();
    save_state();
    fs::remove(snapshot_path, ec);
    logger()->info("backup done: gen={} uploaded={} skipped={} root={}",
                   generation, uploaded, skipped, root.substr(0, 12));
    return BackupResult{uploaded, skipped, generation, root};
}

void BackupEngine::gc_orphans(const Manifest& manifest) {
    std::set<std::string> referenced = manifest.object_keys();
    for (const std::string& key : backend_->list_objects())
        if (!referenced.count(key)) backend_->delete_object(key);
}

// --- core: restore (manual) ------------------------------------------------
int BackupEngine::restore(bool prune) {
    if (!is_linked()) throw std::runtime_error("backup backend is not linked");
    if (store_.has_active_jobs())
        throw std::runtime_error(
            "cannot restore while transcription jobs are active");
    std::lock_guard<std::mutex> lk(sync_lock_);
    set_activity(BackupState::Restoring);
    try {
        int n = do_restore(prune);
        set_activity(BackupState::Idle);
        return n;
    } catch (const std::exception& e) {
        set_activity(BackupState::Error, e.what());
        throw;
    }
}

int BackupEngine::do_restore(bool prune) {
    std::optional<Manifest> remote = backend_->read_manifest();
    if (!remote) throw std::runtime_error("remote has no backup to restore");

    int restored = 0;
    for (const auto& [path, entry] : remote->entries) {
        if (path == "sessions.db") continue;  // swapped last
        std::string dest = (fs::path(data_dir_) / path).string();
        backend_->get_object_to_file(entry.hash, dest);  // streamed + atomic
        restored++;
    }

    if (prune) prune_local(*remote);

    // DB last: stage then atomic swap + reopen.
    auto it = remote->entries.find("sessions.db");
    if (it != remote->entries.end()) {
        std::error_code ec;
        fs::create_directories(snapshot_dir_, ec);
        std::string staged = (fs::path(snapshot_dir_) / "restore.db").string();
        backend_->get_object_to_file(it->second.hash, staged);
        store_.swap_db(staged);
        restored++;
    }

    last_signature_ = cheap_signature(data_dir_);
    last_root_ = merkle_root(*remote);
    if (!remote->created_at.empty()) last_backup_at_ = remote->created_at;
    save_state();
    logger()->info("restore done: gen={} files={}", remote->generation,
                   restored);
    return restored;
}

void BackupEngine::prune_local(const Manifest& remote) {
    std::set<std::string> keep;
    for (const auto& [path, _] : remote.entries) keep.insert(path);

    fs::path sessions_root = fs::path(data_dir_) / "sessions";
    std::error_code ec;
    if (!fs::is_directory(sessions_root, ec)) return;

    // delete files not present in the remote manifest
    std::vector<fs::path> dirs;
    for (auto it = fs::recursive_directory_iterator(
             sessions_root, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (it->is_directory(ec)) {
            dirs.push_back(it->path());
            continue;
        }
        if (!it->is_regular_file(ec)) continue;
        std::string logical =
            fs::relative(it->path(), data_dir_, ec).generic_string();
        if (!keep.count(logical)) fs::remove(it->path(), ec);
    }
    // remove now-empty subdirectories (deepest first)
    std::sort(dirs.begin(), dirs.end(), [](const fs::path& a, const fs::path& b) {
        return a.string().size() > b.string().size();
    });
    for (const fs::path& d : dirs)
        if (fs::is_empty(d, ec)) fs::remove(d, ec);
}

// --- bootstrap on link -----------------------------------------------------
RemoteState BackupEngine::bootstrap() {
    if (!is_linked()) throw std::runtime_error("backup backend is not linked");
    std::optional<Manifest> remote = backend_->probe();
    if (!remote) return RemoteState{};
    RemoteState rs;
    rs.exists = true;
    rs.generation = remote->generation;
    rs.entries = static_cast<int>(remote->entries.size());
    long long total = 0;
    for (const auto& [_, e] : remote->entries) total += e.size;
    rs.total_size = total;
    rs.created_at = remote->created_at;
    return rs;
}

LinkAssessment BackupEngine::assess_link() {
    if (!is_linked()) throw std::runtime_error("backup backend is not linked");
    std::optional<Manifest> remote_m = backend_->probe();
    if (!remote_m) return LinkAssessment{LinkOutcome::Fresh, RemoteState{}};

    RemoteState remote;
    remote.exists = true;
    remote.generation = remote_m->generation;
    remote.entries = static_cast<int>(remote_m->entries.size());
    long long total = 0;
    for (const auto& [_, e] : remote_m->entries) total += e.size;
    remote.total_size = total;
    remote.created_at = remote_m->created_at;

    if (store_.list().empty())  // local has no sessions -> nothing to lose
        return LinkAssessment{LinkOutcome::RemoteOnly, remote};

    std::string lroot = local_root();
    if (lroot == merkle_root(*remote_m)) {
        last_root_ = lroot;
        last_signature_ = cheap_signature(data_dir_);
        if (!remote_m->created_at.empty()) last_backup_at_ = remote_m->created_at;
        save_state();
        set_awaiting_decision(false);
        return LinkAssessment{LinkOutcome::InSync, remote};
    }
    // Real conflict: pause periodic auto-backup until the user picks a side.
    set_awaiting_decision(true);
    return LinkAssessment{LinkOutcome::Diverged, remote};
}

std::string BackupEngine::local_root() {
    std::error_code ec;
    fs::create_directories(snapshot_dir_, ec);
    std::string snap = (fs::path(snapshot_dir_) / "assess.db").string();
    store_.snapshot_db(snap);
    Manifest local = build_local_manifest(snap, data_dir_);
    fs::remove(snap, ec);
    return merkle_root(local);
}

int BackupEngine::adopt_remote() {
    set_awaiting_decision(false);
    return restore();
}

BackupResult BackupEngine::overwrite_remote() {
    set_awaiting_decision(false);
    return backup_now(true);
}

// --- periodic trigger ------------------------------------------------------
void BackupEngine::start_periodic() {
    if (interval_ <= 0 || thread_.joinable()) return;
    thread_ = std::thread([this] { periodic_loop(); });
}

void BackupEngine::stop() {
    {
        std::lock_guard<std::mutex> lk(stop_mu_);
        stop_ = true;
    }
    stop_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void BackupEngine::periodic_loop() {
    for (;;) {
        std::unique_lock<std::mutex> lk(stop_mu_);
        stop_cv_.wait_for(lk, std::chrono::seconds(interval_),
                          [this] { return stop_; });
        if (stop_) break;
        lk.unlock();
        try {
            if (is_linked() && !awaiting_decision_ && is_dirty()) backup_now();
        } catch (...) {
            logger()->warn("periodic backup failed");
        }
    }
}

}  // namespace whisperx::server::backup
