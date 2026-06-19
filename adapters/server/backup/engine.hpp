// Backup orchestration — native port of app/backup/service.py.
//
// All the policy lives here; the StorageBackend just moves bytes. Design notes:
//
// * Consistency. The DB is copied via store.snapshot_db (holds the same lock
//   every mutation takes), so a backup can't capture a torn write. The network
//   upload runs after the lock is released, off the snapshot. Session artifacts
//   are immutable (audio) or written atomically (edits), so they're hashed/
//   uploaded without the store lock.
// * Single-device mirror. Local is authoritative; the remote is a one-way
//   mirror. No multi-device merge.
// * Commit point. backend.write_manifest is written last and atomically, so a
//   backup only "lands" once every referenced blob is already uploaded.
// * Streaming. Objects are 300-400 MB audio blobs: backup_now passes on-disk
//   source paths to put_object (streamed up); restore pulls each blob via
//   get_object_to_file (streamed down). No blob is held whole in memory.
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "backup/backend.hpp"
#include "db/session_store.hpp"

namespace whisperx::server::backup {

enum class BackupState {
    Disabled,    // no backend configured
    Unlinked,    // backend configured, not linked
    Idle,        // linked, in sync, nothing running
    Dirty,       // linked, local changed since last push
    Conflict,    // linked, remote differs — awaiting the user's choice
    BackingUp,   // a push is in progress
    Restoring,   // a pull/restore is in progress
    Error,       // last operation failed (sticky until next success)
};
std::string to_string(BackupState s);

struct RemoteState {
    bool exists = false;
    int generation = 0;
    int entries = 0;
    long long total_size = 0;
    std::string created_at;
};

enum class LinkOutcome {
    Fresh,       // no remote backup -> seed the first push
    RemoteOnly,  // remote has data, local empty -> auto-adopt
    InSync,      // remote == local -> nothing to do
    Diverged,    // both have data and differ -> ASK the user
};

struct LinkAssessment {
    LinkOutcome outcome;
    RemoteState remote;
};

struct BackupResult {
    int uploaded = 0;
    int skipped = 0;
    int generation = 0;
    std::string root;
};

class BackupEngine {
public:
    using OnChange = std::function<void()>;

    BackupEngine(whisperx::db::SessionStore& store, std::string data_dir,
                 std::unique_ptr<StorageBackend> backend, int interval, bool gc,
                 OnChange on_change);
    ~BackupEngine();

    BackupEngine(const BackupEngine&) = delete;
    BackupEngine& operator=(const BackupEngine&) = delete;

    bool is_linked();
    bool is_dirty();
    BackupState state();
    int interval() const { return interval_; }
    nlohmann::json status();  // {state, linked, backend, dirty, last_root,
                              //  last_backup_at, last_error, interval}

    // Snapshot the DB, stream changed blobs up, commit a new manifest. Throws if
    // no backend is linked. Serialized by sync_lock_.
    BackupResult backup_now(std::optional<bool> gc = std::nullopt);

    // Pull the remote mirror down to local (streamed). Returns file count.
    // Refuses while jobs are active. Throws if not linked / no remote.
    int restore(bool prune = true);

    // Inspect the remote right after linking.
    RemoteState bootstrap();
    // Classify a freshly-linked remote against local data.
    LinkAssessment assess_link();
    int adopt_remote();             // "load existing" -> restore
    BackupResult overwrite_remote();  // "start fresh" -> push over remote (GC)

    void start_periodic();
    void stop();

private:
    void notify();
    void load_state();
    void save_state();
    void set_activity(BackupState a, const std::string& error = "");
    void set_awaiting_decision(bool v);
    BackupResult do_backup(bool gc);
    int do_restore(bool prune);
    void gc_orphans(const Manifest& m);
    void prune_local(const Manifest& remote);
    std::string local_root();
    void periodic_loop();

    whisperx::db::SessionStore& store_;
    std::string data_dir_;
    std::unique_ptr<StorageBackend> backend_;
    int interval_;
    bool gc_;
    OnChange on_change_;

    std::string snapshot_dir_;  // <data_dir>/.backup
    std::string state_path_;    // <data_dir>/.backup/state.json

    std::mutex sync_lock_;       // one backup/restore at a time
    std::mutex activity_lock_;
    BackupState activity_ = BackupState::Idle;
    std::string last_error_;
    std::atomic<bool> awaiting_decision_{false};

    std::optional<std::string> last_signature_;
    std::optional<std::string> last_root_;
    std::optional<std::string> last_backup_at_;

    std::thread thread_;
    std::mutex stop_mu_;
    std::condition_variable stop_cv_;
    bool stop_ = false;
};

}  // namespace whisperx::server::backup
