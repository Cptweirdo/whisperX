// C++ SessionStore — the Phase-1 port of app/store.py's SQLite layer.
//
// Scope (decided, Phase 1 = "DB-only"): this owns the SQLite side of the
// session store — schema, the idempotent migration, CRUD + lifecycle, settings,
// speaker-name overrides, the `translations` JSON column, and the WAL-safe
// snapshot/swap backup primitives. It honors the §2 session-DB compatibility
// contract byte-for-byte (same 3 tables, same column order, WAL, ISO-8601 UTC
// seconds timestamps, same migration) so a `sessions.db` written by either the
// Python or the C++ store round-trips through the other.
//
// Out of scope here (kept in the Python facade during the strangler window): the
// file-backed sidecar subsystems — transcript.json / transcript.edits.json
// (edits + undo, app/edits.py) and the per-language translation overlay files —
// plus the pure path helpers. The Python `app.store.SessionStore` facade forwards
// the DB methods below to this class when `WHISPERX_CORE_STAGES` contains `db`.
#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace SQLite {
class Database;
}

namespace whisperx::db {

using nlohmann::json;

// Mirrors app.store._row_to_dict: a session row as a JSON object with the 15
// columns as keys, `options`/`translations` parsed to objects (or null),
// `diarized` as a bool (or null), numeric columns as int/double (or null).
// `get()` returns this (or JSON null for a missing id); `list()` an array of them.
class SessionStore {
public:
    explicit SessionStore(const std::string& data_dir);
    ~SessionStore();

    SessionStore(const SessionStore&) = delete;
    SessionStore& operator=(const SessionStore&) = delete;

    std::string db_path() const;

    // --- backup / restore ------------------------------------------------
    void snapshot_db(const std::string& dest_path);
    void swap_db(const std::string& new_db_path);

    // --- writes ----------------------------------------------------------
    void create(const std::string& session_id, const std::string& filename,
                const std::string& audio_filename, const json& options,
                const std::optional<std::string>& model);
    void mark_running(const std::string& session_id);
    void mark_stage(const std::string& session_id,
                    const std::optional<std::string>& stage);
    void mark_duration(const std::string& session_id, double duration);
    void mark_done(const std::string& session_id,
                   const std::optional<std::string>& language, bool diarized,
                   const std::string& model, long long num_segments,
                   double duration);
    void mark_error(const std::string& session_id, const std::string& message);
    void rename(const std::string& session_id, const std::string& name);
    bool remove(const std::string& session_id);  // app.store: delete()

    // --- speaker name overrides -----------------------------------------
    json get_speaker_names(const std::string& session_id);  // {key: name}
    void set_speaker_name(const std::string& session_id,
                          const std::string& speaker_key, const std::string& name);

    // --- settings --------------------------------------------------------
    std::optional<std::string> get_setting(
        const std::string& key,
        const std::optional<std::string>& def = std::nullopt);
    void set_setting(const std::string& key, const std::string& value);

    // --- translations (the JSON column only; sidecar files stay in Python) ---
    json get_translations(const std::string& session_id);
    json set_translation_status(const std::string& session_id,
                                const std::string& lang, const std::string& status,
                                const std::optional<std::string>& service,
                                const std::optional<std::string>& error);

    // --- reads -----------------------------------------------------------
    json get(const std::string& session_id);  // object or null
    json list();                              // array
    bool has_active_jobs();

    // --- lifecycle -------------------------------------------------------
    std::vector<std::string> reconcile_startup();
    void close();

private:
    void open_();      // connect + WAL + schema + migrate
    void migrate_();   // idempotent ADD COLUMN for stage / translations
    std::string session_dir_(const std::string& session_id) const;

    std::string data_dir_;
    std::string sessions_root_;
    std::mutex lock_;
    std::unique_ptr<SQLite::Database> db_;
};

}  // namespace whisperx::db
