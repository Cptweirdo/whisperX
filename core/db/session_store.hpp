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
// Also owns the file-backed sidecar subsystems (Phase 1 completion slice): the
// transcript.json / transcript.edits.json edits + undo overlay (delegating the
// algorithms to whisperx::edits, the port of app/edits.py) and the per-language
// translation overlay files. These are guarded by a separate `files_lock_` and
// gated independently — the Python facade forwards the DB methods to this class
// when `WHISPERX_CORE_STAGES` contains `db`, and the file methods when it
// contains `edits`. The pure path helpers stay on the Python facade.
#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "db/session_row.hpp"

namespace SQLite {
class Database;
}

namespace whisperx::db {

using nlohmann::json;

// Session rows cross this API as typed `SessionRow` structs (see
// session_row.hpp): status/stage enums, std::optional for nullable columns, a
// typed translations map. Only `options` and the file-backed sidecars stay
// opaque json. The pybind facade converts via SessionRow::to_json(), which
// mirrors app.store._row_to_dict exactly, so the Python-visible dict shape
// (and the §2 on-disk contract) is unchanged.
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
                    const std::optional<Stage>& stage);
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
    TranslationMap get_translations(const std::string& session_id);
    TranslationMap set_translation_status(
        const std::string& session_id, const std::string& lang, Status status,
        const std::optional<std::string>& service,
        const std::optional<std::string>& error);

    // --- reads -----------------------------------------------------------
    std::optional<SessionRow> get(const std::string& session_id);
    std::vector<SessionRow> list();
    bool has_active_jobs();

    // --- lifecycle -------------------------------------------------------
    std::vector<std::string> reconcile_startup();
    void close();

    // --- file-backed sidecars (edits/undo overlay + translation files) ---
    json load_result(const std::string& session_id);   // object or null
    json load_edits(const std::string& session_id);     // object or null
    json current_segments(const std::string& session_id,
                          const json& original_segments);
    long edit_history_len(const std::string& session_id);
    json save_turn_edit(const std::string& session_id, long turn_index,
                        const std::string& new_text);
    json save_turn_reassign(const std::string& session_id, long turn_index,
                            const std::string& new_speaker);
    json save_turn_split(const std::string& session_id, long turn_index,
                         long sel_start, long sel_end,
                         const std::string& new_speaker);
    json undo_turn_edit(const std::string& session_id);
    json load_translation(const std::string& session_id,
                          const std::string& lang);  // object or null
    void save_translation(const std::string& session_id, const std::string& lang,
                          const json& payload);

private:
    void open_();      // connect + WAL + schema + migrate
    void migrate_();   // idempotent ADD COLUMN for stage / translations
    std::string session_dir_(const std::string& session_id) const;

    // sidecar path helpers + internals (mirror app.store private methods).
    std::string result_path_(const std::string& session_id) const;
    std::string edits_path_(const std::string& session_id) const;
    std::string translation_path_(const std::string& session_id,
                                  const std::string& lang) const;
    json original_segments_(const std::string& session_id);
    json baseline_segments_(const std::string& session_id);
    void write_edits_(const std::string& session_id, const json& segments,
                      const json& history);

    std::string data_dir_;
    std::string sessions_root_;
    std::mutex lock_;        // serializes SQLite access
    std::mutex files_lock_;  // serializes the file-backed sidecar writes
    std::unique_ptr<SQLite::Database> db_;
};

}  // namespace whisperx::db
