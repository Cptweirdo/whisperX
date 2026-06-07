#include "db/session_store.hpp"

#include <SQLiteCpp/SQLiteCpp.h>
#include <SQLiteCpp/Backup.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <set>

#include "edits/edits.hpp"
#include "time_iso.hpp"

namespace fs = std::filesystem;

namespace whisperx::db {

namespace {

// Sidecar artifact basenames — mirror app.store module constants.
constexpr const char* kResultFile = "transcript.json";
constexpr const char* kEditsFile = "transcript.edits.json";
constexpr const char* kTranslationBasename = "transcript.translation";

// Read a JSON file into a value, or json null when the file is absent.
json read_json_file(const std::string& path) {
    if (!fs::exists(path)) return json(nullptr);
    std::ifstream f(path, std::ios::binary);
    json j;
    f >> j;
    return j;
}

// Atomic write: dump to <path>.tmp then rename over the target.
void write_json_atomic(const std::string& path, const json& value) {
    fs::create_directories(fs::path(path).parent_path());
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        f << value.dump();  // raw UTF-8, == json.dump(ensure_ascii=False)
    }
    fs::rename(tmp, path);
}

// app.store._SCHEMA, verbatim. `CREATE TABLE IF NOT EXISTS` so re-opening an
// existing DB is a no-op; the 15 `sessions` columns are in the exact §2 order.
constexpr const char* kSchema = R"sql(
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    filename      TEXT,
    audio_filename TEXT,
    status        TEXT NOT NULL,
    stage         TEXT,
    error         TEXT,
    options       TEXT,
    language      TEXT,
    diarized      INTEGER,
    model         TEXT,
    num_segments  INTEGER,
    duration      REAL,
    translations  TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS speaker_names (
    session_id  TEXT NOT NULL,
    speaker_key TEXT NOT NULL,
    name        TEXT NOT NULL,
    PRIMARY KEY (session_id, speaker_key)
);
)sql";

// app.store._COLUMNS — selected order for row_to_dict.
constexpr std::array<const char*, 15> kColumns = {
    "id",           "filename",      "audio_filename", "status",
    "stage",        "error",         "options",        "language",
    "diarized",     "model",         "num_segments",   "duration",
    "translations", "created_at",    "updated_at"};

void bind_opt(SQLite::Statement& q, int i, const std::optional<std::string>& v) {
    if (v.has_value()) {
        q.bind(i, *v);
    } else {
        q.bind(i);  // NULL
    }
}

// Mirror app.store._row_to_dict on a SELECT * row positioned at the current step.
json row_to_dict(SQLite::Statement& q) {
    json d = json::object();
    for (const char* col : kColumns) {
        const SQLite::Column c = q.getColumn(col);
        if (c.isNull()) {
            d[col] = nullptr;
        } else if (std::string(col) == "options" ||
                   std::string(col) == "translations") {
            // JSON text → parsed object (Python: parsed on read; {} on bad JSON).
            try {
                d[col] = json::parse(c.getString());
            } catch (const json::parse_error&) {
                d[col] = json::object();
            }
        } else if (std::string(col) == "diarized") {
            d[col] = c.getInt() != 0;  // INTEGER 0/1 → bool
        } else if (std::string(col) == "num_segments") {
            d[col] = c.getInt64();
        } else if (std::string(col) == "duration") {
            d[col] = c.getDouble();
        } else {
            d[col] = c.getString();
        }
    }
    return d;
}

}  // namespace

SessionStore::SessionStore(const std::string& data_dir) {
    data_dir_ = fs::absolute(data_dir).string();
    sessions_root_ = (fs::path(data_dir_) / "sessions").string();
    fs::create_directories(sessions_root_);
    open_();
}

SessionStore::~SessionStore() = default;

std::string SessionStore::db_path() const {
    return (fs::path(data_dir_) / "sessions.db").string();
}

std::string SessionStore::session_dir_(const std::string& session_id) const {
    return (fs::path(sessions_root_) / session_id).string();
}

void SessionStore::open_() {
    db_ = std::make_unique<SQLite::Database>(
        db_path(), SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
    db_->exec("PRAGMA journal_mode=WAL");
    db_->exec(kSchema);
    migrate_();
}

void SessionStore::migrate_() {
    std::set<std::string> cols;
    SQLite::Statement q(*db_, "PRAGMA table_info(sessions)");
    while (q.executeStep()) {
        cols.insert(q.getColumn("name").getString());
    }
    if (!cols.count("stage")) {
        db_->exec("ALTER TABLE sessions ADD COLUMN stage TEXT");
    }
    if (!cols.count("translations")) {
        db_->exec("ALTER TABLE sessions ADD COLUMN translations TEXT");
    }
}

// --- backup / restore ----------------------------------------------------
void SessionStore::snapshot_db(const std::string& dest_path) {
    const fs::path dest = fs::absolute(dest_path);
    fs::create_directories(dest.parent_path());
    std::lock_guard<std::mutex> g(lock_);
    SQLite::Database dst(dest.string(),
                         SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
    SQLite::Backup backup(dst, *db_);
    backup.executeStep();  // -1: copy all remaining pages in one step
}

void SessionStore::swap_db(const std::string& new_db_path) {
    std::lock_guard<std::mutex> g(lock_);
    db_.reset();  // close (checkpoints WAL)
    const std::string target = db_path();
    std::error_code ec;
    for (const std::string& sidecar : {target + "-wal", target + "-shm"}) {
        if (fs::exists(sidecar)) {
            fs::remove(sidecar, ec);
        }
    }
    fs::rename(new_db_path, target, ec);
    if (ec) {  // cross-device or other: fall back to copy+remove (os.replace semantics)
        fs::copy_file(new_db_path, target, fs::copy_options::overwrite_existing);
        fs::remove(new_db_path, ec);
    }
    open_();
}

// --- writes --------------------------------------------------------------
void SessionStore::create(const std::string& session_id,
                          const std::string& filename,
                          const std::string& audio_filename,
                          const json& options,
                          const std::optional<std::string>& model) {
    const std::string ts = now_iso();
    std::lock_guard<std::mutex> g(lock_);
    SQLite::Statement q(
        *db_,
        "INSERT INTO sessions (id, filename, audio_filename, status, options, "
        "model, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)");
    q.bind(1, session_id);
    q.bind(2, filename);
    q.bind(3, audio_filename);
    q.bind(4, "queued");
    q.bind(5, options.dump());
    bind_opt(q, 6, model);
    q.bind(7, ts);
    q.bind(8, ts);
    q.exec();
}

void SessionStore::mark_running(const std::string& session_id) {
    std::lock_guard<std::mutex> g(lock_);
    SQLite::Statement q(
        *db_, "UPDATE sessions SET status='running', updated_at=? WHERE id=?");
    q.bind(1, now_iso());
    q.bind(2, session_id);
    q.exec();
}

void SessionStore::mark_stage(const std::string& session_id,
                              const std::optional<std::string>& stage) {
    std::lock_guard<std::mutex> g(lock_);
    SQLite::Statement q(
        *db_, "UPDATE sessions SET stage=?, updated_at=? WHERE id=?");
    bind_opt(q, 1, stage);
    q.bind(2, now_iso());
    q.bind(3, session_id);
    q.exec();
}

void SessionStore::mark_duration(const std::string& session_id, double duration) {
    std::lock_guard<std::mutex> g(lock_);
    SQLite::Statement q(
        *db_, "UPDATE sessions SET duration=?, updated_at=? WHERE id=?");
    q.bind(1, duration);
    q.bind(2, now_iso());
    q.bind(3, session_id);
    q.exec();
}

void SessionStore::mark_done(const std::string& session_id,
                             const std::optional<std::string>& language,
                             bool diarized, const std::string& model,
                             long long num_segments, double duration) {
    std::lock_guard<std::mutex> g(lock_);
    SQLite::Statement q(
        *db_,
        "UPDATE sessions SET status='done', stage=NULL, error=NULL, language=?, "
        "diarized=?, model=?, num_segments=?, duration=?, updated_at=? WHERE id=?");
    bind_opt(q, 1, language);
    q.bind(2, diarized ? 1 : 0);
    q.bind(3, model);
    q.bind(4, static_cast<int64_t>(num_segments));
    q.bind(5, duration);
    q.bind(6, now_iso());
    q.bind(7, session_id);
    q.exec();
}

void SessionStore::mark_error(const std::string& session_id,
                              const std::string& message) {
    std::lock_guard<std::mutex> g(lock_);
    SQLite::Statement q(
        *db_,
        "UPDATE sessions SET status='error', stage=NULL, error=?, updated_at=? "
        "WHERE id=?");
    q.bind(1, message);
    q.bind(2, now_iso());
    q.bind(3, session_id);
    q.exec();
}

void SessionStore::rename(const std::string& session_id, const std::string& name) {
    std::lock_guard<std::mutex> g(lock_);
    SQLite::Statement q(
        *db_, "UPDATE sessions SET filename=?, updated_at=? WHERE id=?");
    q.bind(1, name);
    q.bind(2, now_iso());
    q.bind(3, session_id);
    q.exec();
}

bool SessionStore::remove(const std::string& session_id) {
    bool existed;
    {
        std::lock_guard<std::mutex> g(lock_);
        SQLite::Transaction txn(*db_);
        SQLite::Statement del(*db_, "DELETE FROM sessions WHERE id=?");
        del.bind(1, session_id);
        existed = del.exec() > 0;
        SQLite::Statement delsp(*db_,
                                "DELETE FROM speaker_names WHERE session_id=?");
        delsp.bind(1, session_id);
        delsp.exec();
        txn.commit();
    }
    std::error_code ec;
    fs::remove_all(session_dir_(session_id), ec);  // ignore_errors
    return existed;
}

// --- speaker name overrides ---------------------------------------------
json SessionStore::get_speaker_names(const std::string& session_id) {
    std::lock_guard<std::mutex> g(lock_);
    json out = json::object();
    SQLite::Statement q(
        *db_,
        "SELECT speaker_key, name FROM speaker_names WHERE session_id=?");
    q.bind(1, session_id);
    while (q.executeStep()) {
        out[q.getColumn(0).getString()] = q.getColumn(1).getString();
    }
    return out;
}

void SessionStore::set_speaker_name(const std::string& session_id,
                                    const std::string& speaker_key,
                                    const std::string& name) {
    // Python: name = (name or "").strip(); blank clears the override.
    const auto first = name.find_first_not_of(" \t\n\r\f\v");
    std::string trimmed =
        (first == std::string::npos)
            ? std::string()
            : name.substr(first, name.find_last_not_of(" \t\n\r\f\v") - first + 1);
    std::lock_guard<std::mutex> g(lock_);
    if (trimmed.empty()) {
        SQLite::Statement q(
            *db_,
            "DELETE FROM speaker_names WHERE session_id=? AND speaker_key=?");
        q.bind(1, session_id);
        q.bind(2, speaker_key);
        q.exec();
        return;
    }
    SQLite::Statement q(
        *db_,
        "INSERT INTO speaker_names (session_id, speaker_key, name) VALUES (?,?,?) "
        "ON CONFLICT(session_id, speaker_key) DO UPDATE SET name=excluded.name");
    q.bind(1, session_id);
    q.bind(2, speaker_key);
    q.bind(3, trimmed);
    q.exec();
}

// --- settings ------------------------------------------------------------
std::optional<std::string> SessionStore::get_setting(
    const std::string& key, const std::optional<std::string>& def) {
    std::lock_guard<std::mutex> g(lock_);
    SQLite::Statement q(*db_, "SELECT value FROM settings WHERE key=?");
    q.bind(1, key);
    if (q.executeStep()) {
        if (q.getColumn(0).isNull()) return def;
        return q.getColumn(0).getString();
    }
    return def;
}

void SessionStore::set_setting(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> g(lock_);
    SQLite::Statement q(
        *db_,
        "INSERT INTO settings (key, value) VALUES (?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value");
    q.bind(1, key);
    q.bind(2, value);
    q.exec();
}

// --- translations (JSON column) -----------------------------------------
json SessionStore::get_translations(const std::string& session_id) {
    // Python: (row or {}).get("translations") or {}  — get() takes the lock.
    json row = get(session_id);
    if (row.is_null() || !row.contains("translations") ||
        row["translations"].is_null()) {
        return json::object();
    }
    return row["translations"];
}

json SessionStore::set_translation_status(
    const std::string& session_id, const std::string& lang,
    const std::string& status, const std::optional<std::string>& service,
    const std::optional<std::string>& error) {
    std::lock_guard<std::mutex> g(lock_);
    json current = json::object();
    {
        SQLite::Statement q(
            *db_, "SELECT translations FROM sessions WHERE id=?");
        q.bind(1, session_id);
        if (q.executeStep() && !q.getColumn(0).isNull()) {
            try {
                current = json::parse(q.getColumn(0).getString());
            } catch (const json::parse_error&) {
                current = json::object();
            }
            if (!current.is_object()) current = json::object();
        }
    }
    json entry = current.contains(lang) ? current[lang] : json::object();
    entry["status"] = status;
    if (service.has_value()) {
        entry["service"] = *service;
    }
    if (error.has_value()) {
        entry["error"] = *error;
    } else if (status != "error") {
        entry.erase("error");
    }
    current[lang] = entry;
    SQLite::Statement up(
        *db_, "UPDATE sessions SET translations=?, updated_at=? WHERE id=?");
    up.bind(1, current.dump());
    up.bind(2, now_iso());
    up.bind(3, session_id);
    up.exec();
    return current;
}

// --- reads ---------------------------------------------------------------
json SessionStore::get(const std::string& session_id) {
    std::lock_guard<std::mutex> g(lock_);
    SQLite::Statement q(*db_, "SELECT * FROM sessions WHERE id=?");
    q.bind(1, session_id);
    if (q.executeStep()) {
        return row_to_dict(q);
    }
    return json(nullptr);
}

json SessionStore::list() {
    std::lock_guard<std::mutex> g(lock_);
    json out = json::array();
    SQLite::Statement q(
        *db_, "SELECT * FROM sessions ORDER BY created_at DESC, id DESC");
    while (q.executeStep()) {
        out.push_back(row_to_dict(q));
    }
    return out;
}

bool SessionStore::has_active_jobs() {
    std::lock_guard<std::mutex> g(lock_);
    SQLite::Statement q(
        *db_,
        "SELECT 1 FROM sessions WHERE status IN ('queued','running') LIMIT 1");
    return q.executeStep();
}

// --- lifecycle -----------------------------------------------------------
std::vector<std::string> SessionStore::reconcile_startup() {
    std::lock_guard<std::mutex> g(lock_);
    std::vector<std::string> ids;
    {
        SQLite::Statement q(
            *db_,
            "SELECT id FROM sessions WHERE status IN ('queued','running')");
        while (q.executeStep()) {
            ids.push_back(q.getColumn(0).getString());
        }
    }
    if (!ids.empty()) {
        SQLite::Statement up(
            *db_,
            "UPDATE sessions SET status='queued', stage=NULL, error=NULL, "
            "updated_at=? WHERE status IN ('queued','running')");
        up.bind(1, now_iso());
        up.exec();
    }
    return ids;
}

void SessionStore::close() {
    std::lock_guard<std::mutex> g(lock_);
    db_.reset();
}

// --- file-backed sidecars ------------------------------------------------
std::string SessionStore::result_path_(const std::string& sid) const {
    return (fs::path(session_dir_(sid)) / kResultFile).string();
}

std::string SessionStore::edits_path_(const std::string& sid) const {
    return (fs::path(session_dir_(sid)) / kEditsFile).string();
}

std::string SessionStore::translation_path_(const std::string& sid,
                                            const std::string& lang) const {
    return (fs::path(session_dir_(sid)) /
            (std::string(kTranslationBasename) + "." + lang + ".json"))
        .string();
}

json SessionStore::load_result(const std::string& sid) {
    return read_json_file(result_path_(sid));
}

json SessionStore::load_edits(const std::string& sid) {
    return read_json_file(edits_path_(sid));
}

json SessionStore::original_segments_(const std::string& sid) {
    // (self.load_result(sid) or {}).get("segments", [])
    const json result = load_result(sid);
    if (result.is_object()) {
        auto it = result.find("segments");
        if (it != result.end()) return *it;
    }
    return json::array();
}

json SessionStore::baseline_segments_(const std::string& sid) {
    return whisperx::edits::coalesce_segments(original_segments_(sid));
}

json SessionStore::current_segments(const std::string& sid,
                                    const json& original_segments) {
    const json edits = load_edits(sid);
    if (edits.is_object()) {
        auto it = edits.find("segments");
        if (it != edits.end() && !it->is_null()) return *it;
    }
    return whisperx::edits::coalesce_segments(original_segments);
}

long SessionStore::edit_history_len(const std::string& sid) {
    const json edits = load_edits(sid);
    if (edits.is_object()) {
        auto it = edits.find("history");
        if (it != edits.end() && it->is_array()) {
            return static_cast<long>(it->size());
        }
    }
    return 0;
}

void SessionStore::write_edits_(const std::string& sid, const json& segments,
                                const json& history) {
    json overlay = json::object();
    overlay["version"] = 1;
    overlay["segments"] = segments;
    overlay["history"] = history;
    write_json_atomic(edits_path_(sid), overlay);
}

namespace {
// history.append(delta); cap to the last HISTORY_LIMIT entries.
void append_capped(json& history, const json& delta) {
    history.push_back(delta);
    const int limit = whisperx::edits::HISTORY_LIMIT;
    if (static_cast<int>(history.size()) > limit) {
        json trimmed = json::array();
        for (int k = static_cast<int>(history.size()) - limit;
             k < static_cast<int>(history.size()); ++k) {
            trimmed.push_back(history[k]);
        }
        history = std::move(trimmed);
    }
}
}  // namespace

json SessionStore::save_turn_edit(const std::string& sid, long turn_index,
                                  const std::string& new_text) {
    std::lock_guard<std::mutex> g(files_lock_);
    const json edits = load_edits(sid);
    const bool has = edits.is_object();
    json segments = has ? edits["segments"] : baseline_segments_(sid);
    json history = has ? json(edits["history"]) : json::array();
    auto [new_segments, delta] =
        whisperx::edits::apply_turn_edit(segments, turn_index, new_text);
    append_capped(history, delta);
    write_edits_(sid, new_segments, history);
    return new_segments;
}

json SessionStore::save_turn_reassign(const std::string& sid, long turn_index,
                                      const std::string& new_speaker) {
    std::lock_guard<std::mutex> g(files_lock_);
    const json edits = load_edits(sid);
    const bool has = edits.is_object();
    json segments = has ? edits["segments"] : baseline_segments_(sid);
    json history = has ? json(edits["history"]) : json::array();
    json new_segments;
    json delta;
    try {
        auto pr = whisperx::edits::apply_turn_reassign(segments, turn_index,
                                                       new_speaker);
        new_segments = std::move(pr.first);
        delta = std::move(pr.second);
    } catch (const whisperx::edits::NoChange&) {
        return segments;  // no-op: nothing written
    }
    append_capped(history, delta);
    write_edits_(sid, new_segments, history);
    return new_segments;
}

json SessionStore::undo_turn_edit(const std::string& sid) {
    std::lock_guard<std::mutex> g(files_lock_);
    const json edits = load_edits(sid);
    const json baseline = baseline_segments_(sid);
    const bool has = edits.is_object();
    const bool has_history =
        has && edits.contains("history") && !edits["history"].empty();
    if (!has_history) {
        return has ? edits["segments"] : baseline;
    }
    auto [new_segments, new_history] =
        whisperx::edits::undo_last(edits["segments"], edits["history"]);
    if (new_history.empty() && new_segments == baseline) {
        std::error_code ec;
        fs::remove(edits_path_(sid), ec);  // fully reverted -> drop the overlay
        return baseline;
    }
    write_edits_(sid, new_segments, new_history);
    return new_segments;
}

json SessionStore::load_translation(const std::string& sid,
                                    const std::string& lang) {
    return read_json_file(translation_path_(sid, lang));
}

void SessionStore::save_translation(const std::string& sid,
                                    const std::string& lang,
                                    const json& payload) {
    std::lock_guard<std::mutex> g(files_lock_);
    write_json_atomic(translation_path_(sid, lang), payload);
}

}  // namespace whisperx::db
