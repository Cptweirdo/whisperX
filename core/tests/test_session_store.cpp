// Catch2 coverage for the C++ SessionStore (Phase 1 DB layer). Exercises the
// §2 compatibility contract in isolation: schema/migration, CRUD + lifecycle,
// settings, speaker-name overrides, the translations JSON column, ordering, and
// the snapshot/swap backup primitives. Cross-impl byte parity with the Python
// store is checked separately by bindings/test/test_store_parity.py.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <set>
#include <string>

#if defined(_WIN32)
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include <SQLiteCpp/SQLiteCpp.h>

#include "db/session_store.hpp"

namespace fs = std::filesystem;
using whisperx::db::SessionRow;
using whisperx::db::SessionStore;
using whisperx::db::Stage;
using whisperx::db::Status;
using whisperx::db::TranslationMap;
using nlohmann::json;

namespace {

// A unique temp data_dir per test; removed on destruction.
struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("wx_store_" + std::to_string(::getpid()) + "_" +
                std::to_string(reinterpret_cast<uintptr_t>(this)));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string str() const { return path.string(); }
};

// Drop a transcript.json sidecar (the ASR result) next to a session row so the
// file-backed edit methods have a baseline to read.
void write_result(const TempDir& dir, const std::string& sid,
                  const json& segments) {
    fs::path d = fs::path(dir.path) / "sessions" / sid;
    fs::create_directories(d);
    std::ofstream(d / "transcript.json")
        << json{{"segments", segments}}.dump();
}

// One SPEAKER_00 turn over two segments: "Hello there good friend" — atoms
// Hello[0,5) there[6,11) good[12,16) friend[17,23).
json one_turn_segments() {
    return json::array(
        {json{{"start", 0.0},
              {"end", 1.0},
              {"text", "Hello there"},
              {"speaker", "SPEAKER_00"},
              {"words", json::array({json{{"word", "Hello"}, {"start", 0.0}, {"end", 0.5}},
                                     json{{"word", "there"}, {"start", 0.5}, {"end", 1.0}}})}},
         json{{"start", 1.0},
              {"end", 2.0},
              {"text", "good friend"},
              {"speaker", "SPEAKER_00"},
              {"words", json::array({json{{"word", "good"}, {"start", 1.0}, {"end", 1.5}},
                                     json{{"word", "friend"}, {"start", 1.5}, {"end", 2.0}}})}}});
}

}  // namespace

TEST_CASE("save_turn_split writes the 3-way overlay (S1)", "[store]") {
    TempDir dir;
    SessionStore store(dir.str());
    store.create("s1", "rec.mp3", "audio.mp3", json::object(),
                 std::optional<std::string>("tiny"));
    write_result(dir, "s1", one_turn_segments());

    json out = store.save_turn_split("s1", 0, 6, 16, "SPEAKER_01");  // "there good"
    REQUIRE(out.size() == 3);
    CHECK(out[0]["text"] == "Hello");
    CHECK(out[0]["speaker"] == "SPEAKER_00");
    CHECK(out[1]["text"] == "there good");
    CHECK(out[1]["speaker"] == "SPEAKER_01");
    CHECK(out[2]["text"] == "friend");
    CHECK(out[2]["speaker"] == "SPEAKER_00");
    CHECK(store.edit_history_len("s1") == 1);
    CHECK(store.load_edits("s1").is_object());
}

TEST_CASE("undo after a split reverts and drops the overlay (S2)", "[store]") {
    TempDir dir;
    SessionStore store(dir.str());
    store.create("s2", "rec.mp3", "audio.mp3", json::object(),
                 std::optional<std::string>("tiny"));
    write_result(dir, "s2", one_turn_segments());

    store.save_turn_split("s2", 0, 6, 16, "SPEAKER_01");
    json reverted = store.undo_turn_edit("s2");
    REQUIRE(reverted.size() == 2);
    CHECK(reverted[0]["speaker"] == "SPEAKER_00");
    CHECK(reverted[1]["speaker"] == "SPEAKER_00");
    CHECK(store.edit_history_len("s2") == 0);
    CHECK(store.load_edits("s2").is_null());  // back at baseline -> overlay removed
}

TEST_CASE("same-speaker split is a NoChange that writes nothing (S3)", "[store]") {
    TempDir dir;
    SessionStore store(dir.str());
    store.create("s3", "rec.mp3", "audio.mp3", json::object(),
                 std::optional<std::string>("tiny"));
    write_result(dir, "s3", one_turn_segments());

    json out = store.save_turn_split("s3", 0, 6, 16, "SPEAKER_00");  // already 00
    CHECK(out.size() == 2);                       // baseline, untouched
    CHECK(store.edit_history_len("s3") == 0);
    CHECK(store.load_edits("s3").is_null());      // nothing persisted
}

TEST_CASE("create + get round-trips the row shape", "[store]") {
    TempDir dir;
    SessionStore store(dir.str());
    store.create("s1", "My Recording.mp3", "audio.mp3",
                 json{{"diarize", true}, {"model", "large-v2"}}, "large-v2");

    auto row = store.get("s1");
    REQUIRE(row.has_value());
    CHECK(row->id == "s1");
    CHECK(row->filename == "My Recording.mp3");
    CHECK(row->audio_filename == "audio.mp3");
    CHECK(row->status == Status::Queued);
    CHECK(row->model == "large-v2");
    // options stored as JSON text, parsed back to an object.
    CHECK(row->options["diarize"] == true);
    CHECK(row->options["model"] == "large-v2");
    // unset columns are nullopt; created_at == updated_at on insert.
    CHECK_FALSE(row->stage.has_value());
    CHECK_FALSE(row->error.has_value());
    CHECK_FALSE(row->language.has_value());
    CHECK_FALSE(row->diarized.has_value());
    CHECK_FALSE(row->num_segments.has_value());
    CHECK_FALSE(row->duration.has_value());
    CHECK_FALSE(row->translations.has_value());
    CHECK(row->created_at == row->updated_at);

    CHECK_FALSE(store.get("missing").has_value());
}

TEST_CASE("to_json reproduces the _row_to_dict shape", "[store]") {
    // The pybind facade returns row->to_json() — this is the local guard for
    // the dict shape the Python parity oracle (test_store_parity.py) asserts.
    TempDir dir;
    SessionStore store(dir.str());
    store.create("s1", "rec.mp3", "audio.mp3", json{{"language", "en"}}, "tiny");

    json fresh = store.get("s1")->to_json();
    REQUIRE(fresh.size() == 15);
    CHECK(fresh["id"] == "s1");
    CHECK(fresh["filename"] == "rec.mp3");
    CHECK(fresh["audio_filename"] == "audio.mp3");
    CHECK(fresh["status"] == "queued");
    CHECK(fresh["options"] == json{{"language", "en"}});
    for (const char* k : {"stage", "error", "language", "diarized",
                          "num_segments", "duration", "translations"}) {
        CHECK(fresh[k].is_null());
    }
    CHECK(fresh["created_at"].is_string());
    CHECK(fresh["updated_at"].is_string());

    store.mark_running("s1");
    store.mark_stage("s1", Stage::LoadingAlign);
    store.set_translation_status("s1", "es", Status::Running, "deepl",
                                 std::nullopt);
    json mid = store.get("s1")->to_json();
    CHECK(mid["status"] == "running");
    CHECK(mid["stage"] == "loading_align");
    CHECK(mid["translations"] ==
          json{{"es", {{"status", "running"}, {"service", "deepl"}}}});

    store.mark_done("s1", "en", true, "tiny", 42, 12.5);
    json done = store.get("s1")->to_json();
    CHECK(done["status"] == "done");
    CHECK(done["stage"].is_null());
    CHECK(done["language"] == "en");
    CHECK(done["diarized"].is_boolean());
    CHECK(done["diarized"] == true);
    CHECK(done["num_segments"].is_number_integer());
    CHECK(done["num_segments"] == 42);
    CHECK(done["duration"].is_number_float());
    CHECK(done["duration"] == 12.5);
}

TEST_CASE("get is lenient on unknown enum strings in the DB", "[store]") {
    TempDir dir;
    SessionStore store(dir.str());
    store.create("s1", "f", "a.wav", json::object(), std::nullopt);
    {
        SQLite::Database raw((fs::path(dir.path) / "sessions.db").string(),
                             SQLite::OPEN_READWRITE);
        raw.exec(
            "UPDATE sessions SET status='bogus', stage='bogus', "
            "translations='{\"es\": {\"status\": \"bogus\"}, "
            "\"de\": {\"status\": \"done\"}}', options='not json' "
            "WHERE id='s1'");
    }

    auto row = store.get("s1");  // must not throw
    REQUIRE(row.has_value());
    CHECK(row->status == Status::Error);          // unknown status fallback
    CHECK_FALSE(row->stage.has_value());          // unknown stage -> nullopt
    REQUIRE(row->translations.has_value());
    CHECK_FALSE(row->translations->count("es"));  // bad entry dropped
    CHECK(row->translations->at("de").status == Status::Done);
    CHECK(row->options == json::object());        // bad JSON -> {}
}

TEST_CASE("lifecycle marks update status / stage / done / error", "[store]") {
    TempDir dir;
    SessionStore store(dir.str());
    store.create("s1", "f", "a.wav", json::object(), std::nullopt);

    store.mark_running("s1");
    CHECK(store.get("s1")->status == Status::Running);

    store.mark_stage("s1", Stage::Aligning);
    CHECK(store.get("s1")->stage == Stage::Aligning);
    store.mark_stage("s1", std::nullopt);
    CHECK_FALSE(store.get("s1")->stage.has_value());

    store.mark_duration("s1", 12.5);
    CHECK(store.get("s1")->duration == 12.5);

    store.mark_done("s1", "en", true, "large-v2", 42, 12.5);
    auto done = store.get("s1");
    REQUIRE(done.has_value());
    CHECK(done->status == Status::Done);
    CHECK_FALSE(done->stage.has_value());
    CHECK_FALSE(done->error.has_value());
    CHECK(done->language == "en");
    CHECK(done->diarized == true);  // INTEGER 1 -> bool
    CHECK(done->model == "large-v2");
    CHECK(done->num_segments == 42);
    CHECK(done->duration == 12.5);

    store.mark_error("s1", "boom");
    auto err = store.get("s1");
    REQUIRE(err.has_value());
    CHECK(err->status == Status::Error);
    CHECK(err->error == "boom");
    CHECK_FALSE(err->stage.has_value());

    store.rename("s1", "Renamed");
    CHECK(store.get("s1")->filename == "Renamed");
}

TEST_CASE("list orders by created_at DESC, id DESC", "[store]") {
    TempDir dir;
    SessionStore store(dir.str());
    // Same-second inserts tie on created_at, so id DESC breaks the tie.
    store.create("a", "f", "a.wav", json::object(), std::nullopt);
    store.create("c", "f", "a.wav", json::object(), std::nullopt);
    store.create("b", "f", "a.wav", json::object(), std::nullopt);

    std::vector<SessionRow> rows = store.list();
    REQUIRE(rows.size() == 3);
    CHECK(rows[0].id == "c");
    CHECK(rows[1].id == "b");
    CHECK(rows[2].id == "a");
}

TEST_CASE("settings upsert + default", "[store]") {
    TempDir dir;
    SessionStore store(dir.str());
    CHECK(store.get_setting("active_model") == std::nullopt);
    CHECK(store.get_setting("active_model", "fallback") == "fallback");

    store.set_setting("active_model", "large-v2");
    CHECK(store.get_setting("active_model") == "large-v2");
    store.set_setting("active_model", "tiny");  // ON CONFLICT update
    CHECK(store.get_setting("active_model") == "tiny");
}

TEST_CASE("speaker names upsert + blank clears", "[store]") {
    TempDir dir;
    SessionStore store(dir.str());
    store.create("s1", "f", "a.wav", json::object(), std::nullopt);

    store.set_speaker_name("s1", "SPEAKER_00", "Alice");
    store.set_speaker_name("s1", "SPEAKER_01", "Bob");
    json names = store.get_speaker_names("s1");
    CHECK(names["SPEAKER_00"] == "Alice");
    CHECK(names["SPEAKER_01"] == "Bob");

    store.set_speaker_name("s1", "SPEAKER_00", "  Alyssa  ");  // trimmed upsert
    CHECK(store.get_speaker_names("s1")["SPEAKER_00"] == "Alyssa");

    store.set_speaker_name("s1", "SPEAKER_01", "   ");  // blank clears
    CHECK_FALSE(store.get_speaker_names("s1").contains("SPEAKER_01"));
}

TEST_CASE("translations status upsert mirrors the read-modify-write", "[store]") {
    TempDir dir;
    SessionStore store(dir.str());
    store.create("s1", "f", "a.wav", json::object(), std::nullopt);

    TranslationMap m = store.set_translation_status("s1", "es", Status::Running,
                                                    "deepl", std::nullopt);
    CHECK(m.at("es").status == Status::Running);
    CHECK(m.at("es").service == "deepl");

    // success path drops a prior error and keeps the column in sync with get().
    store.set_translation_status("s1", "es", Status::Error, std::nullopt,
                                 "rate limit");
    CHECK(store.get_translations("s1").at("es").error == "rate limit");
    TranslationMap ok = store.set_translation_status("s1", "es", Status::Done,
                                                     std::nullopt, std::nullopt);
    CHECK(ok.at("es").status == Status::Done);
    CHECK_FALSE(ok.at("es").error.has_value());
    CHECK(store.get_translations("s1").at("es").status == Status::Done);
    CHECK(store.get_translations("missing").empty());
}

TEST_CASE("has_active_jobs + reconcile_startup", "[store]") {
    TempDir dir;
    SessionStore store(dir.str());
    store.create("s1", "f", "a.wav", json::object(), std::nullopt);  // queued
    store.create("s2", "f", "a.wav", json::object(), std::nullopt);
    store.mark_done("s2", "en", false, "tiny", 1, 1.0);
    CHECK(store.has_active_jobs());

    store.mark_running("s1");
    std::vector<std::string> requeue = store.reconcile_startup();
    REQUIRE(requeue.size() == 1);
    CHECK(requeue[0] == "s1");
    CHECK(store.get("s1")->status == Status::Queued);
    CHECK_FALSE(store.get("s1")->stage.has_value());
    // s2 was done, untouched.
    CHECK(store.get("s2")->status == Status::Done);
}

TEST_CASE("delete removes the row, speaker_names, and the session dir", "[store]") {
    TempDir dir;
    SessionStore store(dir.str());
    store.create("s1", "f", "a.wav", json::object(), std::nullopt);
    store.set_speaker_name("s1", "SPEAKER_00", "Alice");
    const fs::path sdir = dir.path / "sessions" / "s1";
    fs::create_directories(sdir);
    REQUIRE(fs::exists(sdir));

    CHECK(store.remove("s1"));
    CHECK_FALSE(store.get("s1").has_value());
    CHECK(store.get_speaker_names("s1").empty());
    CHECK_FALSE(fs::exists(sdir));
    CHECK_FALSE(store.remove("s1"));  // already gone
}

TEST_CASE("snapshot_db + swap_db round-trip", "[store]") {
    TempDir dir;
    SessionStore store(dir.str());
    store.create("s1", "f", "a.wav", json::object(), std::nullopt);

    const std::string snap = (dir.path / "backup" / "snap.db").string();
    store.snapshot_db(snap);
    REQUIRE(fs::exists(snap));

    // a write made after the snapshot is rolled back by swapping it in.
    store.create("s2", "f", "a.wav", json::object(), std::nullopt);
    CHECK(store.get("s2").has_value());
    store.swap_db(snap);
    CHECK(store.get("s1").has_value());
    CHECK_FALSE(store.get("s2").has_value());
}

TEST_CASE("migration adds stage/translations to a legacy DB, idempotently",
          "[store]") {
    TempDir dir;
    const std::string db = (dir.path / "sessions.db").string();
    fs::create_directories(dir.path / "sessions");
    {
        // A pre-migration sessions table (no stage / translations columns).
        SQLite::Database legacy(db, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
        legacy.exec(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, filename TEXT, "
            "audio_filename TEXT, status TEXT NOT NULL, error TEXT, options TEXT, "
            "language TEXT, diarized INTEGER, model TEXT, num_segments INTEGER, "
            "duration REAL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)");
        legacy.exec(
            "INSERT INTO sessions (id, status, created_at, updated_at) "
            "VALUES ('old', 'done', '2020-01-01T00:00:00+00:00', "
            "'2020-01-01T00:00:00+00:00')");
    }

    auto columns = [&]() {
        std::set<std::string> cols;
        SQLite::Database d(db, SQLite::OPEN_READONLY);
        SQLite::Statement q(d, "PRAGMA table_info(sessions)");
        while (q.executeStep()) cols.insert(q.getColumn("name").getString());
        return cols;
    };

    {
        SessionStore store(dir.str());  // opens + migrates
        auto cols = columns();
        CHECK(cols.count("stage"));
        CHECK(cols.count("translations"));
        // legacy row survives and reads back with the new columns unset.
        auto row = store.get("old");
        REQUIRE(row.has_value());
        CHECK(row->status == Status::Done);
        CHECK_FALSE(row->stage.has_value());
        CHECK_FALSE(row->translations.has_value());
    }
    {
        SessionStore store(dir.str());  // re-open: migration is a no-op
        auto cols = columns();
        CHECK(cols.count("stage"));
        CHECK(cols.count("translations"));
    }
}
