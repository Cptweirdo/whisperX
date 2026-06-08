// Catch2 tests for the backup engine — no network. Drives BackupEngine with the
// LocalFsBackend (the filesystem reference store) over a real SessionStore on a
// temp data dir, pinning manifest shaping, content-addressed upload/skip/GC,
// streamed restore + prune, and the link-assessment outcomes. Plus a fake-upload
// check that DriveClient::upload_resumable shapes the resumable request.
#include <catch2/catch_test_macros.hpp>

#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "backup/engine.hpp"
#include "backup/local_backend.hpp"
#include "backup/manifest.hpp"
#include "db/session_store.hpp"
#include "drive/drive_client.hpp"

namespace fs = std::filesystem;
using namespace whisperx::server::backup;
using whisperx::db::SessionStore;

namespace {

std::atomic<int> g_counter{0};

std::string temp_dir(const std::string& tag) {
    fs::path p = fs::temp_directory_path() /
                 ("wxbk_" + tag + "_" + std::to_string(::getpid()) + "_" +
                  std::to_string(g_counter.fetch_add(1)));
    fs::create_directories(p);
    return p.string();
}

void write_file(const std::string& path, const std::string& content) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << content;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// A done-ish session: a DB row + an artifact file under sessions/<id>/.
void seed_session(SessionStore& store, const std::string& data_dir,
                  const std::string& id, const std::string& audio) {
    store.create(id, id + ".wav", "audio.wav", nlohmann::json::object(),
                 std::optional<std::string>("tiny"));
    write_file((fs::path(data_dir) / "sessions" / id / "audio.wav").string(),
               audio);
}

std::unique_ptr<BackupEngine> make_engine(SessionStore& store,
                                          const std::string& data_dir,
                                          const std::string& remote) {
    auto backend = std::make_unique<LocalFsBackend>(remote);
    return std::make_unique<BackupEngine>(store, data_dir, std::move(backend),
                                          /*interval=*/0, /*gc=*/true,
                                          BackupEngine::OnChange{});
}

}  // namespace

TEST_CASE("manifest round-trips and merkle_root tracks path+hash", "[backup]") {
    Manifest m;
    m.generation = 3;
    m.created_at = "2026-06-09T00:00:00+00:00";
    m.entries["sessions.db"] = FileEntry{"aa", 10, 1.0};
    m.entries["sessions/s1/audio.wav"] = FileEntry{"bb", 20, 2.0};

    Manifest back = Manifest::from_json(m.to_json());
    REQUIRE(back.generation == 3);
    REQUIRE(back.entries.size() == 2);
    REQUIRE(back.entries.at("sessions.db").hash == "aa");
    REQUIRE(back.entries.at("sessions/s1/audio.wav").size == 20);
    REQUIRE(back.object_keys() == std::set<std::string>{"aa", "bb"});

    std::string root = merkle_root(m);
    REQUIRE(root == merkle_root(back));  // stable across serialize
    Manifest changed = m;
    changed.entries["sessions/s1/audio.wav"].hash = "cc";
    REQUIRE(merkle_root(changed) != root);  // content change moves the root
    // mtime is informational only — it must NOT move the merkle root.
    Manifest touched = m;
    touched.entries["sessions.db"].mtime = 999.0;
    REQUIRE(merkle_root(touched) == root);
}

TEST_CASE("changed_paths is everything vs no remote, nothing vs identical",
          "[backup]") {
    Manifest local;
    local.entries["a"] = FileEntry{"h1", 1, 0};
    local.entries["b"] = FileEntry{"h2", 1, 0};
    REQUIRE(changed_paths(local, std::nullopt).size() == 2);
    REQUIRE(changed_paths(local, local).empty());

    Manifest remote;
    remote.entries["a"] = FileEntry{"h1", 1, 0};  // only "a" is known remotely
    auto diff = changed_paths(local, remote);
    REQUIRE(diff.size() == 1);
    REQUIRE(diff[0] == "b");
}

TEST_CASE("backup_now uploads new blobs, skips repeats, GCs orphans",
          "[backup]") {
    std::string data_dir = temp_dir("data");
    std::string remote = temp_dir("remote");
    SessionStore store(data_dir);
    seed_session(store, data_dir, "s1", "HELLO-AUDIO");

    auto engine = make_engine(store, data_dir, remote);
    REQUIRE(engine->is_linked());

    BackupResult r1 = engine->backup_now();
    REQUIRE(r1.generation == 1);
    REQUIRE(r1.uploaded >= 2);  // sessions.db + audio.wav (at least)
    REQUIRE(fs::exists(fs::path(remote) / "manifest.json"));
    REQUIRE(fs::is_directory(fs::path(remote) / "objects"));

    // Nothing changed -> a second push uploads nothing (content-addressed).
    BackupResult r2 = engine->backup_now();
    REQUIRE(r2.uploaded == 0);
    REQUIRE(r2.generation == 2);

    // An orphan blob in the store is reclaimed by GC on the next push.
    write_file((fs::path(remote) / "objects" / "deadbeef").string(), "junk");
    engine->backup_now();
    REQUIRE_FALSE(fs::exists(fs::path(remote) / "objects" / "deadbeef"));
}

TEST_CASE("restore reproduces the tree and swaps the DB into a fresh dir",
          "[backup]") {
    std::string src_dir = temp_dir("src");
    std::string remote = temp_dir("remote");
    {
        SessionStore store(src_dir);
        seed_session(store, src_dir, "s1", "RESTORE-ME");
        make_engine(store, src_dir, remote)->backup_now();
    }

    // Fresh local: empty data dir + its own store, restore from the same remote.
    std::string dst_dir = temp_dir("dst");
    SessionStore store2(dst_dir);
    REQUIRE(store2.list().empty());
    auto engine2 = make_engine(store2, dst_dir, remote);
    int restored = engine2->restore();
    REQUIRE(restored >= 2);
    REQUIRE(read_file(
                (fs::path(dst_dir) / "sessions" / "s1" / "audio.wav").string()) ==
            "RESTORE-ME");
    REQUIRE_FALSE(store2.list().empty());          // DB swapped in
    REQUIRE_FALSE(store2.get("s1").is_null());     // the row came back
}

TEST_CASE("restore with prune deletes local files absent from the manifest",
          "[backup]") {
    std::string src_dir = temp_dir("src");
    std::string remote = temp_dir("remote");
    {
        SessionStore store(src_dir);
        seed_session(store, src_dir, "s1", "KEEP");
        make_engine(store, src_dir, remote)->backup_now();
    }
    // Local has an extra session the remote never saw.
    SessionStore store2(src_dir);
    write_file((fs::path(src_dir) / "sessions" / "ghost" / "audio.wav").string(),
               "ORPHAN");
    auto engine2 = make_engine(store2, src_dir, remote);
    engine2->restore(/*prune=*/true);
    REQUIRE_FALSE(
        fs::exists(fs::path(src_dir) / "sessions" / "ghost" / "audio.wav"));
    REQUIRE(fs::exists(fs::path(src_dir) / "sessions" / "s1" / "audio.wav"));
}

TEST_CASE("assess_link classifies fresh / remote-only / in-sync / diverged",
          "[backup]") {
    std::string data_dir = temp_dir("data");
    std::string remote = temp_dir("remote");
    SessionStore store(data_dir);
    seed_session(store, data_dir, "s1", "V1");
    auto engine = make_engine(store, data_dir, remote);

    // No remote manifest yet, local has data -> FRESH.
    REQUIRE(engine->assess_link().outcome == LinkOutcome::Fresh);

    engine->backup_now();  // seed the remote
    // Remote == local now -> IN_SYNC.
    REQUIRE(engine->assess_link().outcome == LinkOutcome::InSync);

    // Mutate local -> DIVERGED.
    write_file((fs::path(data_dir) / "sessions" / "s1" / "audio.wav").string(),
               "V2-DIFFERENT");
    REQUIRE(engine->assess_link().outcome == LinkOutcome::Diverged);

    // Empty local against a populated remote -> REMOTE_ONLY.
    std::string empty_dir = temp_dir("empty");
    SessionStore store_empty(empty_dir);
    auto engine_empty = make_engine(store_empty, empty_dir, remote);
    REQUIRE(engine_empty->assess_link().outcome == LinkOutcome::RemoteOnly);
}

TEST_CASE("DriveClient::upload_resumable shapes the resumable request",
          "[backup]") {
    using whisperx::server::drive::DriveClient;
    using whisperx::server::drive::File;
    using whisperx::server::net::Response;

    std::string seen_url, seen_auth, seen_meta, seen_src, seen_mime;
    DriveClient::UploadFn fake = [&](const std::string& url,
                                     const std::string& auth,
                                     const std::string& meta,
                                     const std::string& src,
                                     const std::string& mime) {
        seen_url = url;
        seen_auth = auth;
        seen_meta = meta;
        seen_src = src;
        seen_mime = mime;
        Response r;
        r.status = 200;
        r.body = R"({"id":"obj42"})";
        return r;
    };
    DriveClient c([] { return std::optional<std::string>("Bearer TOK"); }, {},
                  fake);
    c.set_endpoints("https://api/drive/v3", "https://api/upload/drive/v3");

    nlohmann::json meta = {{"name", "abc123"}, {"parents", {"objparent"}}};
    File f = c.upload_resumable(meta, "application/octet-stream",
                               "/tmp/blob.bin");
    REQUIRE(f.id == "obj42");
    REQUIRE(seen_url ==
            "https://api/upload/drive/v3/files?uploadType=resumable&fields=id");
    REQUIRE(seen_auth == "Bearer TOK");
    REQUIRE(seen_src == "/tmp/blob.bin");
    REQUIRE(seen_mime == "application/octet-stream");
    REQUIRE(nlohmann::json::parse(seen_meta)["name"] == "abc123");
}
