#include "backup/manifest.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <utility>

#include "oauth/crypto.hpp"
#include "time_iso.hpp"

namespace fs = std::filesystem;

namespace whisperx::server::backup {

using nlohmann::json;

namespace {

// POSIX stat so size + mtime match the Python oracle's os.stat semantics.
struct StatInfo {
    bool ok = false;
    long long size = 0;
    double mtime = 0.0;        // seconds (st_mtim.tv_sec + tv_nsec/1e9)
    long long mtime_ns = 0;    // st_mtim.tv_sec*1e9 + tv_nsec
};

StatInfo stat_info(const std::string& path) {
    StatInfo s;
#if defined(_WIN32)
    // No st_mtim on Windows; std::filesystem keeps the sub-second precision
    // (FILETIME is 100 ns) that Python's os.stat reports there.
    std::error_code ec;
    fs::path p(path);
    auto size = fs::file_size(p, ec);
    if (ec) return s;
    auto ftime = fs::last_write_time(p, ec);
    if (ec) return s;
    auto sys = std::chrono::clock_cast<std::chrono::system_clock>(ftime);
    long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       sys.time_since_epoch())
                       .count();
    s.ok = true;
    s.size = static_cast<long long>(size);
    s.mtime = static_cast<double>(ns) / 1e9;
    s.mtime_ns = ns;
#else
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) return s;
    s.ok = true;
    s.size = static_cast<long long>(st.st_size);
#if defined(__APPLE__)
    long long sec = st.st_mtimespec.tv_sec, nsec = st.st_mtimespec.tv_nsec;
#else
    long long sec = st.st_mtim.tv_sec, nsec = st.st_mtim.tv_nsec;
#endif
    s.mtime = static_cast<double>(sec) + static_cast<double>(nsec) / 1e9;
    s.mtime_ns = sec * 1000000000LL + nsec;
#endif
    return s;
}

// "<data_dir>/sessions/<...>" relative files, yielding (abs, logical) with a
// forward-slash logical path rooted at "sessions/".
std::vector<std::pair<std::string, std::string>> iter_sessions(
    const std::string& data_dir) {
    std::vector<std::pair<std::string, std::string>> out;
    fs::path root = fs::path(data_dir) / "sessions";
    std::error_code ec;
    if (!fs::is_directory(root, ec)) return out;
    for (auto it = fs::recursive_directory_iterator(
             root, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        std::string logical =
            ("sessions" / fs::relative(it->path(), root, ec)).generic_string();
        out.emplace_back(it->path().string(), logical);
    }
    return out;
}

}  // namespace

std::string Manifest::to_json() const {
    // ordered_json preserves the Python field order (version, generation,
    // created_at, entries; each entry hash, size, mtime). std::map keeps entries
    // sorted by logical path, matching Python's sorted(entries.items()).
    nlohmann::ordered_json j;
    j["version"] = version;
    j["generation"] = generation;
    j["created_at"] = created_at;
    nlohmann::ordered_json ents = nlohmann::ordered_json::object();
    for (const auto& [path, e] : entries) {
        nlohmann::ordered_json ej;
        ej["hash"] = e.hash;
        ej["size"] = e.size;
        ej["mtime"] = e.mtime;
        ents[path] = ej;
    }
    j["entries"] = ents;
    return j.dump();
}

Manifest Manifest::from_json(const std::string& raw) {
    json d = json::parse(raw);
    Manifest m;
    m.version = d.value("version", kManifestVersion);
    m.generation = d.value("generation", 0);
    m.created_at = d.value("created_at", std::string());
    if (d.contains("entries") && d["entries"].is_object()) {
        for (const auto& [path, e] : d["entries"].items()) {
            FileEntry fe;
            fe.hash = e.value("hash", std::string());
            fe.size = e.value("size", 0LL);
            fe.mtime = e.value("mtime", 0.0);
            m.entries[path] = fe;
        }
    }
    return m;
}

std::set<std::string> Manifest::object_keys() const {
    std::set<std::string> keys;
    for (const auto& [_, e] : entries) keys.insert(e.hash);
    return keys;
}

Manifest build_local_manifest(const std::string& db_snapshot_path,
                              const std::string& data_dir, int generation) {
    Manifest m;
    m.version = kManifestVersion;
    m.generation = generation;
    m.created_at = whisperx::now_iso();

    StatInfo dbs = stat_info(db_snapshot_path);
    m.entries["sessions.db"] =
        FileEntry{oauth::sha256_hex_file(db_snapshot_path), dbs.size, dbs.mtime};

    for (const auto& [ap, logical] : iter_sessions(data_dir)) {
        StatInfo s = stat_info(ap);
        m.entries[logical] =
            FileEntry{oauth::sha256_hex_file(ap), s.size, s.mtime};
    }
    return m;
}

std::string merkle_root(const Manifest& m) {
    oauth::Sha256 ctx;
    const std::string nul(1, '\0');
    for (const auto& [path, e] : m.entries) {  // std::map => sorted
        ctx.update(path);
        ctx.update(nul);
        ctx.update(e.hash);
        ctx.update(nul);
    }
    auto d = ctx.finish();
    static const char* H = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (auto b : d) {
        out += H[b >> 4];
        out += H[b & 0xf];
    }
    return out;
}

std::vector<std::string> changed_paths(const Manifest& local,
                                       const std::optional<Manifest>& remote) {
    std::set<std::string> remote_keys =
        remote ? remote->object_keys() : std::set<std::string>{};
    std::vector<std::string> out;
    for (const auto& [path, e] : local.entries)  // sorted by path already
        if (!remote_keys.count(e.hash)) out.push_back(path);
    return out;
}

std::string cheap_signature(const std::string& data_dir) {
    // Collect (abs, logical) for the DB + the sessions tree, sorted by logical.
    std::vector<std::pair<std::string, std::string>> targets;
    std::string db = (fs::path(data_dir) / "sessions.db").string();
    if (fs::exists(db)) targets.emplace_back(db, "sessions.db");
    auto sess = iter_sessions(data_dir);
    targets.insert(targets.end(), sess.begin(), sess.end());
    std::sort(targets.begin(), targets.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    oauth::Sha256 ctx;
    for (const auto& [ap, logical] : targets) {
        StatInfo s = stat_info(ap);
        if (!s.ok) continue;
        std::string rec = logical + '\0' + std::to_string(s.size) + '\0' +
                          std::to_string(s.mtime_ns) + '\0';
        ctx.update(rec);
    }
    auto d = ctx.finish();
    static const char* H = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (auto b : d) {
        out += H[b >> 4];
        out += H[b & 0xf];
    }
    return out;
}

}  // namespace whisperx::server::backup
