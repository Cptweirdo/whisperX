// Merkle manifest for incremental backups — native port of app/backup/manifest.py.
//
// A *manifest* is the snapshot of the data dir at one instant: a map of relative
// path -> (content hash, size, mtime). Backups are content-addressed — a file's
// blob is stored remotely under its sha256, so an unchanged file (immutable
// audio) is never re-uploaded, and the manifest is the merkle root naming the
// whole tree by its contents.
//
// Layout captured (paths relative to data_dir):
//   sessions.db                         (from the *snapshot*, never the live file)
//   sessions/<id>/audio.*               (immutable once written)
//   sessions/<id>/transcript.json       (+ .edits.json / .srt / .vtt / .txt)
//
// Cross-host contract: the JSON shape (field set + sorted entries) matches the
// Python host's Manifest.to_json(), so a remote manifest written by either host
// is readable by the other. Only path + hash drive merkle_root()/changed_paths(),
// so float mtime formatting differences across hosts are irrelevant (mtime is
// stored, never compared).
#pragma once

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace whisperx::server::backup {

inline constexpr int kManifestVersion = 1;

struct FileEntry {
    std::string hash;       // lowercase sha256 hex (the object-store key)
    long long size = 0;     // byte size
    double mtime = 0.0;     // unix seconds (informational; never compared)
};

struct Manifest {
    int version = kManifestVersion;
    int generation = 0;
    std::string created_at;                  // ISO-8601 UTC, seconds precision
    std::map<std::string, FileEntry> entries;  // sorted by logical path

    std::string to_json() const;
    static Manifest from_json(const std::string& raw);

    // The set of content hashes this manifest references (upload diffs / GC).
    std::set<std::string> object_keys() const;
};

// Build a manifest from a DB snapshot + the session-artifact tree. sessions.db is
// taken from db_snapshot_path (the consistent copy made under the store lock) but
// recorded under the logical key "sessions.db"; everything under
// data_dir/sessions/ is included as-is (forward-slash logical paths).
Manifest build_local_manifest(const std::string& db_snapshot_path,
                              const std::string& data_dir, int generation = 0);

// A single hash over (sorted path, content hash) pairs — cheap equality check for
// "is the remote identical to local?" without diffing every entry.
std::string merkle_root(const Manifest& m);

// Paths whose content hash is new vs the remote manifest (need upload), sorted.
std::vector<std::string> changed_paths(const Manifest& local,
                                       const std::optional<Manifest>& remote);

// A fast tree fingerprint from (path, size, mtime_ns) — no content hashing. Used
// for dirty detection: if it matches the signature at the last successful push,
// nothing changed and the backup is skipped. Per-device only (compared to itself).
std::string cheap_signature(const std::string& data_dir);

}  // namespace whisperx::server::backup
