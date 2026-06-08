// The pluggable storage-backend interface — native port of app/backup/backend.py.
//
// A backend is a dumb content-addressed object store plus one mutable pointer
// (the manifest). All sync intelligence — snapshotting, diffing, GC — lives in
// BackupEngine; a backend only moves bytes. This keeps Google Drive swappable for
// a local folder (or S3/WebDAV later).
//
// Remote layout a backend must present:
//   manifest.json        # the merkle root; written LAST as the commit point
//   objects/<sha256>     # immutable content-addressed blobs
//
// Objects are 300-400 MB audio blobs, so the object I/O is by *path*: put_object
// streams a local file up, get_object_to_file streams a blob down to disk. Only
// the small manifest is moved through memory (read_manifest / get_object).
//
// write_manifest MUST be atomic (write-then-rename) so a reader never sees a
// half-written manifest — the manifest swap is what makes a backup "land".
#pragma once

#include <optional>
#include <set>
#include <string>

#include "backup/manifest.hpp"

namespace whisperx::server::backup {

class StorageBackend {
public:
    virtual ~StorageBackend() = default;

    // Short identifier for logs / status (e.g. "gdrive", "local").
    virtual std::string name() const = 0;

    // Whether usable credentials / a reachable target exist right now.
    virtual bool is_linked() = 0;

    // The current remote manifest, or nullopt if the target is fresh/empty.
    virtual std::optional<Manifest> read_manifest() = 0;
    // Alias used during bootstrap-on-link.
    virtual std::optional<Manifest> probe() { return read_manifest(); }

    // Atomically replace the remote manifest (write-then-rename).
    virtual void write_manifest(const Manifest& manifest) = 0;

    // Cheap existence check so an unchanged blob is never re-uploaded.
    virtual bool has_object(const std::string& key) = 0;
    // Stream the file at src_path up, stored under content hash key (idempotent).
    virtual void put_object(const std::string& key,
                            const std::string& src_path) = 0;
    // Stream a blob down to dest (atomic). Throws if the key is absent.
    virtual void get_object_to_file(const std::string& key,
                                    const std::string& dest) = 0;
    // Fetch a small blob (manifest-sized) into memory. Throws if absent.
    virtual std::string get_object(const std::string& key) = 0;
    // Remove a blob (no-op if absent). Used by GC.
    virtual void delete_object(const std::string& key) = 0;
    // Every stored blob key — so GC can find orphans the manifest dropped.
    virtual std::set<std::string> list_objects() = 0;

    // Re-target at a different destination folder/name. No-op by default;
    // backends with a user-chosen destination (Drive) override this.
    virtual void set_folder(const std::string& name) { (void)name; }
};

}  // namespace whisperx::server::backup
