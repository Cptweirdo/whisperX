// Google Drive StorageBackend — native port of app/backup/gdrive.py.
//
// Maps the object-store contract onto Drive:
//   <backup folder>/manifest.json
//   <backup folder>/objects/<sha256>
//
// Uses the drive.file scope, so files.list only ever returns files this app
// created — the folder lookup can't see (or clobber) the user's other data.
//
// A drive::DriveClient is obtained lazily per call via an injected factory
// (BackupService::drive()), so the backend never owns token refresh: each op
// re-fetches a live-token-bound client. Object puts stream from disk
// (upload_resumable); gets stream to disk (download_to_file); only the small
// manifest moves through memory.
#pragma once

#include <functional>
#include <optional>
#include <set>
#include <string>

#include "backup/backend.hpp"
#include "drive/drive_client.hpp"

namespace whisperx::server::backup {

class GDriveBackend : public StorageBackend {
public:
    using DriveFactory = std::function<std::optional<drive::DriveClient>()>;

    explicit GDriveBackend(DriveFactory factory,
                           std::string folder_name = "Manuscript Backup");

    std::string name() const override { return "gdrive"; }
    bool is_linked() override;

    std::optional<Manifest> read_manifest() override;
    void write_manifest(const Manifest& manifest) override;

    bool has_object(const std::string& key) override;
    void put_object(const std::string& key, const std::string& src_path) override;
    void get_object_to_file(const std::string& key,
                            const std::string& dest) override;
    std::string get_object(const std::string& key) override;
    void delete_object(const std::string& key) override;
    std::set<std::string> list_objects() override;

    void set_folder(const std::string& name) override;

private:
    drive::DriveClient client();   // throws if not linked
    std::string root_id();         // ensure + cache the backup folder
    std::string objects_id();      // ensure + cache <folder>/objects

    DriveFactory factory_;
    std::string folder_name_;
    std::optional<std::string> root_id_;
    std::optional<std::string> objects_id_;
};

}  // namespace whisperx::server::backup
