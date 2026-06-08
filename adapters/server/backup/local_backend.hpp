// Filesystem reference backend — native port of app/backup/local.py.
//
// Implements StorageBackend against a local directory. Drives the offline engine
// tests (no network, no Google) and doubles as a "backup to another disk /
// mounted share" option (WHISPERX_BACKUP_BACKEND=local). On-disk layout is exactly
// the remote contract:
//
//   <root>/manifest.json
//   <root>/objects/<sha256>
#pragma once

#include <optional>
#include <set>
#include <string>

#include "backup/backend.hpp"

namespace whisperx::server::backup {

class LocalFsBackend : public StorageBackend {
public:
    explicit LocalFsBackend(const std::string& root);

    std::string name() const override { return "local"; }
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

private:
    std::string object_path(const std::string& key) const;

    std::string root_;
    std::string objects_dir_;
    std::string manifest_path_;
};

}  // namespace whisperx::server::backup
