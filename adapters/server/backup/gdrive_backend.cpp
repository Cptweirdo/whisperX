#include "backup/gdrive_backend.hpp"

#include <stdexcept>
#include <utility>

#include "drive/drive_client.hpp"

namespace whisperx::server::backup {

using nlohmann::json;

namespace {
constexpr const char* kManifestName = "manifest.json";
constexpr const char* kObjectsFolder = "objects";
}  // namespace

GDriveBackend::GDriveBackend(DriveFactory factory, std::string folder_name)
    : factory_(std::move(factory)), folder_name_(std::move(folder_name)) {}

drive::DriveClient GDriveBackend::client() {
    auto c = factory_ ? factory_() : std::nullopt;
    if (!c) throw std::runtime_error("Google Drive is not linked");
    return *c;
}

bool GDriveBackend::is_linked() {
    return factory_ && factory_().has_value();
}

std::string GDriveBackend::root_id() {
    if (!root_id_) root_id_ = client().ensure_folder(folder_name_, "");
    return *root_id_;
}

std::string GDriveBackend::objects_id() {
    if (!objects_id_)
        objects_id_ = client().ensure_folder(kObjectsFolder, root_id());
    return *objects_id_;
}

void GDriveBackend::set_folder(const std::string& name) {
    std::string n = name;
    // trim
    size_t b = n.find_first_not_of(" \t\r\n");
    size_t e = n.find_last_not_of(" \t\r\n");
    n = (b == std::string::npos) ? std::string() : n.substr(b, e - b + 1);
    if (n.empty() || n == folder_name_) return;
    folder_name_ = n;
    root_id_.reset();
    objects_id_.reset();
}

std::optional<Manifest> GDriveBackend::read_manifest() {
    drive::DriveClient c = client();
    auto fid = c.find_child(kManifestName, root_id(), false);
    if (!fid) return std::nullopt;
    return Manifest::from_json(c.download(*fid));  // manifest is small
}

void GDriveBackend::write_manifest(const Manifest& manifest) {
    drive::DriveClient c = client();
    std::string body = manifest.to_json();
    auto existing = c.find_child(kManifestName, root_id(), false);
    if (existing) {  // single-request content replace == atomic server-side
        c.update_content(*existing, "application/json", body);
    } else {
        json meta = {{"name", kManifestName},
                     {"parents", json::array({root_id()})}};
        c.create_file(meta, "application/json", body);
    }
}

bool GDriveBackend::has_object(const std::string& key) {
    return client().find_child(key, objects_id(), false).has_value();
}

void GDriveBackend::put_object(const std::string& key,
                               const std::string& src_path) {
    if (has_object(key)) return;  // content-addressed => already identical
    json meta = {{"name", key}, {"parents", json::array({objects_id()})}};
    client().upload_resumable(meta, "application/octet-stream", src_path);
}

void GDriveBackend::get_object_to_file(const std::string& key,
                                       const std::string& dest) {
    drive::DriveClient c = client();
    auto fid = c.find_child(key, objects_id(), false);
    if (!fid) throw std::runtime_error("no such object: " + key);
    c.download_to_file(*fid, dest);
}

std::string GDriveBackend::get_object(const std::string& key) {
    drive::DriveClient c = client();
    auto fid = c.find_child(key, objects_id(), false);
    if (!fid) throw std::runtime_error("no such object: " + key);
    return c.download(*fid);
}

void GDriveBackend::delete_object(const std::string& key) {
    drive::DriveClient c = client();
    auto fid = c.find_child(key, objects_id(), false);
    if (fid) c.remove(*fid);
}

std::set<std::string> GDriveBackend::list_objects() {
    std::set<std::string> keys;
    drive::DriveClient c = client();
    std::string parent = objects_id();
    std::string q = "'" + drive::escape_query_value(parent) +
                    "' in parents and trashed = false";
    std::string token;
    do {
        auto page = c.list(q, "nextPageToken, files(name)", 1000, token);
        for (const auto& f : page.files) keys.insert(f.name);
        token = page.next_page_token;
    } while (!token.empty());
    return keys;
}

}  // namespace whisperx::server::backup
