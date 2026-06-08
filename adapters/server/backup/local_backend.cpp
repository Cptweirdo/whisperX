#include "backup/local_backend.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace whisperx::server::backup {

LocalFsBackend::LocalFsBackend(const std::string& root)
    : root_(fs::absolute(root).string()),
      objects_dir_((fs::path(root_) / "objects").string()),
      manifest_path_((fs::path(root_) / "manifest.json").string()) {
    std::error_code ec;
    fs::create_directories(objects_dir_, ec);
}

bool LocalFsBackend::is_linked() {
    std::error_code ec;
    return fs::is_directory(root_, ec);
}

std::string LocalFsBackend::object_path(const std::string& key) const {
    return (fs::path(objects_dir_) / key).string();
}

std::optional<Manifest> LocalFsBackend::read_manifest() {
    std::ifstream f(manifest_path_, std::ios::binary);
    if (!f) return std::nullopt;
    std::ostringstream ss;
    ss << f.rdbuf();
    return Manifest::from_json(ss.str());
}

void LocalFsBackend::write_manifest(const Manifest& manifest) {
    std::string tmp = manifest_path_ + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("cannot write manifest: " + tmp);
        f << manifest.to_json();
    }
    std::error_code ec;
    fs::rename(tmp, manifest_path_, ec);  // atomic commit
    if (ec) throw std::runtime_error("manifest rename failed: " + ec.message());
}

bool LocalFsBackend::has_object(const std::string& key) {
    std::error_code ec;
    return fs::exists(object_path(key), ec);
}

void LocalFsBackend::put_object(const std::string& key,
                                const std::string& src_path) {
    std::string dest = object_path(key);
    std::error_code ec;
    if (fs::exists(dest, ec)) return;  // content-addressed => identical, skip
    std::string tmp = dest + ".tmp";
    fs::copy_file(src_path, tmp, fs::copy_options::overwrite_existing, ec);
    if (ec) throw std::runtime_error("put_object copy failed: " + ec.message());
    fs::rename(tmp, dest, ec);
    if (ec) {
        fs::remove(tmp);
        throw std::runtime_error("put_object rename failed: " + ec.message());
    }
}

void LocalFsBackend::get_object_to_file(const std::string& key,
                                        const std::string& dest) {
    std::string src = object_path(key);
    std::error_code ec;
    if (!fs::exists(src, ec)) throw std::runtime_error("no such object: " + key);
    fs::path dpath(dest);
    if (dpath.has_parent_path()) fs::create_directories(dpath.parent_path(), ec);
    std::string part = dest + ".part";
    fs::copy_file(src, part, fs::copy_options::overwrite_existing, ec);
    if (ec) throw std::runtime_error("get_object copy failed: " + ec.message());
    fs::rename(part, dest, ec);
    if (ec) {
        fs::remove(part);
        throw std::runtime_error("get_object rename failed: " + ec.message());
    }
}

std::string LocalFsBackend::get_object(const std::string& key) {
    std::ifstream f(object_path(key), std::ios::binary);
    if (!f) throw std::runtime_error("no such object: " + key);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void LocalFsBackend::delete_object(const std::string& key) {
    std::error_code ec;
    fs::remove(object_path(key), ec);
}

std::set<std::string> LocalFsBackend::list_objects() {
    std::set<std::string> keys;
    std::error_code ec;
    for (auto it = fs::directory_iterator(objects_dir_, ec);
         !ec && it != fs::directory_iterator(); it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        std::string name = it->path().filename().string();
        if (name.size() >= 4 && name.compare(name.size() - 4, 4, ".tmp") == 0)
            continue;
        keys.insert(name);
    }
    return keys;
}

}  // namespace whisperx::server::backup
