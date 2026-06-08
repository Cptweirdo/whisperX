// DriveClient — a thin, reusable wrapper over the Google Drive v3 REST files.*
// endpoints (the native analog of app/backup/gdrive.py's googleapiclient calls):
//
//   files.list    GET    /drive/v3/files?q=...&fields=...
//   files.get     GET    /drive/v3/files/{id}?fields=...   (metadata)
//                 GET    /drive/v3/files/{id}?alt=media    (content)
//   files.create  POST   /drive/v3/files                   (metadata only, e.g. a folder)
//                 POST   /upload/drive/v3/files?uploadType=multipart  (metadata + content)
//   files.update  PATCH  /upload/drive/v3/files/{id}?uploadType=media (content replace)
//   files.delete  DELETE /drive/v3/files/{id}
//
// It carries no manifest / object-store logic — that backup engine sits on top.
// Auth is an injected callback returning a live "Authorization: Bearer <token>"
// (BackupService::bearer_header), so the client never owns token refresh. The
// HTTP transport is injectable too, so the request shaping (queries, multipart
// bodies, pagination, error mapping) is unit-testable without a network.
//
// Scope note: drive.file — files.list only ever returns files this app created.
#pragma once

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "http/curl_client.hpp"

namespace whisperx::server::drive {

using nlohmann::json;

// A Drive file as returned by list/get/create (only the fields we request).
struct File {
    std::string id;
    std::string name;
    std::string mime_type;
};

// Thrown on any non-2xx (or transport failure, status 0) Drive response.
class DriveError : public std::runtime_error {
public:
    DriveError(long status, const std::string& message)
        : std::runtime_error(message), status(status) {}
    long status;  // HTTP status, or 0 on a transport-level failure
};

class DriveClient {
public:
    // Returns a live "Authorization: Bearer <token>" header, or nullopt when not
    // linked (then calls throw DriveError(0, "not linked")).
    using AuthFn = std::function<std::optional<std::string>()>;
    // (method, url, headers, body) -> response. Defaults to net::* (curl).
    using Transport = std::function<net::Response(
        const std::string& method, const std::string& url,
        const std::vector<std::string>& headers, const std::string& body)>;

    explicit DriveClient(AuthFn auth, Transport transport = {});

    // Override the API/upload roots (tests point these at a fake host).
    void set_endpoints(std::string api_base, std::string upload_base);

    struct ListPage {
        std::vector<File> files;
        std::string next_page_token;
    };
    // files.list. `query` is the raw Drive `q` (e.g. "name = 'x' and trashed =
    // false"); pass an empty page_token for the first page.
    ListPage list(const std::string& query,
                  const std::string& fields = "files(id,name)",
                  int page_size = 100, const std::string& page_token = "");

    // files.get metadata.
    File get(const std::string& file_id,
             const std::string& fields = "id,name,mimeType");

    // files.get?alt=media — file content into memory.
    std::string download(const std::string& file_id);
    // files.get?alt=media streamed to `dest` (atomic .part rename).
    void download_to_file(const std::string& file_id, const std::string& dest);

    // files.create with metadata only (no content) — used to make folders.
    File create_metadata(const json& metadata,
                         const std::string& fields = "id");
    // Convenience: create (or just create) a folder; returns its id.
    std::string create_folder(const std::string& name,
                              const std::string& parent_id = "");

    // files.create multipart upload (metadata + content). mime_type is the
    // content type; metadata carries name/parents/etc.
    File create_file(const json& metadata, const std::string& mime_type,
                     const std::string& content,
                     const std::string& fields = "id");

    // files.update content replace (uploadType=media). Single-request atomic
    // content swap — what the manifest pointer uses.
    File update_content(const std::string& file_id, const std::string& mime_type,
                        const std::string& content,
                        const std::string& fields = "id");

    // files.delete.
    void remove(const std::string& file_id);

    // --- convenience helpers (mirror gdrive.py) ---------------------------
    // First child of `parent_id` named `name` (folder-typed if `folder`), or
    // nullopt. Empty parent_id means "don't constrain by parent".
    std::optional<std::string> find_child(const std::string& name,
                                          const std::string& parent_id,
                                          bool folder);
    // find_child(folder=true) else create_folder — returns the folder id.
    std::string ensure_folder(const std::string& name,
                              const std::string& parent_id);

private:
    // Issue a request with auth; parse JSON body; throw DriveError on non-2xx.
    json call_json(const std::string& method, const std::string& url,
                   const std::string& content_type, const std::string& body);
    std::string bearer();  // throws DriveError(0) if not linked

    AuthFn auth_;
    Transport transport_;
    std::string api_base_ = "https://www.googleapis.com/drive/v3";
    std::string upload_base_ = "https://www.googleapis.com/upload/drive/v3";
};

// Escape a string for use inside a Drive `q` single-quoted literal (\\ and ').
std::string escape_query_value(const std::string& v);

}  // namespace whisperx::server::drive
