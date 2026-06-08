#include "drive/drive_client.hpp"

#include <utility>

#include "encoding/url.hpp"
#include "oauth/crypto.hpp"

namespace whisperx::server::drive {

namespace {

std::string enc(const std::string& s) {
    encoding::Url::Config cfg;  // RFC 3986 unreserved; space -> %20
    return encoding::Url::encode(s, cfg);
}

File parse_file(const json& j) {
    File f;
    if (j.contains("id") && j["id"].is_string()) f.id = j["id"];
    if (j.contains("name") && j["name"].is_string()) f.name = j["name"];
    if (j.contains("mimeType") && j["mimeType"].is_string())
        f.mime_type = j["mimeType"];
    return f;
}

}  // namespace

std::string escape_query_value(const std::string& v) {
    std::string out;
    out.reserve(v.size());
    for (char c : v) {
        if (c == '\\' || c == '\'') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

DriveClient::DriveClient(AuthFn auth, Transport transport, UploadFn upload)
    : auth_(std::move(auth)),
      transport_(std::move(transport)),
      upload_(std::move(upload)) {
    if (!transport_)
        transport_ = [](const std::string& method, const std::string& url,
                        const std::vector<std::string>& headers,
                        const std::string& body) {
            net::Options o;
            o.headers = headers;
            if (method == "GET") return net::get(url, o);
            return net::request(method, url, body, o);
        };
    if (!upload_)
        upload_ = [](const std::string& init_url, const std::string& auth_value,
                     const std::string& metadata, const std::string& src_path,
                     const std::string& mime) {
            return net::upload_file(init_url, auth_value, metadata, src_path,
                                    mime);
        };
}

void DriveClient::set_endpoints(std::string api_base, std::string upload_base) {
    api_base_ = std::move(api_base);
    upload_base_ = std::move(upload_base);
}

std::string DriveClient::bearer() {
    auto h = auth_ ? auth_() : std::nullopt;
    if (!h || h->empty())
        throw DriveError(0, "Google Drive is not linked.");
    return *h;
}

json DriveClient::call_json(const std::string& method, const std::string& url,
                            const std::string& content_type,
                            const std::string& body) {
    std::vector<std::string> headers{"Authorization: " + bearer()};
    if (!content_type.empty())
        headers.push_back("Content-Type: " + content_type);
    net::Response r = transport_(method, url, headers, body);
    if (r.status == 0)
        throw DriveError(0, "Couldn't reach Google Drive.");
    if (r.status < 200 || r.status >= 300)
        throw DriveError(r.status, "Drive API error: HTTP " +
                                       std::to_string(r.status) + " " + r.body);
    if (r.body.empty()) return json::object();  // e.g. delete -> 204
    try {
        return json::parse(r.body);
    } catch (...) {
        throw DriveError(r.status, "Drive returned malformed JSON.");
    }
}

DriveClient::ListPage DriveClient::list(const std::string& query,
                                        const std::string& fields,
                                        int page_size,
                                        const std::string& page_token) {
    std::string url = api_base_ + "/files?spaces=drive&pageSize=" +
                      std::to_string(page_size) + "&fields=" + enc(fields);
    if (!query.empty()) url += "&q=" + enc(query);
    if (!page_token.empty()) url += "&pageToken=" + enc(page_token);
    json j = call_json("GET", url, "", "");
    ListPage page;
    if (j.contains("files") && j["files"].is_array())
        for (const auto& f : j["files"]) page.files.push_back(parse_file(f));
    if (j.contains("nextPageToken") && j["nextPageToken"].is_string())
        page.next_page_token = j["nextPageToken"];
    return page;
}

File DriveClient::get(const std::string& file_id, const std::string& fields) {
    std::string url = api_base_ + "/files/" + enc(file_id) +
                      "?fields=" + enc(fields);
    return parse_file(call_json("GET", url, "", ""));
}

std::string DriveClient::download(const std::string& file_id) {
    std::string url = api_base_ + "/files/" + enc(file_id) + "?alt=media";
    std::vector<std::string> headers{"Authorization: " + bearer()};
    net::Response r = transport_("GET", url, headers, "");
    if (r.status == 0) throw DriveError(0, "Couldn't reach Google Drive.");
    if (r.status < 200 || r.status >= 300)
        throw DriveError(r.status, "Drive download failed: HTTP " +
                                       std::to_string(r.status));
    return r.body;
}

void DriveClient::download_to_file(const std::string& file_id,
                                   const std::string& dest) {
    std::string url = api_base_ + "/files/" + enc(file_id) + "?alt=media";
    net::Options o;
    o.headers = {"Authorization: " + bearer()};
    long status = net::download_to_file(url, dest, o);
    if (status < 200 || status >= 300)
        throw DriveError(status, "Drive download failed: HTTP " +
                                     std::to_string(status));
}

File DriveClient::create_metadata(const json& metadata,
                                  const std::string& fields) {
    std::string url = api_base_ + "/files?fields=" + enc(fields);
    return parse_file(
        call_json("POST", url, "application/json", metadata.dump()));
}

std::string DriveClient::create_folder(const std::string& name,
                                       const std::string& parent_id) {
    json meta = {{"name", name},
                 {"mimeType", "application/vnd.google-apps.folder"}};
    if (!parent_id.empty()) meta["parents"] = json::array({parent_id});
    return create_metadata(meta, "id").id;
}

File DriveClient::create_file(const json& metadata, const std::string& mime_type,
                              const std::string& content,
                              const std::string& fields) {
    // multipart/related: a JSON metadata part then the raw content part.
    std::string boundary =
        "wxdrive" + oauth::base64url_nopad(oauth::random_bytes(16));
    std::string body;
    body += "--" + boundary + "\r\n";
    body += "Content-Type: application/json; charset=UTF-8\r\n\r\n";
    body += metadata.dump() + "\r\n";
    body += "--" + boundary + "\r\n";
    body += "Content-Type: " + mime_type + "\r\n\r\n";
    body += content + "\r\n";
    body += "--" + boundary + "--";
    std::string url =
        upload_base_ + "/files?uploadType=multipart&fields=" + enc(fields);
    return parse_file(call_json(
        "POST", url, "multipart/related; boundary=" + boundary, body));
}

File DriveClient::upload_resumable(const json& metadata,
                                  const std::string& mime_type,
                                  const std::string& src_path,
                                  const std::string& fields) {
    std::string url =
        upload_base_ + "/files?uploadType=resumable&fields=" + enc(fields);
    net::Response r =
        upload_(url, bearer(), metadata.dump(), src_path, mime_type);
    if (r.status == 0)
        throw DriveError(0, "Couldn't reach Google Drive (upload).");
    if (r.status < 200 || r.status >= 300)
        throw DriveError(r.status, "Drive upload failed: HTTP " +
                                       std::to_string(r.status) + " " + r.body);
    if (r.body.empty()) return File{};
    try {
        return parse_file(json::parse(r.body));
    } catch (...) {
        throw DriveError(r.status, "Drive returned malformed JSON (upload).");
    }
}

File DriveClient::update_content(const std::string& file_id,
                                 const std::string& mime_type,
                                 const std::string& content,
                                 const std::string& fields) {
    std::string url = upload_base_ + "/files/" + enc(file_id) +
                      "?uploadType=media&fields=" + enc(fields);
    return parse_file(call_json("PATCH", url, mime_type, content));
}

void DriveClient::remove(const std::string& file_id) {
    std::string url = api_base_ + "/files/" + enc(file_id);
    call_json("DELETE", url, "", "");  // 204, empty body tolerated
}

std::optional<std::string> DriveClient::find_child(const std::string& name,
                                                   const std::string& parent_id,
                                                   bool folder) {
    std::string q = "name = '" + escape_query_value(name) +
                    "' and trashed = false";
    if (folder) q += " and mimeType = 'application/vnd.google-apps.folder'";
    if (!parent_id.empty())
        q += " and '" + escape_query_value(parent_id) + "' in parents";
    ListPage page = list(q, "files(id,name)", 1);
    if (page.files.empty()) return std::nullopt;
    return page.files.front().id;
}

std::string DriveClient::ensure_folder(const std::string& name,
                                       const std::string& parent_id) {
    if (auto existing = find_child(name, parent_id, true)) return *existing;
    return create_folder(name, parent_id);
}

}  // namespace whisperx::server::drive
