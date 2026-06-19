// Minimal libcurl wrapper shared by the model downloader (assets/downloader) and
// the HF token verifier (secrets/hf_verify). No global state beyond
// curl_global_init/cleanup, which main.cpp owns. Synchronous/blocking — callers
// run it off the request hot path (background warm thread / job worker).
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace whisperx::server::net {

struct Response {
    long status = 0;  // HTTP status, or 0 on a transport-level failure
    std::string body;
    bool ok() const { return status >= 200 && status < 300; }
};

struct Options {
    std::vector<std::string> headers;  // each "Header: value"
    long timeout_s = 60;
    bool follow_redirects = true;  // HF resolve/ URLs 302 to a CDN
};

// GET the URL into memory. status == 0 means the request never completed.
Response get(const std::string& url, const Options& opts = {});

// POST `body` into memory. Defaults the Content-Type to
// application/x-www-form-urlencoded when the caller didn't set one in
// opts.headers. status == 0 means the request never completed.
Response post(const std::string& url, const std::string& body,
              const Options& opts = {});

// application/x-www-form-urlencoded body from key/value pairs (percent-encoded,
// space -> '+'), reusing the RFC-3986 codec in encoding/url.
std::string form_encode(const std::vector<std::pair<std::string, std::string>>&
                            fields);

// Generic request with an arbitrary verb (PATCH, DELETE, PUT, GET, POST) via
// CURLOPT_CUSTOMREQUEST — the body is sent for non-GET methods. Unlike post() no
// Content-Type is added; the caller sets it in opts.headers when there's a body.
// Used by the Drive client (PATCH content, DELETE files).
Response request(const std::string& method, const std::string& url,
                 const std::string& body, const Options& opts = {});

// GET the URL streamed to `dest` (written to dest + ".part", then atomically
// renamed on a 2xx). Creates parent dirs. Returns the HTTP status (0 on a
// transport failure); the partial file is removed unless the status was 2xx.
long download_to_file(const std::string& url, const std::string& dest,
                      const Options& opts = {});

// Streamed resumable upload (Google "uploadType=resumable"): a single PUT of the
// whole file, so a 400 MB blob never sits in memory. Two requests:
//   1. POST `init_url` with the JSON `metadata` body + X-Upload-Content-{Type,
//      Length} headers; the 200 response's `Location` header is the session URI.
//   2. PUT the session URI streaming `src_path` from disk (CURLOPT_READFUNCTION);
//      the 200/201 body is the file-resource JSON.
// `auth_value` is the Authorization value (e.g. "Bearer xyz"); `content_mime` is
// the blob's Content-Type. Returns the final {status, body}; status 0 on a
// transport failure or a missing session Location.
Response upload_file(const std::string& init_url, const std::string& auth_value,
                     const std::string& metadata, const std::string& src_path,
                     const std::string& content_mime);

// "Authorization: Bearer <token>" when a non-empty token is present, else nullopt.
std::optional<std::string> bearer_header(
    const std::optional<std::string>& token);

}  // namespace whisperx::server::net
