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

// GET the URL streamed to `dest` (written to dest + ".part", then atomically
// renamed on a 2xx). Creates parent dirs. Returns the HTTP status (0 on a
// transport failure); the partial file is removed unless the status was 2xx.
long download_to_file(const std::string& url, const std::string& dest,
                      const Options& opts = {});

// "Authorization: Bearer <token>" when a non-empty token is present, else nullopt.
std::optional<std::string> bearer_header(
    const std::optional<std::string>& token);

}  // namespace whisperx::server::net
