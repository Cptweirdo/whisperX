#include "http/curl_client.hpp"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <system_error>

#include <curl/curl.h>

#include "encoding/url.hpp"

namespace fs = std::filesystem;

namespace whisperx::server::net {

namespace {

size_t write_to_string(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* out = static_cast<std::string*>(userdata);
    out->append(ptr, size * nmemb);
    return size * nmemb;
}

size_t write_to_file(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* fp = static_cast<std::FILE*>(userdata);
    return std::fwrite(ptr, size, nmemb, fp) * size;
}

size_t read_from_file(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* fp = static_cast<std::FILE*>(userdata);
    return std::fread(ptr, size, nmemb, fp);
}

// Header callback that captures the `Location:` response header into a string*.
size_t capture_location(char* buffer, size_t size, size_t nitems,
                        void* userdata) {
    size_t len = size * nitems;
    auto* loc = static_cast<std::string*>(userdata);
    static const char* key = "location:";
    if (len >= 9) {
        std::string head(buffer, 9);
        for (char& c : head) c = static_cast<char>(std::tolower((unsigned char)c));
        if (head == key) {
            std::string val(buffer + 9, len - 9);
            size_t b = val.find_first_not_of(" \t");
            size_t e = val.find_last_not_of(" \t\r\n");
            if (b != std::string::npos)
                *loc = val.substr(b, e - b + 1);
        }
    }
    return len;
}

// Apply the common easy-handle options (headers, timeout, redirects, UA).
// Caller owns freeing the returned slist after curl_easy_perform.
curl_slist* apply_common(CURL* h, const Options& opts) {
    curl_slist* list = nullptr;
    for (const auto& hdr : opts.headers) list = curl_slist_append(list, hdr.c_str());
    if (list) curl_easy_setopt(h, CURLOPT_HTTPHEADER, list);
    curl_easy_setopt(h, CURLOPT_TIMEOUT, opts.timeout_s);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 20L);
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, opts.follow_redirects ? 1L : 0L);
    curl_easy_setopt(h, CURLOPT_USERAGENT, "whisperx-server/1.0");
    curl_easy_setopt(h, CURLOPT_FAILONERROR, 0L);  // we read the status ourselves
    return list;
}

}  // namespace

Response get(const std::string& url, const Options& opts) {
    Response r;
    CURL* h = curl_easy_init();
    if (!h) return r;
    curl_easy_setopt(h, CURLOPT_URL, url.c_str());
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_to_string);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &r.body);
    curl_slist* list = apply_common(h, opts);
    CURLcode rc = curl_easy_perform(h);
    if (rc == CURLE_OK)
        curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &r.status);
    if (list) curl_slist_free_all(list);
    curl_easy_cleanup(h);
    return r;
}

Response post(const std::string& url, const std::string& body,
              const Options& opts) {
    Response r;
    CURL* h = curl_easy_init();
    if (!h) return r;
    // Default the content type unless the caller already set one.
    Options o = opts;
    bool has_ct = false;
    for (const auto& hdr : o.headers) {
        std::string lower = hdr.substr(0, hdr.find(':'));
        for (char& ch : lower) ch = static_cast<char>(std::tolower((unsigned char)ch));
        if (lower == "content-type") { has_ct = true; break; }
    }
    if (!has_ct)
        o.headers.push_back("Content-Type: application/x-www-form-urlencoded");

    curl_easy_setopt(h, CURLOPT_URL, url.c_str());
    curl_easy_setopt(h, CURLOPT_POST, 1L);
    curl_easy_setopt(h, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(h, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_to_string);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &r.body);
    curl_slist* list = apply_common(h, o);
    CURLcode rc = curl_easy_perform(h);
    if (rc == CURLE_OK)
        curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &r.status);
    if (list) curl_slist_free_all(list);
    curl_easy_cleanup(h);
    return r;
}

Response request(const std::string& method, const std::string& url,
                 const std::string& body, const Options& opts) {
    Response r;
    CURL* h = curl_easy_init();
    if (!h) return r;
    curl_easy_setopt(h, CURLOPT_URL, url.c_str());
    curl_easy_setopt(h, CURLOPT_CUSTOMREQUEST, method.c_str());
    if (method != "GET" && method != "HEAD") {
        curl_easy_setopt(h, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(h, CURLOPT_POSTFIELDSIZE,
                         static_cast<long>(body.size()));
    }
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_to_string);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &r.body);
    curl_slist* list = apply_common(h, opts);
    CURLcode rc = curl_easy_perform(h);
    if (rc == CURLE_OK) curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &r.status);
    if (list) curl_slist_free_all(list);
    curl_easy_cleanup(h);
    return r;
}

std::string form_encode(
    const std::vector<std::pair<std::string, std::string>>& fields) {
    encoding::Url::Config cfg;
    cfg.spaceToPlus = true;
    std::string out;
    for (const auto& [k, v] : fields) {
        if (!out.empty()) out += '&';
        out += encoding::Url::encode(k, cfg);
        out += '=';
        out += encoding::Url::encode(v, cfg);
    }
    return out;
}

long download_to_file(const std::string& url, const std::string& dest,
                      const Options& opts) {
    std::error_code ec;
    fs::path dpath(dest);
    if (dpath.has_parent_path()) fs::create_directories(dpath.parent_path(), ec);
    std::string part = dest + ".part";

    std::FILE* fp = std::fopen(part.c_str(), "wb");
    if (!fp) return 0;

    CURL* h = curl_easy_init();
    if (!h) {
        std::fclose(fp);
        std::remove(part.c_str());
        return 0;
    }
    curl_easy_setopt(h, CURLOPT_URL, url.c_str());
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_to_file);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, fp);
    curl_slist* list = apply_common(h, opts);
    CURLcode rc = curl_easy_perform(h);
    long status = 0;
    if (rc == CURLE_OK) curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    if (list) curl_slist_free_all(list);
    curl_easy_cleanup(h);
    std::fclose(fp);

    if (rc == CURLE_OK && status >= 200 && status < 300) {
        fs::rename(part, dest, ec);
        if (ec) {  // cross-device or race — copy then drop the temp
            fs::copy_file(part, dest, fs::copy_options::overwrite_existing, ec);
            std::remove(part.c_str());
        }
    } else {
        std::remove(part.c_str());
    }
    return status;
}

Response upload_file(const std::string& init_url, const std::string& auth_value,
                     const std::string& metadata, const std::string& src_path,
                     const std::string& content_mime) {
    Response r;
    std::error_code ec;
    std::uintmax_t fsize = fs::file_size(src_path, ec);
    if (ec) return r;  // status 0 — can't size the source

    // 1. Initiate the resumable session; capture the session URI from Location.
    std::string location;
    {
        CURL* h = curl_easy_init();
        if (!h) return r;
        Options o;
        o.follow_redirects = false;  // Location is the PUT target, not a redirect
        o.headers = {"Authorization: " + auth_value,
                     "Content-Type: application/json; charset=UTF-8",
                     "X-Upload-Content-Type: " + content_mime,
                     "X-Upload-Content-Length: " + std::to_string(fsize)};
        curl_easy_setopt(h, CURLOPT_URL, init_url.c_str());
        curl_easy_setopt(h, CURLOPT_POST, 1L);
        curl_easy_setopt(h, CURLOPT_POSTFIELDS, metadata.c_str());
        curl_easy_setopt(h, CURLOPT_POSTFIELDSIZE,
                         static_cast<long>(metadata.size()));
        curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, capture_location);
        curl_easy_setopt(h, CURLOPT_HEADERDATA, &location);
        curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_to_string);
        curl_easy_setopt(h, CURLOPT_WRITEDATA, &r.body);
        curl_slist* list = apply_common(h, o);
        CURLcode rc = curl_easy_perform(h);
        long status = 0;
        if (rc == CURLE_OK) curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
        if (list) curl_slist_free_all(list);
        curl_easy_cleanup(h);
        if (rc != CURLE_OK || status != 200 || location.empty()) {
            r.status = status;  // surface the init failure (0/4xx/5xx)
            return r;
        }
    }

    // 2. Stream the file body to the session URI in one PUT.
    std::FILE* fp = std::fopen(src_path.c_str(), "rb");
    if (!fp) return r;  // status still 0
    r.body.clear();
    CURL* h = curl_easy_init();
    if (!h) {
        std::fclose(fp);
        return r;
    }
    Options o;
    o.timeout_s = 0;  // no overall timeout — large blobs over slow links
    o.headers = {"Authorization: " + auth_value,
                 "Content-Type: " + content_mime};
    curl_easy_setopt(h, CURLOPT_URL, location.c_str());
    curl_easy_setopt(h, CURLOPT_UPLOAD, 1L);
    curl_easy_setopt(h, CURLOPT_READFUNCTION, read_from_file);
    curl_easy_setopt(h, CURLOPT_READDATA, fp);
    curl_easy_setopt(h, CURLOPT_INFILESIZE_LARGE,
                     static_cast<curl_off_t>(fsize));
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_to_string);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &r.body);
    curl_slist* list = apply_common(h, o);
    CURLcode rc = curl_easy_perform(h);
    if (rc == CURLE_OK) curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &r.status);
    if (list) curl_slist_free_all(list);
    curl_easy_cleanup(h);
    std::fclose(fp);
    return r;
}

std::optional<std::string> bearer_header(
    const std::optional<std::string>& token) {
    if (token && !token->empty()) return "Authorization: Bearer " + *token;
    return std::nullopt;
}

}  // namespace whisperx::server::net
