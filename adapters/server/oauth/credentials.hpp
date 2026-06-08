// OAuth2 credentials — the on-disk (keyring) shape, deliberately byte-compatible
// with Python's google.oauth2.credentials.Credentials.to_json() so a keyring
// entry written by either the Python or the native host is interchangeable
// (secrets/keyring.hpp key GDRIVE_CREDS). We persist the same field set Python's
// from_authorized_user_info() consumes; extra keys are tolerated on read.
#pragma once

#include <ctime>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace whisperx::server::oauth {

using nlohmann::json;

struct Credentials {
    std::string token;          // short-lived access token
    std::string refresh_token;  // long-lived; may be empty if never granted
    std::string token_uri;
    std::string client_id;
    std::string client_secret;
    std::vector<std::string> scopes;
    std::time_t expiry = 0;  // unix UTC seconds; 0 = unknown

    bool valid() const { return !token.empty(); }
    // Expired (or about to be) — `skew` seconds of safety margin. Unknown expiry
    // (0) is treated as not-expired (we only refresh when we know it's stale).
    bool expired(int skew = 60) const {
        return expiry != 0 && std::time(nullptr) + skew >= expiry;
    }

    static Credentials from_json(const json& j);
    json to_json() const;
};

// Format / parse the `expiry` field in Python's style:
// "YYYY-MM-DDTHH:MM:SS.ffffffZ" (UTC). parse tolerates a missing fractional part.
std::string format_expiry(std::time_t t);
std::time_t parse_expiry(const std::string& s);

}  // namespace whisperx::server::oauth
