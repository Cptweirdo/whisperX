// OAuth2 provider configuration — the small, injectable description of an
// endpoint pair + client credentials that makes OAuthClient provider-agnostic.
// google_drive_provider() wires it for Google (drive.file scope, loopback
// redirect on the running server's port), reading GOOGLE_CLIENT_ID /
// GOOGLE_CLIENT_SECRET from the environment exactly like the Python host
// (app/backup/oauth.py::_client_config).
#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace whisperx::server::oauth {

struct ProviderConfig {
    std::string auth_uri;
    std::string token_uri;
    std::string client_id;
    std::string client_secret;
    std::string redirect_uri;  // loopback, e.g. http://127.0.0.1:8000/oauth/callback
    std::vector<std::string> scopes;
};

// Whether the Google client credentials are provisioned (env set). Used to keep
// the settings/backup endpoints rendering even when backup isn't configured.
inline bool google_client_configured() {
    const char* id = std::getenv("GOOGLE_CLIENT_ID");
    const char* secret = std::getenv("GOOGLE_CLIENT_SECRET");
    return id && id[0] && secret && secret[0];
}

// Build the Google Drive provider config for a loopback redirect on `port`.
// Throws std::runtime_error if the client env vars aren't set (mirrors
// oauth.py::_client_config). Scope is drive.file (least privilege — the app only
// sees files it created).
inline ProviderConfig google_drive_provider(int port) {
    const char* id = std::getenv("GOOGLE_CLIENT_ID");
    const char* secret = std::getenv("GOOGLE_CLIENT_SECRET");
    if (!id || !id[0] || !secret || !secret[0])
        throw std::runtime_error(
            "GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET are not set — register a "
            "Google Cloud OAuth (Desktop app) client and put them in app/.env.");
    ProviderConfig p;
    p.auth_uri = "https://accounts.google.com/o/oauth2/auth";
    p.token_uri = "https://oauth2.googleapis.com/token";
    p.client_id = id;
    p.client_secret = secret;
    p.redirect_uri =
        "http://127.0.0.1:" + std::to_string(port) + "/oauth/callback";
    p.scopes = {"https://www.googleapis.com/auth/drive.file"};
    return p;
}

}  // namespace whisperx::server::oauth
