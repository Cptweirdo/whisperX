#include "oauth/client.hpp"

#include <ctime>
#include <stdexcept>

#include "encoding/url.hpp"

namespace whisperx::server::oauth {

namespace {

std::string join_scopes(const std::vector<std::string>& scopes) {
    std::string s;
    for (const auto& sc : scopes) {
        if (!s.empty()) s += ' ';
        s += sc;
    }
    return s;
}

// Pull the standard token-endpoint fields out of a JSON response, applying them
// onto `creds` (refresh_token is only overwritten when present — Google omits it
// on refresh, and we must keep the original). Sets expiry from expires_in.
void apply_token_response(const json& j, Credentials& creds) {
    if (j.contains("access_token") && j["access_token"].is_string())
        creds.token = j["access_token"].get<std::string>();
    if (j.contains("refresh_token") && j["refresh_token"].is_string() &&
        !j["refresh_token"].get<std::string>().empty())
        creds.refresh_token = j["refresh_token"].get<std::string>();
    if (j.contains("expires_in") && j["expires_in"].is_number())
        creds.expiry =
            std::time(nullptr) + j["expires_in"].get<long>();
}

}  // namespace

OAuthClient::OAuthClient(ProviderConfig provider, TokenStore& store, PostFn post)
    : provider_(std::move(provider)), store_(store), post_(std::move(post)) {
    if (!post_)
        post_ = [](const std::string& url, const std::string& body) {
            return net::post(url, body);
        };
}

std::string OAuthClient::build_authorize_url(
    const std::string& state, const std::string& code_challenge) const {
    std::string body = net::form_encode({
        {"client_id", provider_.client_id},
        {"redirect_uri", provider_.redirect_uri},
        {"response_type", "code"},
        {"scope", join_scopes(provider_.scopes)},
        {"code_challenge", code_challenge},
        {"code_challenge_method", "S256"},
        {"state", state},
        {"access_type", "offline"},  // ask Google for a refresh token
        {"prompt", "consent"},
    });
    return provider_.auth_uri + "?" + body;
}

Credentials OAuthClient::exchange_code(const std::string& code,
                                       const std::string& code_verifier) {
    std::string body = net::form_encode({
        {"grant_type", "authorization_code"},
        {"code", code},
        {"client_id", provider_.client_id},
        {"client_secret", provider_.client_secret},
        {"redirect_uri", provider_.redirect_uri},
        {"code_verifier", code_verifier},
    });
    net::Response r = post_(provider_.token_uri, body);
    if (r.status == 0)
        throw std::runtime_error(
            "Couldn't reach the token endpoint. Check your connection.");
    if (!r.ok())
        throw std::runtime_error("Token exchange failed: HTTP " +
                                 std::to_string(r.status) + " " + r.body);
    json j;
    try {
        j = json::parse(r.body);
    } catch (...) {
        throw std::runtime_error("Token endpoint returned malformed JSON.");
    }
    Credentials creds;
    creds.token_uri = provider_.token_uri;
    creds.client_id = provider_.client_id;
    creds.client_secret = provider_.client_secret;
    creds.scopes = provider_.scopes;
    apply_token_response(j, creds);
    if (creds.token.empty())
        throw std::runtime_error("Token endpoint returned no access_token.");
    {
        std::lock_guard<std::mutex> lk(mu_);
        persist(creds);
    }
    return creds;
}

Credentials OAuthClient::refresh(Credentials creds) {
    std::string body = net::form_encode({
        {"grant_type", "refresh_token"},
        {"refresh_token", creds.refresh_token},
        {"client_id", provider_.client_id},
        {"client_secret", provider_.client_secret},
    });
    net::Response r = post_(provider_.token_uri, body);
    if (r.status == 0)
        throw std::runtime_error("Couldn't reach the token endpoint to refresh.");
    if (!r.ok())
        throw std::runtime_error("Token refresh failed: HTTP " +
                                 std::to_string(r.status) + " " + r.body);
    json j;
    try {
        j = json::parse(r.body);
    } catch (...) {
        throw std::runtime_error("Refresh returned malformed JSON.");
    }
    apply_token_response(j, creds);
    persist(creds);
    return creds;
}

std::optional<std::string> OAuthClient::access_token() {
    std::lock_guard<std::mutex> lk(mu_);
    auto raw = store_.load();
    if (!raw) return std::nullopt;
    Credentials creds;
    try {
        creds = Credentials::from_json(json::parse(*raw));
    } catch (...) {
        return std::nullopt;
    }
    if (creds.expired() && !creds.refresh_token.empty()) {
        try {
            creds = refresh(creds);
        } catch (...) {
            return std::nullopt;
        }
    }
    if (creds.token.empty()) return std::nullopt;
    return creds.token;
}

std::optional<std::string> OAuthClient::bearer_header() {
    return net::bearer_header(access_token());
}

bool OAuthClient::is_linked() {
    std::lock_guard<std::mutex> lk(mu_);
    auto raw = store_.load();
    if (!raw) return false;
    try {
        Credentials c = Credentials::from_json(json::parse(*raw));
        return !c.refresh_token.empty();
    } catch (...) {
        return false;
    }
}

void OAuthClient::unlink() {
    std::lock_guard<std::mutex> lk(mu_);
    store_.clear();
}

void OAuthClient::persist(const Credentials& creds) {
    store_.save(creds.to_json().dump());
}

}  // namespace whisperx::server::oauth
