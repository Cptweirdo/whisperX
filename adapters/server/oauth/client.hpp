// OAuthClient — the token lifecycle for an auth-code + PKCE flow against one
// ProviderConfig. It builds the authorization URL, exchanges the code for
// credentials, refreshes them when stale, and hands back a live access token /
// bearer header. Persistence goes through an injected TokenStore; the outbound
// HTTP POST goes through an injectable transport (defaulting to net::post) so the
// refresh/exchange logic is unit-testable without a network. Thread-safe:
// access_token() may be called from request threads while the link flow runs.
//
// This is the token half only — it knows nothing about the Drive API. A future
// Drive client consumes bearer_header() to make files.* calls.
#pragma once

#include <functional>
#include <mutex>
#include <optional>
#include <string>

#include "http/curl_client.hpp"
#include "oauth/credentials.hpp"
#include "oauth/provider.hpp"
#include "oauth/token_store.hpp"

namespace whisperx::server::oauth {

class OAuthClient {
public:
    // Outbound transport: (url, x-www-form-urlencoded body) -> response. Defaults
    // to net::post; tests inject a fake.
    using PostFn = std::function<net::Response(const std::string& url,
                                               const std::string& body)>;

    OAuthClient(ProviderConfig provider, TokenStore& store, PostFn post = {});

    const ProviderConfig& provider() const { return provider_; }

    // The URL to send the user's browser to (with our PKCE challenge + state).
    std::string build_authorize_url(const std::string& state,
                                    const std::string& code_challenge) const;

    // Exchange an authorization code for credentials, verifying with the PKCE
    // verifier. Persists on success. Throws std::runtime_error on HTTP/parse
    // failure.
    Credentials exchange_code(const std::string& code,
                              const std::string& code_verifier);

    // A live access token: loads stored creds, refreshes if expired, persists the
    // rotation, returns the token. nullopt if not linked. Mutex-guarded.
    std::optional<std::string> access_token();

    // "Authorization: Bearer <token>" for a live token, or nullopt.
    std::optional<std::string> bearer_header();

    bool is_linked();   // a refresh token is on file
    void unlink();      // forget stored creds

private:
    Credentials refresh(Credentials creds);  // POST refresh_token; persists
    void persist(const Credentials& creds);

    ProviderConfig provider_;
    TokenStore& store_;
    PostFn post_;
    std::mutex mu_;
};

}  // namespace whisperx::server::oauth
