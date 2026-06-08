// Catch2 tests for the hand-rolled OAuth2 (auth-code + PKCE) lib — no network.
// Pins the PKCE S256 transform to the RFC 7636 Appendix-B vector, checks the
// Python-compatible credentials JSON, the authorize URL contents, and drives
// OAuthClient::access_token() through a fake transport + in-memory token store to
// exercise refresh + refresh-token retention.
#include <catch2/catch_test_macros.hpp>

#include <cctype>
#include <ctime>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "http/curl_client.hpp"
#include "oauth/client.hpp"
#include "oauth/credentials.hpp"
#include "oauth/crypto.hpp"
#include "oauth/pkce.hpp"
#include "oauth/provider.hpp"
#include "oauth/token_store.hpp"

using namespace whisperx::server::oauth;
using nlohmann::json;

namespace {

struct MemStore : TokenStore {
    std::optional<std::string> value;
    std::optional<std::string> load() override { return value; }
    void save(const std::string& v) override { value = v; }
    void clear() override { value.reset(); }
};

ProviderConfig test_provider() {
    ProviderConfig p;
    p.auth_uri = "https://accounts.example/auth";
    p.token_uri = "https://oauth.example/token";
    p.client_id = "cid.apps";
    p.client_secret = "secret";
    p.redirect_uri = "http://127.0.0.1:8000/oauth/callback";
    p.scopes = {"https://www.googleapis.com/auth/drive.file"};
    return p;
}

}  // namespace

TEST_CASE("PKCE S256 matches RFC 7636 Appendix-B vector", "[oauth][pkce]") {
    const std::string verifier =
        "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
    REQUIRE(code_challenge_s256(verifier) ==
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
}

TEST_CASE("base64url uses the url-safe alphabet, no padding", "[oauth][crypto]") {
    // 0xFB 0xFF -> '+/' in std base64; '-_' in url-safe; no '=' padding.
    std::string in;
    in.push_back(static_cast<char>(0xFB));
    in.push_back(static_cast<char>(0xFF));
    std::string out = base64url_nopad(in);
    REQUIRE(out.find('+') == std::string::npos);
    REQUIRE(out.find('/') == std::string::npos);
    REQUIRE(out.find('=') == std::string::npos);
    REQUIRE(out == "-_8");
}

TEST_CASE("code_verifier is in the unreserved 43-char range", "[oauth][pkce]") {
    std::string v = code_verifier();
    REQUIRE(v.size() == 43);
    for (char c : v)
        REQUIRE((std::isalnum((unsigned char)c) || c == '-' || c == '_'));
    REQUIRE(code_verifier() != code_verifier());  // random
}

TEST_CASE("credentials round-trip the Python field set", "[oauth][creds]") {
    json src = {
        {"token", "ya29.abc"},
        {"refresh_token", "1//refresh"},
        {"token_uri", "https://oauth2.googleapis.com/token"},
        {"client_id", "cid"},
        {"client_secret", "csecret"},
        {"scopes", {"https://www.googleapis.com/auth/drive.file"}},
        {"expiry", "2030-01-02T03:04:05.000000Z"},
        {"universe_domain", "googleapis.com"},  // extra key tolerated
    };
    Credentials c = Credentials::from_json(src);
    REQUIRE(c.token == "ya29.abc");
    REQUIRE(c.refresh_token == "1//refresh");
    REQUIRE(c.scopes.size() == 1);
    REQUIRE(c.expiry == parse_expiry("2030-01-02T03:04:05.000000Z"));

    json out = c.to_json();
    REQUIRE(out["refresh_token"] == "1//refresh");
    REQUIRE(out["expiry"] == "2030-01-02T03:04:05.000000Z");
    // re-parse is stable
    REQUIRE(Credentials::from_json(out).expiry == c.expiry);
}

TEST_CASE("expired() respects skew and unknown expiry", "[oauth][creds]") {
    Credentials c;
    c.expiry = std::time(nullptr) + 30;  // 30s out, default skew 60s
    REQUIRE(c.expired());
    c.expiry = std::time(nullptr) + 600;
    REQUIRE_FALSE(c.expired());
    c.expiry = 0;  // unknown
    REQUIRE_FALSE(c.expired());
}

TEST_CASE("authorize URL carries client_id, scope, PKCE challenge, state",
          "[oauth][client]") {
    MemStore store;
    OAuthClient client(test_provider(), store);
    std::string url = client.build_authorize_url("STATE123", "CHALLENGE456");
    REQUIRE(url.rfind("https://accounts.example/auth?", 0) == 0);
    REQUIRE(url.find("client_id=cid.apps") != std::string::npos);
    REQUIRE(url.find("code_challenge=CHALLENGE456") != std::string::npos);
    REQUIRE(url.find("code_challenge_method=S256") != std::string::npos);
    REQUIRE(url.find("state=STATE123") != std::string::npos);
    REQUIRE(url.find("response_type=code") != std::string::npos);
    REQUIRE(url.find("drive.file") != std::string::npos);
}

TEST_CASE("exchange_code persists credentials from the token response",
          "[oauth][client]") {
    MemStore store;
    std::string seen_url, seen_body;
    OAuthClient client(
        test_provider(), store,
        [&](const std::string& url, const std::string& body) {
            seen_url = url;
            seen_body = body;
            whisperx::server::net::Response r;
            r.status = 200;
            r.body =
                R"({"access_token":"AT1","refresh_token":"RT1","expires_in":3600})";
            return r;
        });
    Credentials c = client.exchange_code("authcode", "verifier123");
    REQUIRE(seen_url == "https://oauth.example/token");
    REQUIRE(seen_body.find("grant_type=authorization_code") != std::string::npos);
    REQUIRE(seen_body.find("code=authcode") != std::string::npos);
    REQUIRE(seen_body.find("code_verifier=verifier123") != std::string::npos);
    REQUIRE(c.token == "AT1");
    REQUIRE(store.value.has_value());
    REQUIRE(client.is_linked());
}

TEST_CASE("access_token refreshes when expired and keeps the refresh token",
          "[oauth][client]") {
    MemStore store;
    // Seed an expired credential with a refresh token.
    Credentials seed;
    seed.token = "OLD";
    seed.refresh_token = "RT-keep";
    seed.token_uri = "https://oauth.example/token";
    seed.expiry = std::time(nullptr) - 10;  // already expired
    store.save(seed.to_json().dump());

    int calls = 0;
    OAuthClient client(
        test_provider(), store,
        [&](const std::string&, const std::string& body) {
            ++calls;
            REQUIRE(body.find("grant_type=refresh_token") != std::string::npos);
            REQUIRE(body.find("refresh_token=RT-keep") != std::string::npos);
            whisperx::server::net::Response r;
            r.status = 200;
            // Google omits refresh_token on refresh — must retain the old one.
            r.body = R"({"access_token":"NEW","expires_in":3600})";
            return r;
        });

    auto tok = client.access_token();
    REQUIRE(tok.has_value());
    REQUIRE(*tok == "NEW");
    REQUIRE(calls == 1);

    // Persisted creds keep the refresh token and the new access token.
    Credentials after = Credentials::from_json(json::parse(*store.value));
    REQUIRE(after.refresh_token == "RT-keep");
    REQUIRE(after.token == "NEW");

    // A second call is now fresh — no further refresh.
    auto tok2 = client.access_token();
    REQUIRE(*tok2 == "NEW");
    REQUIRE(calls == 1);
}

TEST_CASE("access_token is nullopt when nothing is stored", "[oauth][client]") {
    MemStore store;
    OAuthClient client(test_provider(), store,
                       [](const std::string&, const std::string&) {
                           whisperx::server::net::Response r;
                           r.status = 500;
                           return r;
                       });
    REQUIRE_FALSE(client.access_token().has_value());
    REQUIRE_FALSE(client.is_linked());
}

TEST_CASE("form_encode produces a urlencoded body", "[oauth][net]") {
    std::string body = whisperx::server::net::form_encode(
        {{"grant_type", "authorization_code"}, {"redirect_uri",
                                                "http://127.0.0.1:8000/x"}});
    REQUIRE(body ==
            "grant_type=authorization_code&redirect_uri="
            "http%3A%2F%2F127.0.0.1%3A8000%2Fx");
}
