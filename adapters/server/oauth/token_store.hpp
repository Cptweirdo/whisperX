// Where OAuthClient persists the credentials JSON. The interface keeps the OAuth
// core decoupled from any concrete store (so tests can inject an in-memory one);
// KeyringTokenStore is the production impl over the OS keyring (secrets/keyring),
// under the GDRIVE_CREDS key shared with the Python host.
#pragma once

#include <optional>
#include <string>

#include "secrets/keyring.hpp"

namespace whisperx::server::oauth {

struct TokenStore {
    virtual ~TokenStore() = default;
    virtual std::optional<std::string> load() = 0;
    virtual void save(const std::string& value) = 0;  // may throw on no backend
    virtual void clear() = 0;
};

class KeyringTokenStore : public TokenStore {
public:
    std::optional<std::string> load() override {
        return secrets::get(secrets::keys::GDRIVE_CREDS);
    }
    void save(const std::string& value) override {
        secrets::set(secrets::keys::GDRIVE_CREDS, value);
    }
    void clear() override { secrets::erase(secrets::keys::GDRIVE_CREDS); }
};

}  // namespace whisperx::server::oauth
