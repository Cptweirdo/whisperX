#include "secrets/keyring.hpp"

#include <cstdlib>

#include <keychain/keychain.h>

namespace whisperx::server::secrets {

namespace {

// keychain's (package, service) pair scopes the entries; `user` is the key name.
// Mirrors app/secret_store.py's SERVICE = "manuscript-whisperx".
constexpr const char* PACKAGE = "manuscript-whisperx";
constexpr const char* SERVICE = "manuscript-whisperx";

std::optional<std::string> env(const char* key) {
    const char* v = std::getenv(key);
    if (v && v[0]) return std::string(v);
    return std::nullopt;
}

}  // namespace

bool available() {
    // Probe a read for a sentinel key. A working backend returns NoError or
    // NotFound; a missing/unusable backend (no Secret Service, access denied)
    // returns GenericError / AccessDenied. Mirrors keyring_available().
    keychain::Error err;
    keychain::getPassword(PACKAGE, SERVICE, "__probe__", err);
    return err.type == keychain::ErrorType::NoError ||
           err.type == keychain::ErrorType::NotFound;
}

void set(const std::string& key, const std::string& value) {
    if (!available())
        throw SecretStoreUnavailable(
            "No OS keyring backend is available on this host, so the secret "
            "can't be stored securely. Install a Secret Service provider (e.g. "
            "gnome-keyring) or set the HF_TOKEN environment variable instead.");
    keychain::Error err;
    keychain::setPassword(PACKAGE, SERVICE, key, value, err);
    if (err)
        throw SecretStoreUnavailable("Keyring write failed: " + err.message);
}

std::optional<std::string> get(const std::string& key) {
    keychain::Error err;
    std::string val = keychain::getPassword(PACKAGE, SERVICE, key, err);
    if (err) return std::nullopt;  // NotFound / no backend / read error
    return val;
}

void erase(const std::string& key) {
    keychain::Error err;
    keychain::deletePassword(PACKAGE, SERVICE, key, err);
    // Ignore errors (NotFound when absent, no backend) — matches the oracle.
}

std::optional<std::string> resolve_hf_token() {
    if (auto e = env("HF_TOKEN")) return e;
    if (auto e = env("HUGGINGFACE_TOKEN")) return e;
    return get(keys::HF_TOKEN);
}

std::optional<std::string> resolve_google_api_key() {
    if (auto e = env("GOOGLE_TRANSLATE_API_KEY")) return e;
    return get(keys::GOOGLE_TRANSLATE);
}

}  // namespace whisperx::server::secrets
