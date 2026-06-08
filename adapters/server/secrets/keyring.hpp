// OS keyring secret store — the C++ port of app/secret_store.py. Secrets (the HF
// token, and later the Google/gdrive creds) live in the native keyring
// (macOS Keychain / Windows Credential Locker / Linux Secret Service via
// libsecret), never in SQLite or a plaintext file. There is intentionally NO file
// fallback: on a host with no keyring backend, set() raises SecretStoreUnavailable
// rather than writing the secret somewhere readable (matching the Python contract);
// such hosts still supply the token via the HF_TOKEN env var, honoured first by
// resolve_hf_token(). Backed by hrantzsch/keychain.
#pragma once

#include <optional>
#include <stdexcept>
#include <string>

namespace whisperx::server::secrets {

// Raised by set() when no usable keyring backend exists (or the backend errors).
class SecretStoreUnavailable : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Keyring entry key names — identical to the app/secret_store.py constants so an
// existing native-keyring entry written by either host is interchangeable. Only
// HF_TOKEN is exercised by the CPU MVP; the rest land with the deferred features.
namespace keys {
inline constexpr const char* HF_TOKEN = "hf_token";
inline constexpr const char* GDRIVE_CREDS = "google_drive_creds";
inline constexpr const char* GDRIVE_FOLDER = "google_drive_folder";
inline constexpr const char* GOOGLE_TRANSLATE = "google_translate_api_key";
}  // namespace keys

// Whether a usable keyring backend is present (probe, not just a try/catch).
bool available();

// Store value under key. Throws SecretStoreUnavailable if no backend / on error.
void set(const std::string& key, const std::string& value);

// Read key, or nullopt if unset / no backend / read error.
std::optional<std::string> get(const std::string& key);

// Remove key (no-op if absent or no backend).
void erase(const std::string& key);

// The HF token in effect: HF_TOKEN / HUGGINGFACE_TOKEN env first (operator
// override always wins), else the keyring entry.
std::optional<std::string> resolve_hf_token();

// The Google Translation API key in effect: GOOGLE_TRANSLATE_API_KEY env first,
// else the keyring entry. (Translation is deferred; surfaced for settings parity.)
std::optional<std::string> resolve_google_api_key();

}  // namespace whisperx::server::secrets
