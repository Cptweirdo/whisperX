// Google Cloud Translation API (v2) backend — the C++ port of
// app/translation/google.py. Uses the simple API-key REST endpoint (no service
// account / OAuth), going over the shared libcurl client. Requests are batched to
// respect the documented v2 limits (<=128 strings, ~30k UTF-8 code points/call)
// and reassembled in order, so a caller can zip the results back onto the
// segments they came from. verify_api_key() ports secret_store.verify_google_api_key.
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace whisperx::server::translate {

// A translation request failed (bad key, quota, network, API error). Mirrors
// app.translation.base.TranslationError.
class TranslationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Translate `texts` into `target_lang`, auto-detecting the source when
// `source_lang` is empty. Returns a list of the same length/order as `texts`.
// Throws TranslationError on failure. `api_key` must be non-empty.
std::vector<std::string> google_translate(
    const std::vector<std::string>& texts, const std::string& target_lang,
    const std::string& api_key, const std::string& source_lang = "");

// Live-check a Google Translation API key against the supported-languages
// endpoint. Returns (ok, user-facing detail). Port of verify_google_api_key.
std::pair<bool, std::string> verify_api_key(const std::string& key);

}  // namespace whisperx::server::translate
