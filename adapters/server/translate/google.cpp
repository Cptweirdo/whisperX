#include "translate/google.hpp"

#include <nlohmann/json.hpp>

#include "http/curl_client.hpp"

namespace whisperx::server::translate {

namespace {

using nlohmann::json;
namespace net = whisperx::server::net;

constexpr const char* kEndpoint =
    "https://translation.googleapis.com/language/translate/v2";
// Conservative caps below Google's hard limits (128 segments / 30k code points).
constexpr std::size_t kMaxSegmentsPerRequest = 100;
constexpr std::size_t kMaxCharsPerRequest = 25000;

// Split `texts` into chunks within the per-request segment/char limits. A single
// text longer than the char cap still goes out alone (the API rejects it if it
// truly exceeds the hard limit, surfaced as a TranslationError).
std::vector<std::vector<std::string>> batch(
    const std::vector<std::string>& texts) {
    std::vector<std::vector<std::string>> batches;
    std::vector<std::string> cur;
    std::size_t cur_chars = 0;
    for (const auto& t : texts) {
        if (!cur.empty() && (cur.size() >= kMaxSegmentsPerRequest ||
                             cur_chars + t.size() > kMaxCharsPerRequest)) {
            batches.push_back(std::move(cur));
            cur.clear();
            cur_chars = 0;
        }
        cur_chars += t.size();
        cur.push_back(t);
    }
    if (!cur.empty()) batches.push_back(std::move(cur));
    return batches;
}

std::vector<std::string> translate_batch(const std::vector<std::string>& b,
                                         const std::string& target_lang,
                                         const std::string& api_key,
                                         const std::string& source_lang) {
    // Body as repeated q= fields (the v2 API accepts q arrays).
    std::vector<std::pair<std::string, std::string>> fields = {
        {"target", target_lang}, {"format", "text"}};
    if (!source_lang.empty()) fields.emplace_back("source", source_lang);
    for (const auto& t : b) fields.emplace_back("q", t);

    std::string url = std::string(kEndpoint) + "?" +
                      net::form_encode({{"key", api_key}});
    net::Response resp = net::post(url, net::form_encode(fields), {{}, 60, true});

    if (resp.status == 0)
        throw TranslationError("Couldn't reach Google: request failed");
    if (!resp.ok())
        throw TranslationError("Google Translation API error (" +
                               std::to_string(resp.status) + "): " + resp.body);

    json payload = json::parse(resp.body, nullptr, false);
    if (payload.is_discarded())
        throw TranslationError("Unexpected response from Google: invalid JSON.");
    const json& translations =
        payload.value("data", json::object()).value("translations", json::array());
    if (translations.size() != b.size())
        throw TranslationError(
            "Unexpected response from Google: " +
            std::to_string(translations.size()) + " translations for " +
            std::to_string(b.size()) + " inputs.");
    std::vector<std::string> out;
    out.reserve(b.size());
    for (const auto& t : translations)
        out.push_back(t.value("translatedText", std::string()));
    return out;
}

}  // namespace

std::vector<std::string> google_translate(
    const std::vector<std::string>& texts, const std::string& target_lang,
    const std::string& api_key, const std::string& source_lang) {
    if (api_key.empty())
        throw TranslationError("No Google Translation API key configured.");
    if (texts.empty()) return {};
    std::vector<std::string> out;
    out.reserve(texts.size());
    for (const auto& b : batch(texts)) {
        auto part = translate_batch(b, target_lang, api_key, source_lang);
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

std::pair<bool, std::string> verify_api_key(const std::string& key) {
    // trim
    auto b = key.find_first_not_of(" \t\r\n");
    std::string k = (b == std::string::npos)
                        ? ""
                        : key.substr(b, key.find_last_not_of(" \t\r\n") - b + 1);
    if (k.empty()) return {false, "Enter an API key to continue."};

    // Hits the (free, cheap) supported-languages endpoint to confirm the key is
    // valid and the Translation API is enabled for it.
    std::string url = std::string(kEndpoint) + "/languages?" +
                      net::form_encode({{"key", k}});
    net::Response resp = net::get(url, {{}, 15, true});
    if (resp.status == 0) return {false, "Couldn't reach Google: request failed"};
    if (resp.status == 400 || resp.status == 401 || resp.status == 403)
        return {false,
                "Invalid or unauthorized API key. Check the key and that the "
                "Cloud Translation API is enabled for its project."};
    if (!resp.ok())
        return {false, "Couldn't verify key: HTTP " + std::to_string(resp.status)};
    return {true, "Key verified — translation is ready."};
}

}  // namespace whisperx::server::translate
