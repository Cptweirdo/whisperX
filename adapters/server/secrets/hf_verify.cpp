#include "secrets/hf_verify.hpp"

#include <cctype>
#include <cstdlib>

#include <nlohmann/json.hpp>

#include "http/curl_client.hpp"

namespace whisperx::server::secrets {

using nlohmann::json;

namespace {

std::string diarize_model() {
    const char* m = std::getenv("WHISPERX_DIARIZE_MODEL");
    if (m && m[0]) return m;
    return "pyannote/speaker-diarization-community-1";
}

net::Options bearer_opts(const std::string& token) {
    net::Options o;
    o.timeout_s = 15;
    o.headers.push_back("Authorization: Bearer " + token);
    return o;
}

}  // namespace

std::pair<bool, std::string> verify_token(const std::string& raw) {
    std::string token = raw;
    // trim
    while (!token.empty() && std::isspace((unsigned char)token.front()))
        token.erase(token.begin());
    while (!token.empty() && std::isspace((unsigned char)token.back()))
        token.pop_back();
    if (token.empty()) return {false, "Enter a token to continue."};

    // 1. whoami — is the token valid?
    auto who = net::get("https://huggingface.co/api/whoami-v2",
                        bearer_opts(token));
    if (who.status == 0)
        return {false, "Couldn't reach Hugging Face. Check your connection."};
    if (who.status == 401 || who.status == 403)
        return {false, "Invalid token. Check that you copied it correctly."};
    if (!who.ok())
        return {false,
                "Couldn't verify token: HTTP " + std::to_string(who.status)};
    std::string name;
    try {
        json j = json::parse(who.body);
        if (j.contains("name") && j["name"].is_string())
            name = j["name"].get<std::string>();
    } catch (...) {
        // non-fatal — a 2xx without a parseable name still means valid
    }

    // 2. gated diarization model — are its conditions accepted?
    const std::string model = diarize_model();
    auto info = net::get("https://huggingface.co/api/models/" + model,
                         bearer_opts(token));
    if (info.status == 403) {
        std::string url = "https://huggingface.co/" + model;
        return {false,
                "Token is valid, but you haven't accepted the conditions for " +
                    model + ". Open " + url + " and accept them, then retry."};
    }
    if (info.status != 0 && !info.ok())
        return {false,
                "Couldn't check model access: HTTP " +
                    std::to_string(info.status)};

    std::string who_str = name.empty() ? "" : (" as " + name);
    return {true, "Token verified" + who_str + " — diarization is ready."};
}

}  // namespace whisperx::server::secrets
