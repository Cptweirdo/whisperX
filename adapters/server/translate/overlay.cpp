#include "translate/overlay.hpp"

#include <cstdio>
#include <utility>

namespace whisperx::server::translate {

std::optional<std::string> start_key(const json& start) {
    if (start.is_null() || !start.is_number()) return std::nullopt;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3f", start.get<double>());
    return std::string(buf);
}

json build_entries(const json& segments,
                   const std::vector<std::string>& translated_texts) {
    json entries = json::object();
    std::size_t n = std::min(segments.size(), translated_texts.size());
    for (std::size_t i = 0; i < n; ++i) {
        const json& seg = segments[i];
        auto key = start_key(seg.value("start", json(nullptr)));
        if (!key) continue;
        std::string src = seg.value("text", std::string());
        entries[*key] = {{"src", src}, {"tr", translated_texts[i]}};
    }
    return entries;
}

namespace {

// Return (entries, has_src). has_src is false for a legacy v1 overlay (no stored
// source text), where freshness can only be decided by an exact start match.
std::pair<json, bool> entries_of(const json& overlay) {
    if (overlay.contains("entries") && overlay["entries"].is_object())
        return {overlay["entries"], true};
    json derived = json::object();
    for (const auto& seg : overlay.value("segments", json::array())) {
        auto key = start_key(seg.value("start", json(nullptr)));
        if (key) derived[*key] = {{"tr", seg.value("text", std::string())}};
    }
    return {derived, false};
}

}  // namespace

json apply_overlay(const json& orig_segments, const json& overlay) {
    auto [entries, has_src] = entries_of(overlay.is_object() ? overlay : json::object());
    json out = json::array();
    for (const auto& seg : orig_segments) {
        std::string src_text = seg.value("text", std::string());
        auto key = start_key(seg.value("start", json(nullptr)));
        std::string text;
        bool stale;
        json entry = (key && entries.contains(*key)) ? entries[*key] : json(nullptr);
        if (entry.is_object() &&
            (!has_src || entry.value("src", std::string()) == src_text)) {
            text = entry.value("tr", std::string());
            stale = false;
        } else {
            text = src_text;
            stale = true;
        }
        out.push_back({{"start", seg.value("start", json(nullptr))},
                       {"end", seg.value("end", json(nullptr))},
                       {"speaker", seg.value("speaker", json(nullptr))},
                       {"text", text},
                       {"stale", stale}});
    }
    return out;
}

}  // namespace whisperx::server::translate
