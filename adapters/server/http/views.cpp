#include "http/views.hpp"

#include <cmath>
#include <cstdio>
#include <ctime>
#include <regex>
#include <string>

#include "edits/edits.hpp"

namespace whisperx::server::views {

namespace {

std::string trim(const std::string& s) {
    auto b = s.find_first_not_of(" \t\n\r\f\v");
    if (b == std::string::npos) return "";
    auto e = s.find_last_not_of(" \t\n\r\f\v");
    return s.substr(b, e - b + 1);
}

double round3(double x) { return std::nearbyint(x * 1000.0) / 1000.0; }

bool truthy(const json& v) {
    if (v.is_null()) return false;
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number()) return v.get<double>() != 0.0;
    if (v.is_string()) return !v.get<std::string>().empty();
    return false;
}

std::string fmt_duration(const json& row, const char* key) {
    long sec = 0;
    if (row.is_object() && row.contains(key) && row[key].is_number())
        sec = static_cast<long>(row[key].get<double>());
    long h = sec / 3600, rem = sec % 3600, m = rem / 60, s = rem % 60;
    char buf[32];
    if (h)
        std::snprintf(buf, sizeof(buf), "%ldh %02ldm", h, m);
    else
        std::snprintf(buf, sizeof(buf), "%ldm %02lds", m, s);
    return buf;
}

std::string fmt_clock(double total) {
    long sec = static_cast<long>(total);
    long h = sec / 3600, rem = sec % 3600, m = rem / 60;
    char buf[32];
    if (h)
        std::snprintf(buf, sizeof(buf), "%ldh %02ldm", h, m);
    else
        std::snprintf(buf, sizeof(buf), "%ldm", m);
    return buf;
}

// "2026-06-07T14:30:00+00:00" -> "Jun 07, 2026 · 14:30" (server.py::_fmt_date).
std::string fmt_date(const json& iso_j) {
    if (!iso_j.is_string()) return "";
    std::string iso = iso_j.get<std::string>();
    if (iso.empty()) return "";
    std::tm tm{};
    // Parse the leading "YYYY-MM-DDTHH:MM:SS" (ignore any zone suffix).
    if (std::sscanf(iso.c_str(), "%d-%d-%dT%d:%d:%d", &tm.tm_year, &tm.tm_mon,
                    &tm.tm_mday, &tm.tm_hour, &tm.tm_min, &tm.tm_sec) < 5)
        return iso;
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    char buf[64];
    std::strftime(buf, sizeof(buf), "%b %d, %Y \xc2\xb7 %H:%M", &tm);
    return buf;
}

}  // namespace

std::string speaker_label(const json& raw) {
    if (!raw.is_string() || raw.get<std::string>().empty()) return "Speaker";
    std::string s = raw.get<std::string>();
    static const std::regex re(R"(SPEAKER_(\d+))");
    std::smatch m;
    if (std::regex_match(s, m, re))
        return "Speaker " + std::to_string(std::stoi(m[1].str()) + 1);
    return s;
}

std::string resolve_label(const json& raw, const json& names) {
    if (names.is_object() && raw.is_string()) {
        auto key = raw.get<std::string>();
        if (!key.empty() && names.contains(key) && names[key].is_string() &&
            !names[key].get<std::string>().empty())
            return names[key].get<std::string>();
    }
    return speaker_label(raw);
}

json turn_words(const json& segments, const json& seg_indices) {
    // Derive display words from the canonical tokenization (edits::turn_atoms) so
    // the rendered tokens — and thus the char offsets the SPA computes over them —
    // match exactly what apply_turn_split splits on. Here we only add the
    // view-only concerns: round timing to 3dp and rename the token key to `text`.
    json out = json::array();
    for (const auto& a : whisperx::edits::turn_atoms(segments, seg_indices)) {
        json wd = {{"text", a.at("word")}};
        if (a.contains("start") && a.contains("end")) {
            wd["start"] = round3(a["start"].get<double>());
            wd["end"] = round3(a["end"].get<double>());
        }
        if (a.contains("stale")) wd["stale"] = true;
        out.push_back(std::move(wd));
    }
    return out;
}

json build_turns(const json& segments, const json& names) {
    json turns = json::array();
    for (const auto& t : whisperx::edits::group_turns(segments)) {
        json words = turn_words(segments, t.at("seg_indices"));
        bool any = false;
        for (const auto& w : words)
            if (!w.at("text").get<std::string>().empty()) {
                any = true;
                break;
            }
        if (!any) continue;
        turns.push_back({
            {"index", t.at("index")},
            {"speaker", t.at("speaker")},
            {"label", resolve_label(t.at("speaker"), names)},
            {"start", t.at("start")},
            {"end", t.at("end")},
            {"words", std::move(words)},
            {"text", t.at("text")},
        });
    }
    return turns;
}

namespace {
struct StatusMeta {
    const char* label;
    const char* chip_class;
    bool viewable;
};
StatusMeta status_meta(const std::string& status) {
    if (status == "done") return {"Done", "chip--ok", true};
    if (status == "running") return {"Processing", "chip--run", false};
    if (status == "queued") return {"Queued", "chip--run", false};
    if (status == "error") return {"Error", "chip--err", false};
    return {"Unprocessed", "", false};
}
}  // namespace

json card(const json& row) {
    std::string status =
        row.contains("status") && row["status"].is_string()
            ? row["status"].get<std::string>()
            : "";
    auto meta = status_meta(status);
    std::string sub;
    if (status == "done") {
        long ns = (row.contains("num_segments") && row["num_segments"].is_number())
                      ? row["num_segments"].get<long>()
                      : 0;
        sub = std::to_string(ns) + " segments transcribed";
        if (row.contains("language") && row["language"].is_string() &&
            !row["language"].get<std::string>().empty())
            sub += " \xc2\xb7 language " + row["language"].get<std::string>();
        sub += ".";
    } else if (status == "error") {
        std::string err =
            (row.contains("error") && row["error"].is_string())
                ? row["error"].get<std::string>()
                : "unknown error";
        sub = "Failed: " + (err.empty() ? "unknown error" : err);
    } else {
        sub = "Awaiting transcription on CPU.";
    }
    auto str_or = [&](const char* k, const char* def) -> json {
        if (row.contains(k) && row[k].is_string() &&
            !row[k].get<std::string>().empty())
            return row[k];
        return def;
    };
    json out = {
        {"id", row.value("id", "")},
        {"name", (row.contains("filename") && row["filename"].is_string() &&
                  !row["filename"].get<std::string>().empty())
                     ? row["filename"]
                     : json("Untitled recording")},
        {"chip_label", meta.label},
        {"chip_class", meta.chip_class},
        {"viewable", meta.viewable},
        {"dur", fmt_duration(row, "duration")},
        {"date", fmt_date(row.value("created_at", json(nullptr)))},
        {"sub", sub},
        {"model", row.value("model", json(nullptr))},
        {"language", row.value("language", json(nullptr))},
        {"diarized", row.contains("diarized") && truthy(row["diarized"])},
        {"num_segments", (row.contains("num_segments") &&
                          row["num_segments"].is_number())
                             ? row["num_segments"]
                             : json(0)},
        {"status", status},
        {"stage", row.value("stage", json(nullptr))},
        {"error", row.value("error", json(nullptr))},
        {"translations", (row.contains("translations") &&
                          row["translations"].is_object())
                             ? row["translations"]
                             : json::object()},
    };
    (void)str_or;
    return out;
}

json summary(const json& rows) {
    long count = static_cast<long>(rows.size());
    long done = 0;
    double total_audio = 0, transcribed = 0;
    for (const auto& r : rows) {
        double d = (r.contains("duration") && r["duration"].is_number())
                       ? r["duration"].get<double>()
                       : 0.0;
        total_audio += d;
        if (r.value("status", "") == "done") {
            ++done;
            transcribed += d;
        }
    }
    long pct = count ? std::lround(static_cast<double>(done) / count * 100) : 0;
    return {
        {"count", count},
        {"transcribed", fmt_clock(transcribed)},
        {"total_audio", fmt_clock(total_audio)},
        {"pct", pct},
    };
}

json models_event(const json& status) {
    json active = status.value("active", json(nullptr));
    json active_meta = nullptr;
    if (status.contains("models") && status["models"].is_array())
        for (const auto& m : status["models"])
            if (m.value("name", json(nullptr)) == active) {
                active_meta = m;
                break;
            }
    bool ready = active_meta.is_object() && active_meta.value("loaded", false);
    return {
        {"type", "models"},
        {"models_ready", ready},
        {"active", active},
        {"bundle_error", active_meta.is_object()
                             ? active_meta.value("error", json(nullptr))
                             : json(nullptr)},
        {"diarize_available", status.value("diarize_available", json(nullptr))},
        {"diarize_error", status.value("diarize_error", json(nullptr))},
        {"models", status.value("models", json::array())},
    };
}

namespace {
std::string fmt_ts(const json& sec_j) {
    if (!sec_j.is_number()) return "--:--";
    long sec = static_cast<long>(sec_j.get<double>());
    long m = sec / 60, s = sec % 60, h = m / 60;
    m %= 60;
    char buf[32];
    if (h)
        std::snprintf(buf, sizeof(buf), "%ld:%02ld:%02ld", h, m, s);
    else
        std::snprintf(buf, sizeof(buf), "%ld:%02ld", m, s);
    return buf;
}
}  // namespace

std::string render_markdown(const json& result, const json& names,
                            const std::string& title) {
    std::string t = trim(title);
    std::string out = "# " + (t.empty() ? "Transcript" : t) + "\n\n";
    const json segments =
        result.contains("segments") && result["segments"].is_array()
            ? result["segments"]
            : json::array();
    if (segments.empty()) return out + "_No speech detected._\n";
    for (const auto& turn : whisperx::edits::group_turns(segments)) {
        std::string text = trim(turn.value("text", ""));
        if (text.empty()) continue;
        std::string label = resolve_label(turn.at("speaker"), names);
        std::string span =
            fmt_ts(turn.at("start")) + " \xe2\x80\x93 " + fmt_ts(turn.at("end"));
        out += "**" + label + "** [" + span + "]\n\n" + text + "\n\n";
    }
    return out;
}

}  // namespace whisperx::server::views
