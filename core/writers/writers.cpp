#include "writers/writers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "text/utf8.hpp"

namespace whisperx::writers {

namespace {

constexpr const char* kWS = " \t\n\r\f\v";

bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v';
}

// Python str.strip() — both ends, ASCII whitespace (the parity domain).
std::string strip(const std::string& s) {
    std::size_t b = s.find_first_not_of(kWS);
    if (b == std::string::npos) return "";
    std::size_t e = s.find_last_not_of(kWS);
    return s.substr(b, e - b + 1);
}

// Python len(str) — codepoint count, not bytes.
std::size_t cp_len(const std::string& s) {
    return whisperx::text::utf8_chars(s).size();
}

std::string replace_all(std::string s, const std::string& from,
                        const std::string& to) {
    if (from.empty()) return s;
    std::size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

// Python repr/str of a float: the shortest decimal that round-trips, in repr's
// fixed-vs-scientific style. Emulated via the classic increasing-%.{p}g loop
// (matches CPython repr across the realistic timestamp domain), then the repr
// ".0" rule for integral-valued floats shown in fixed notation.
std::string py_repr_double(double v) {
    char buf[40];
    int prec = 17;
    for (int p = 1; p <= 17; ++p) {
        std::snprintf(buf, sizeof buf, "%.*g", p, v);
        if (std::strtod(buf, nullptr) == v) {
            prec = p;
            break;
        }
    }
    std::snprintf(buf, sizeof buf, "%.*g", prec, v);
    std::string s(buf);
    // repr shows a decimal point for finite values rendered without one (e.g.
    // "2" -> "2.0"); leave exponential / nan / inf untouched.
    if (s.find_first_of(".eEnN") == std::string::npos) s += ".0";
    return s;
}

std::string snprintf_str(const char* fmt, long long a) {
    char buf[64];
    std::snprintf(buf, sizeof buf, fmt, a);
    return buf;
}

std::optional<std::string> get_speaker_not_none(const json& obj) {
    if (obj.contains("speaker") && !obj["speaker"].is_null())
        return obj["speaker"].get<std::string>();
    return std::nullopt;
}

bool has_num(const json& obj, const char* key) {
    return obj.contains(key) && !obj[key].is_null();
}

// re.sub(r"^(\s*)(.*)$", r"\1<u>\2</u>", word) — wrap the word (after any leading
// whitespace run) in <u>…</u>. Manual leading-ws split, no std::regex.
std::string highlight_wrap(const std::string& word) {
    std::size_t i = 0;
    while (i < word.size() && is_ws(word[i])) ++i;
    return word.substr(0, i) + "<u>" + word.substr(i) + "</u>";
}

}  // namespace

std::string format_timestamp(double seconds, bool always_include_hours,
                             char decimal_marker) {
    long long ms = static_cast<long long>(std::nearbyint(seconds * 1000.0));
    long long hours = ms / 3'600'000;
    ms -= hours * 3'600'000;
    long long minutes = ms / 60'000;
    ms -= minutes * 60'000;
    long long secs = ms / 1'000;
    ms -= secs * 1'000;
    std::string hours_marker =
        (always_include_hours || hours > 0) ? snprintf_str("%02lld:", hours) : "";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%02lld:%02lld%c%03lld", minutes, secs,
                  decimal_marker, ms);
    return hours_marker + buf;
}

std::vector<Cue> iterate_result(const json& result, const json& options,
                                bool always_include_hours, char decimal_marker) {
    json raw_mlw = options.value("max_line_width", json());
    json mlc = options.value("max_line_count", json());
    bool highlight_words = options.value("highlight_words", false);
    bool mlw_is_none = raw_mlw.is_null();
    bool mlc_is_none = mlc.is_null();
    int max_line_width = mlw_is_none ? 1000 : raw_mlw.get<int>();
    int max_line_count = mlc_is_none ? 0 : mlc.get<int>();
    bool preserve_segments = mlc_is_none || mlw_is_none;

    auto ft = [&](double s) {
        return format_timestamp(s, always_include_hours, decimal_marker);
    };

    std::vector<Cue> out;
    if (!result.contains("segments") || result["segments"].empty()) return out;
    const json& segs = result["segments"];

    struct Times {
        double seg_start;
        double seg_end;
        std::optional<std::string> speaker;
    };
    std::vector<std::pair<std::vector<json>, std::vector<Times>>> groups;

    {
        int line_len = 0;
        int line_count = 1;
        std::vector<json> subtitle;
        std::vector<Times> times;
        double last = segs[0].value("start", 0.0);
        for (const auto& segment : segs) {
            const json words = segment.value("words", json::array());
            for (std::size_t i = 0; i < words.size(); ++i) {
                json timing = words[i];  // copy
                bool long_pause = !preserve_segments;
                if (has_num(timing, "start"))
                    long_pause =
                        long_pause && (timing["start"].get<double>() - last > 3.0);
                else
                    long_pause = false;
                int wlen = static_cast<int>(cp_len(timing["word"].get<std::string>()));
                bool has_room = line_len + wlen <= max_line_width;
                bool seg_break = (i == 0 && !subtitle.empty() && preserve_segments);
                if (line_len > 0 && has_room && !long_pause && !seg_break) {
                    line_len += wlen;  // line continuation
                } else {
                    std::string stripped = strip(timing["word"].get<std::string>());
                    timing["word"] = stripped;
                    if ((!subtitle.empty() && !mlc_is_none &&
                         (long_pause || line_count >= max_line_count)) ||
                        seg_break) {
                        groups.emplace_back(subtitle, times);
                        subtitle.clear();
                        times.clear();
                        line_count = 1;
                    } else if (line_len > 0) {
                        line_count += 1;
                        timing["word"] = std::string("\n") + stripped;
                    }
                    line_len = static_cast<int>(
                        cp_len(strip(timing["word"].get<std::string>())));
                }
                subtitle.push_back(timing);
                times.push_back({segment.value("start", 0.0),
                                 segment.value("end", 0.0),
                                 get_speaker_not_none(segment)});
                if (has_num(timing, "start")) last = timing["start"].get<double>();
            }
        }
        if (!subtitle.empty()) groups.emplace_back(subtitle, times);
    }

    bool has_words = segs[0].contains("words");
    if (has_words) {
        std::string lang = result.value("language", std::string());
        bool no_spaces = (lang == "ja" || lang == "zh");
        for (auto& [subtitle, times] : groups) {
            std::optional<std::string> speaker = times[0].speaker;
            std::vector<double> word_starts, word_ends;
            for (auto& w : subtitle) {
                if (has_num(w, "start")) word_starts.push_back(w["start"].get<double>());
                if (has_num(w, "end")) word_ends.push_back(w["end"].get<double>());
            }
            std::string subtitle_start, subtitle_end;
            if (!word_starts.empty() && !word_ends.empty()) {
                subtitle_start =
                    ft(*std::min_element(word_starts.begin(), word_starts.end()));
                subtitle_end =
                    ft(*std::max_element(word_ends.begin(), word_ends.end()));
            } else {
                subtitle_start = ft(times[0].seg_start);
                subtitle_end = ft(times[0].seg_end);
            }
            std::string subtitle_text;
            for (std::size_t k = 0; k < subtitle.size(); ++k) {
                if (k && !no_spaces) subtitle_text += " ";
                subtitle_text += subtitle[k]["word"].get<std::string>();
            }
            bool has_timing = false;
            for (auto& w : subtitle)
                if (has_num(w, "start")) {
                    has_timing = true;
                    break;
                }
            std::string prefix = speaker ? "[" + *speaker + "]: " : "";
            if (highlight_words && has_timing) {
                std::string last = subtitle_start;
                std::vector<std::string> all_words;
                for (auto& w : subtitle)
                    all_words.push_back(w["word"].get<std::string>());
                for (std::size_t i = 0; i < subtitle.size(); ++i) {
                    if (!has_num(subtitle[i], "start")) continue;
                    std::string start = ft(subtitle[i]["start"].get<double>());
                    std::string end = ft(subtitle[i]["end"].get<double>());
                    if (last != start)
                        out.push_back({last, start, prefix + subtitle_text});
                    std::string hl;
                    for (std::size_t j = 0; j < all_words.size(); ++j) {
                        if (j) hl += " ";
                        hl += (j == i) ? highlight_wrap(all_words[j]) : all_words[j];
                    }
                    out.push_back({start, end, prefix + hl});
                    last = end;
                }
            } else {
                out.push_back({subtitle_start, subtitle_end, prefix + subtitle_text});
            }
        }
    } else {
        for (const auto& segment : segs) {
            std::string ss = ft(segment.value("start", 0.0));
            std::string se = ft(segment.value("end", 0.0));
            std::string text = replace_all(strip(segment.value("text", std::string())),
                                           "-->", "->");
            if (segment.contains("speaker"))
                text = "[" + segment["speaker"].get<std::string>() + "]: " + text;
            out.push_back({ss, se, text});
        }
    }
    return out;
}

std::string write_txt(const json& result, const json& /*options*/) {
    std::string out;
    for (const auto& segment : result.value("segments", json::array())) {
        auto speaker = get_speaker_not_none(segment);
        std::string text = strip(segment.value("text", std::string()));
        if (speaker)
            out += "[" + *speaker + "]: " + text + "\n";
        else
            out += text + "\n";
    }
    return out;
}

std::string write_srt(const json& result, const json& options) {
    std::string out;
    int i = 1;
    for (const auto& cue : iterate_result(result, options, true, ',')) {
        out += std::to_string(i) + "\n" + cue.start + " --> " + cue.end + "\n" +
               cue.text + "\n\n";
        ++i;
    }
    return out;
}

std::string write_vtt(const json& result, const json& options) {
    std::string out = "WEBVTT\n\n";
    for (const auto& cue : iterate_result(result, options, false, '.'))
        out += cue.start + " --> " + cue.end + "\n" + cue.text + "\n\n";
    return out;
}

std::string write_tsv(const json& result, const json& /*options*/) {
    std::string out = "start\tend\ttext\n";
    for (const auto& segment : result.value("segments", json::array())) {
        long long s = static_cast<long long>(
            std::nearbyint(1000.0 * segment.value("start", 0.0)));
        long long e = static_cast<long long>(
            std::nearbyint(1000.0 * segment.value("end", 0.0)));
        std::string text =
            replace_all(strip(segment.value("text", std::string())), "\t", " ");
        out += std::to_string(s) + "\t" + std::to_string(e) + "\t" + text + "\n";
    }
    return out;
}

std::string write_aud(const json& result, const json& /*options*/) {
    std::string out;
    for (const auto& segment : result.value("segments", json::array())) {
        std::string prefix;
        if (segment.contains("speaker"))
            prefix = "[[" + segment["speaker"].get<std::string>() + "]]";
        std::string text =
            replace_all(strip(segment.value("text", std::string())), "\t", " ");
        out += py_repr_double(segment.value("start", 0.0)) + "\t" +
               py_repr_double(segment.value("end", 0.0)) + "\t" + prefix + text +
               "\n";
    }
    return out;
}

std::string write_json(const json& result, const json& /*options*/) {
    return result.dump();
}

}  // namespace whisperx::writers
