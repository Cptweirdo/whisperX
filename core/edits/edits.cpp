#include "edits/edits.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "text/sequence_matcher.hpp"
#include "time_iso.hpp"

namespace whisperx::edits {

namespace {

constexpr const char* kWhitespace = " \t\n\r\f\v";

// Python str.strip() (ASCII whitespace — tokens never carry exotic spaces).
std::string py_strip(const std::string& s) {
    const auto b = s.find_first_not_of(kWhitespace);
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(kWhitespace);
    return s.substr(b, e - b + 1);
}

// Python str.split() with no args: split on runs of whitespace, drop empties.
std::vector<std::string> py_split(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        const size_t b = s.find_first_not_of(kWhitespace, i);
        if (b == std::string::npos) break;
        size_t e = s.find_first_of(kWhitespace, b);
        if (e == std::string::npos) e = s.size();
        out.push_back(s.substr(b, e - b));
        i = e;
    }
    return out;
}

std::string py_join(const std::vector<std::string>& parts, const char* sep) {
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out += sep;
        out += parts[i];
    }
    return out;
}

// (obj.get(key) or "") coerced to a string ("" for missing/null/non-string).
std::string get_str(const json& obj, const char* key) {
    auto it = obj.find(key);
    if (it == obj.end() || it->is_null() || !it->is_string()) return "";
    return it->get<std::string>();
}

// obj.get(key) -> value or json null (never auto-inserts).
json get_or_null(const json& obj, const char* key) {
    auto it = obj.find(key);
    return (it == obj.end()) ? json(nullptr) : *it;
}

// _seg_key: seg.get("speaker") or None — falsy (null / empty string) -> null.
json seg_key(const json& seg) {
    json v = get_or_null(seg, "speaker");
    if (v.is_null()) return json(nullptr);
    if (v.is_string() && v.get<std::string>().empty()) return json(nullptr);
    return v;
}

// re.fullmatch(r"SPEAKER_(\d+)", s) -> the int group, or false.
bool match_speaker(const std::string& s, int& num) {
    static const std::string kPrefix = "SPEAKER_";
    if (s.size() <= kPrefix.size()) return false;
    if (s.compare(0, kPrefix.size(), kPrefix) != 0) return false;
    const std::string digits = s.substr(kPrefix.size());
    for (char c : digits) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    try {
        num = std::stoi(digits);
    } catch (...) {
        return false;  // overflow: treat as non-conforming
    }
    return true;
}

}  // namespace

json group_turns(const json& segments) {
    json turns = json::array();
    for (size_t i = 0; i < segments.size(); ++i) {
        const json key = seg_key(segments[i]);
        if (!turns.empty() && turns.back()["speaker"] == key) {
            turns.back()["seg_indices"].push_back(static_cast<int>(i));
        } else {
            json t = json::object();
            t["index"] = static_cast<int>(turns.size());
            t["speaker"] = key;
            t["start"] = nullptr;
            t["end"] = nullptr;
            t["seg_indices"] = json::array({static_cast<int>(i)});
            t["text"] = "";
            turns.push_back(std::move(t));
        }
    }
    for (json& t : turns) {
        const int first = t["seg_indices"].front().get<int>();
        const int last = t["seg_indices"].back().get<int>();
        t["start"] = get_or_null(segments[first], "start");
        t["end"] = get_or_null(segments[last], "end");
        std::vector<std::string> parts;
        for (const json& k : t["seg_indices"]) {
            std::string p = py_strip(get_str(segments[k.get<int>()], "text"));
            if (!p.empty()) parts.push_back(std::move(p));
        }
        t["text"] = py_join(parts, " ");
    }
    return turns;
}

json turn_atoms(const json& segments, const json& seg_indices) {
    auto truthy = [](const json& v) {
        if (v.is_boolean()) return v.get<bool>();
        if (v.is_number()) return v.get<double>() != 0.0;
        if (v.is_string()) return !v.get<std::string>().empty();
        return false;
    };
    json out = json::array();
    for (const json& k_j : seg_indices) {
        const std::size_t k = k_j.get<std::size_t>();
        if (k >= segments.size()) continue;
        const json& seg = segments[k];
        const json words = (seg.contains("words") && seg["words"].is_array())
                               ? seg["words"]
                               : json::array();
        if (!words.empty()) {
            for (const json& w : words) {
                const std::string token = py_strip(get_str(w, "word"));
                if (token.empty()) continue;
                json wd = {{"word", token}};
                if (w.contains("start") && w["start"].is_number() &&
                    w.contains("end") && w["end"].is_number()) {
                    wd["start"] = w["start"];
                    wd["end"] = w["end"];
                }
                out.push_back(std::move(wd));
            }
        } else {
            const std::string text = py_strip(get_str(seg, "text"));
            if (text.empty()) continue;
            json wd = {{"word", text}};
            if (seg.contains("start") && seg["start"].is_number() &&
                seg.contains("end") && seg["end"].is_number()) {
                wd["start"] = seg["start"];
                wd["end"] = seg["end"];
            }
            if (seg.contains("stale") && truthy(seg["stale"])) wd["stale"] = true;
            out.push_back(std::move(wd));
        }
    }
    return out;
}

json distinct_speakers(const json& segments) {
    json seen = json::array();
    for (const json& seg : segments) {
        const json key = seg_key(seg);
        if (key.is_null()) continue;
        bool found = false;
        for (const json& s : seen) {
            if (s == key) {
                found = true;
                break;
            }
        }
        if (!found) seen.push_back(key);
    }
    return seen;
}

std::string next_speaker_key(const json& existing_keys) {
    int highest = -1;
    if (existing_keys.is_array()) {
        for (const json& key : existing_keys) {
            const std::string s = key.is_string() ? key.get<std::string>()
                                                  : key.dump();  // str(key)
            int n = 0;
            if (match_speaker(s, n)) highest = std::max(highest, n);
        }
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "SPEAKER_%02d", highest + 1);
    return buf;
}

// --- interpolation / realign -------------------------------------------------

namespace {

// _interpolate_gaps — transcribed statement-for-statement (float-order
// sensitive; this TU is built with -ffp-contract=off). Mutates `words`.
void interpolate_gaps(json& words, const json& start, const json& end) {
    const int n = static_cast<int>(words.size());
    int i = 0;
    while (i < n) {
        if (words[i].contains("start")) {
            ++i;
            continue;
        }
        int j = i;
        while (j < n && !words[j].contains("start")) ++j;
        const int run = j - i;

        json* left_word = (i > 0 && words[i - 1].contains("end"))
                              ? &words[i - 1]
                              : nullptr;
        json* right_word =
            (j < n && words[j].contains("start")) ? &words[j] : nullptr;
        const json left_j = left_word ? (*left_word)["end"] : start;
        const json right_j = right_word ? (*right_word)["start"] : end;
        if (left_j.is_null() || right_j.is_null() ||
            right_j.get<double>() < left_j.get<double>()) {
            i = j;
            continue;
        }
        double left = left_j.get<double>();
        double right = right_j.get<double>();

        const double deficit = run * MIN_WORD_WIDTH - (right - left);
        if (deficit > 0) {
            const double cap_l =
                left_word ? std::max(0.0, ((*left_word)["end"].get<double>() -
                                           (*left_word)["start"].get<double>()) -
                                              MIN_WORD_WIDTH)
                          : 0.0;
            const double cap_r =
                right_word ? std::max(0.0, ((*right_word)["end"].get<double>() -
                                            (*right_word)["start"].get<double>()) -
                                               MIN_WORD_WIDTH)
                           : 0.0;
            double take_l = std::min(deficit / 2, cap_l);
            double take_r = std::min(deficit / 2, cap_r);
            take_l += std::min(cap_l - take_l, deficit - take_l - take_r);
            take_r += std::min(cap_r - take_r, deficit - take_l - take_r);
            if (left_word) {
                (*left_word)["end"] = (*left_word)["end"].get<double>() - take_l;
            }
            if (right_word) {
                (*right_word)["start"] =
                    (*right_word)["start"].get<double>() + take_r;
            }
            left -= take_l;
            right += take_r;
        }

        const double step = (right - left) / run;
        for (int k = 0; k < run; ++k) {
            words[i + k]["start"] = left + k * step;
            words[i + k]["end"] = left + (k + 1) * step;
        }
        i = j;
    }
}

}  // namespace

json realign_words(const json& old_words, const std::string& new_text,
                   const json& start, const json& end) {
    const std::vector<std::string> new_tokens = py_split(new_text);
    std::vector<std::string> old_tokens;
    old_tokens.reserve(old_words.size());
    for (const json& w : old_words) {
        old_tokens.push_back(py_strip(get_str(w, "word")));
    }

    // get_opcodes() only ever yields equal vs replace/insert/delete here; only
    // "equal" pairs carry timing. Map each equal new-index to its old-index from
    // the matching blocks; every other new token is an untimed insert/replace.
    whisperx::text::SequenceMatcher sm(old_tokens, new_tokens);
    std::unordered_map<int, int> equal_old_of_new;
    for (const auto& blk : sm.get_matching_blocks()) {
        for (int t = 0; t < blk.size; ++t) {
            equal_old_of_new[blk.b + t] = blk.a + t;
        }
    }

    json out = json::array();
    bool kept_timing = false;
    for (int nj = 0; nj < static_cast<int>(new_tokens.size()); ++nj) {
        json w = json::object();
        w["word"] = new_tokens[nj];
        auto it = equal_old_of_new.find(nj);
        if (it != equal_old_of_new.end()) {
            const json& ow = old_words[it->second];
            const json wstart = get_or_null(ow, "start");
            const json wend = get_or_null(ow, "end");
            if (!wstart.is_null() && !wend.is_null()) {
                w["start"] = wstart;
                w["end"] = wend;
                kept_timing = true;
            }
        }
        out.push_back(std::move(w));
    }
    if (!kept_timing) return json::array();
    interpolate_gaps(out, start, end);
    return out;
}

// --- coalesce ----------------------------------------------------------------

namespace {

double seg_dur(const json& seg) {
    const json s = get_or_null(seg, "start");
    const json e = get_or_null(seg, "end");
    if (!s.is_null() && !e.is_null()) return e.get<double>() - s.get<double>();
    return 0.0;
}

// _merge_segments(a, b): fuse b onto a (same speaker), fresh dict.
json merge_segments(const json& a, const json& b) {
    json merged = json::object();
    const json a_start = get_or_null(a, "start");
    merged["start"] = !a_start.is_null() ? a_start : get_or_null(b, "start");
    const json b_end = get_or_null(b, "end");
    merged["end"] = !b_end.is_null() ? b_end : get_or_null(a, "end");

    std::vector<std::string> parts;
    std::string ta = py_strip(get_str(a, "text"));
    std::string tb = py_strip(get_str(b, "text"));
    if (!ta.empty()) parts.push_back(std::move(ta));
    if (!tb.empty()) parts.push_back(std::move(tb));
    merged["text"] = py_join(parts, " ");

    json words = json::array();
    if (auto it = a.find("words"); it != a.end() && it->is_array()) {
        for (const json& w : *it) words.push_back(w);
    }
    if (auto it = b.find("words"); it != b.end() && it->is_array()) {
        for (const json& w : *it) words.push_back(w);
    }
    merged["words"] = std::move(words);

    const json a_spk = get_or_null(a, "speaker");
    if (!a_spk.is_null()) merged["speaker"] = a_spk;
    return merged;
}

// _coalesce_run: merge one same-speaker run until each output clears threshold.
json coalesce_run(const json& run, double threshold) {
    json out = json::array();
    json cur;
    bool has_cur = false;
    for (const json& seg : run) {
        cur = has_cur ? merge_segments(cur, seg) : seg;  // seg == dict(seg) copy
        has_cur = true;
        if (seg_dur(cur) >= threshold) {
            out.push_back(cur);
            has_cur = false;
        }
    }
    if (has_cur) {
        if (!out.empty()) {
            out[out.size() - 1] = merge_segments(out.back(), cur);
        } else {
            out.push_back(cur);
        }
    }
    return out;
}

}  // namespace

json coalesce_segments(const json& segments, double threshold) {
    json out = json::array();
    const int n = static_cast<int>(segments.size());
    int i = 0;
    while (i < n) {
        const json key = seg_key(segments[i]);
        int j = i;
        while (j < n && seg_key(segments[j]) == key) ++j;
        json run = json::array();
        for (int x = i; x < j; ++x) run.push_back(segments[x]);
        for (json& s : coalesce_run(run, threshold)) out.push_back(std::move(s));
        i = j;
    }
    return out;
}

// --- turn edits --------------------------------------------------------------

namespace {

void require_turn(long turn_index, const json& turns) {
    if (turn_index < 0 || turn_index >= static_cast<long>(turns.size())) {
        throw std::out_of_range("turn_index out of range: " +
                                std::to_string(turn_index));
    }
}

// Length of a UTF-8 string in UTF-16 code units — the unit the browser's
// Selection/Range API (and JS `string.length`) counts in, so split offsets line
// up with what the SPA sent. A codepoint above U+FFFF needs a surrogate pair (2).
long utf16_len(const std::string& s) {
    long n = 0;
    std::size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        std::uint32_t cp;
        int adv;
        if (c < 0x80) {
            cp = c;
            adv = 1;
        } else if ((c >> 5) == 0x6) {
            cp = c & 0x1F;
            adv = 2;
        } else if ((c >> 4) == 0xE) {
            cp = c & 0x0F;
            adv = 3;
        } else if ((c >> 3) == 0x1E) {
            cp = c & 0x07;
            adv = 4;
        } else {
            cp = c;
            adv = 1;  // stray byte: count as one unit, stay in sync
        }
        for (int k = 1; k < adv && i + k < s.size(); ++k)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        n += (cp > 0xFFFF) ? 2 : 1;
        i += adv;
    }
    return n;
}

}  // namespace

std::pair<json, json> apply_turn_edit(const json& segments, long turn_index,
                                      const std::string& new_text) {
    // Reject negatives explicitly (a stray -1 must not edit the last turn).
    if (turn_index < 0) {
        throw std::out_of_range("turn_index out of range: " +
                                std::to_string(turn_index));
    }
    const json turns = group_turns(segments);
    require_turn(turn_index, turns);
    const json& turn = turns[turn_index];
    const int i = turn["seg_indices"].front().get<int>();
    const int j = turn["seg_indices"].back().get<int>();

    json old_segments = json::array();
    for (int k = i; k <= j; ++k) old_segments.push_back(segments[k]);

    const std::string text = py_strip(new_text);
    json replacement = json::array();
    json new_segment(nullptr);
    if (!text.empty()) {
        json old_words = json::array();
        for (const json& idx : turn["seg_indices"]) {
            const json& seg = segments[idx.get<int>()];
            if (auto it = seg.find("words"); it != seg.end() && it->is_array()) {
                for (const json& w : *it) old_words.push_back(w);
            }
        }
        json words = realign_words(old_words, text, turn["start"], turn["end"]);
        json new_seg = json::object();
        new_seg["start"] = turn["start"];
        new_seg["end"] = turn["end"];
        new_seg["text"] = text;
        new_seg["words"] = std::move(words);
        if (!turn["speaker"].is_null()) new_seg["speaker"] = turn["speaker"];
        replacement.push_back(new_seg);
        new_segment = new_seg;
    }

    json new_segments = json::array();
    for (int k = 0; k < i; ++k) new_segments.push_back(segments[k]);
    for (json& s : replacement) new_segments.push_back(std::move(s));
    for (int k = j + 1; k < static_cast<int>(segments.size()); ++k) {
        new_segments.push_back(segments[k]);
    }

    json delta = json::object();
    delta["ts"] = now_iso();
    delta["turn_index"] = turn_index;
    delta["seg_range"] = json::array({i, j});
    delta["old_segments"] = std::move(old_segments);
    delta["new_segment"] = std::move(new_segment);
    return {std::move(new_segments), std::move(delta)};
}

std::pair<json, json> apply_turn_reassign(const json& segments, long turn_index,
                                          const std::string& new_speaker) {
    if (turn_index < 0) {
        throw std::out_of_range("turn_index out of range: " +
                                std::to_string(turn_index));
    }
    const json turns = group_turns(segments);
    require_turn(turn_index, turns);
    const json& turn = turns[turn_index];
    // turn.speaker == new_speaker (null speaker never equals a str -> no NoChange).
    if (turn["speaker"].is_string() &&
        turn["speaker"].get<std::string>() == new_speaker) {
        throw NoChange();
    }
    const int i = turn["seg_indices"].front().get<int>();
    const int j = turn["seg_indices"].back().get<int>();

    json old_segments = json::array();
    for (int k = i; k <= j; ++k) old_segments.push_back(segments[k]);

    json replacement = json::array();
    for (const json& seg : old_segments) {
        json new_seg = seg;
        new_seg["speaker"] = new_speaker;
        if (auto it = new_seg.find("words"); it != new_seg.end() &&
                                             it->is_array()) {
            for (json& w : *it) w["speaker"] = new_speaker;
        }
        replacement.push_back(std::move(new_seg));
    }

    json new_segments = json::array();
    for (int k = 0; k < i; ++k) new_segments.push_back(segments[k]);
    for (const json& s : replacement) new_segments.push_back(s);
    for (int k = j + 1; k < static_cast<int>(segments.size()); ++k) {
        new_segments.push_back(segments[k]);
    }

    json delta = json::object();
    delta["ts"] = now_iso();
    delta["turn_index"] = turn_index;
    delta["seg_range"] = json::array({i, j});
    delta["old_segments"] = old_segments;
    delta["new_len"] = static_cast<int>(replacement.size());
    return {std::move(new_segments), std::move(delta)};
}

std::pair<json, json> apply_turn_split(const json& segments, long turn_index,
                                       long sel_start, long sel_end,
                                       const std::string& new_speaker) {
    if (turn_index < 0) {
        throw std::out_of_range("turn_index out of range: " +
                                std::to_string(turn_index));
    }
    const json turns = group_turns(segments);
    require_turn(turn_index, turns);
    const json& turn = turns[turn_index];
    const json orig_speaker = turn["speaker"];  // string or null
    const int i = turn["seg_indices"].front().get<int>();
    const int j = turn["seg_indices"].back().get<int>();

    const json atoms = turn_atoms(segments, turn["seg_indices"]);
    const int n = static_cast<int>(atoms.size());

    // Map the UTF-16 offsets onto the atom range they cover. Atom k occupies
    // [pos, pos+len) in the space-joined text; the trailing +1 is the joiner.
    // An atom overlapping [sel_start, sel_end) is taken whole (partial-word
    // selections snap outward), so the middle is never empty for a real span.
    long pos = 0;
    int wi = -1, wj = -1;
    for (int k = 0; k < n; ++k) {
        const long len = utf16_len(atoms[k]["word"].get<std::string>());
        if (pos < sel_end && sel_start < pos + len) {
            if (wi < 0) wi = k;
            wj = k + 1;
        }
        pos += len + 1;
    }
    if (wi < 0 || wi >= wj) throw NoChange();  // empty / out-of-text selection
    // Moving the passage to the speaker it already has changes nothing (a null
    // turn speaker never equals a key, mirroring apply_turn_reassign).
    if (orig_speaker.is_string() && orig_speaker.get<std::string>() == new_speaker) {
        throw NoChange();
    }

    json old_segments = json::array();
    for (int k = i; k <= j; ++k) old_segments.push_back(segments[k]);

    // Build one segment from atoms [lo, hi) under `spk` (key omitted when null),
    // mirroring apply_turn_edit's new-seg shape: bounds from the first/last timed
    // atom (null when none), text from the tokens, words carrying key-presence timing.
    auto build_seg = [&](int lo, int hi, const json& spk) -> json {
        json words = json::array();
        std::vector<std::string> tokens;
        for (int k = lo; k < hi; ++k) {
            json w = json::object();
            w["word"] = atoms[k]["word"];
            if (atoms[k].contains("start") && atoms[k].contains("end")) {
                w["start"] = atoms[k]["start"];
                w["end"] = atoms[k]["end"];
            }
            tokens.push_back(atoms[k]["word"].get<std::string>());
            words.push_back(std::move(w));
        }
        json sstart(nullptr), send(nullptr);
        for (int k = lo; k < hi; ++k)
            if (atoms[k].contains("start")) { sstart = atoms[k]["start"]; break; }
        for (int k = hi - 1; k >= lo; --k)
            if (atoms[k].contains("end")) { send = atoms[k]["end"]; break; }
        json seg = json::object();
        seg["start"] = sstart;
        seg["end"] = send;
        seg["text"] = py_join(tokens, " ");
        seg["words"] = std::move(words);
        if (!spk.is_null()) seg["speaker"] = spk;
        return seg;
    };

    json replacement = json::array();
    if (wi > 0) replacement.push_back(build_seg(0, wi, orig_speaker));     // head
    replacement.push_back(build_seg(wi, wj, json(new_speaker)));           // middle
    if (wj < n) replacement.push_back(build_seg(wj, n, orig_speaker));     // tail

    json new_segments = json::array();
    for (int k = 0; k < i; ++k) new_segments.push_back(segments[k]);
    for (json& s : replacement) new_segments.push_back(std::move(s));
    for (int k = j + 1; k < static_cast<int>(segments.size()); ++k) {
        new_segments.push_back(segments[k]);
    }

    json delta = json::object();
    delta["ts"] = now_iso();
    delta["turn_index"] = turn_index;
    delta["seg_range"] = json::array({i, j});
    delta["old_segments"] = old_segments;
    delta["new_len"] = static_cast<int>(replacement.size());
    return {std::move(new_segments), std::move(delta)};
}

std::pair<json, json> undo_last(const json& segments, const json& history) {
    if (history.empty()) {
        return {segments, history};
    }
    json hist = history;
    const json delta = hist.back();
    hist.erase(hist.size() - 1);

    const int i = delta["seg_range"][0].get<int>();
    int repl_len;
    const json new_len = get_or_null(delta, "new_len");
    if (!new_len.is_null()) {
        repl_len = new_len.get<int>();
    } else {
        repl_len = !get_or_null(delta, "new_segment").is_null() ? 1 : 0;
    }

    const json& restored = delta["old_segments"];
    json new_segments = json::array();
    for (int k = 0; k < i; ++k) new_segments.push_back(segments[k]);
    for (const json& s : restored) new_segments.push_back(s);
    for (int k = i + repl_len; k < static_cast<int>(segments.size()); ++k) {
        new_segments.push_back(segments[k]);
    }
    return {std::move(new_segments), std::move(hist)};
}

}  // namespace whisperx::edits
