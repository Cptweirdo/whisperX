#include "align/align.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "align/interpolate.hpp"
#include "text/sentence_split.hpp"
#include "text/utf8.hpp"

namespace whisperx::align {

namespace {

// ja/zh have no word spaces (alignment.py:30) — word_idx advances every char and
// sentence text is joined without spaces.
bool nospaces(const std::string& lang) { return lang == "ja" || lang == "zh"; }

// Round to 3 decimals matching Python's round() — round-half-to-**even**
// (banker's). nearbyint uses the default FE_TONEAREST mode (ties to even), so
// 0.3125 → 0.312 like Python (std::round would give 0.313). Residual *1000 float
// error is sub-ulp and well under the ±1-frame timing tolerance.
double round3(double v) { return std::nearbyint(v * 1000.0) / 1000.0; }

// One enumerated codepoint of the segment text + its assigned timing/word.
struct CharCell {
    std::string ch;                  // the codepoint (UTF-8)
    bool is_space;                   // ch == " "
    std::optional<double> start;     // None == unaligned (NaN)
    std::optional<double> end;
    std::optional<double> score;
    int word_idx;
};

std::optional<double> min_opt(const std::vector<std::optional<double>>& xs) {
    std::optional<double> m;
    for (const auto& x : xs)
        if (x && (!m || *x < *m)) m = x;
    return m;
}
std::optional<double> max_opt(const std::vector<std::optional<double>>& xs) {
    std::optional<double> m;
    for (const auto& x : xs)
        if (x && (!m || *x > *m)) m = x;
    return m;
}
std::optional<double> mean_opt(const std::vector<std::optional<double>>& xs) {
    double sum = 0.0;
    int n = 0;
    for (const auto& x : xs)
        if (x) {
            sum += *x;
            ++n;
        }
    if (n == 0) return std::nullopt;
    return sum / n;
}

}  // namespace

AssembleResult align_assemble(const Emission& emission,
                              const std::vector<int>& tokens, int blank_id,
                              const std::string& text_clean,
                              const std::string& text,
                              const std::vector<int>& clean_cdx, double t1,
                              double t2, const std::string& language,
                              const std::string& interpolate_method,
                              bool return_char_alignments,
                              const std::optional<double>& avg_logprob) {
    std::vector<float> trellis = get_trellis(emission, tokens, blank_id);
    auto path = backtrack(trellis, emission, tokens, blank_id);
    if (!path) return {false, json::array()};

    std::vector<Segment> char_segments = merge_repeats(*path, text_clean);

    // ratio = duration * waveform.size(0) / (trellis.rows - 1); waveform.size(0)
    // is the channel dim (1), trellis.rows - 1 == emission frames (alignment.py:296).
    const double duration = t2 - t1;
    const double ratio =
        duration * 1.0 / static_cast<double>(emission.T);

    // clean_cdx[j] -> char_segments[j]
    std::unordered_map<int, std::size_t> cdx_to_j;
    cdx_to_j.reserve(clean_cdx.size());
    for (std::size_t j = 0; j < clean_cdx.size(); ++j)
        cdx_to_j.emplace(clean_cdx[j], j);

    // Enumerate text codepoints, assign char timings + word indices
    // (alignment.py:298-323).
    const std::vector<whisperx::text::Utf8Char> uchars =
        whisperx::text::utf8_chars(text);
    const int ncp = static_cast<int>(uchars.size());
    const bool ns = nospaces(language);
    std::vector<CharCell> cells(ncp);
    int word_idx = 0;
    for (int cdx = 0; cdx < ncp; ++cdx) {
        const auto& uc = uchars[cdx];
        std::string ch = text.substr(uc.offset, uc.length);
        CharCell cell;
        cell.ch = ch;
        cell.is_space = (ch == " ");
        cell.word_idx = word_idx;
        auto it = cdx_to_j.find(cdx);
        if (it != cdx_to_j.end()) {
            const Segment& cs = char_segments[it->second];
            cell.start = round3(cs.start * ratio + t1);
            cell.end = round3(cs.end * ratio + t1);
            cell.score = round3(static_cast<double>(cs.score));
        }
        cells[cdx] = std::move(cell);
        if (ns) {
            ++word_idx;
        } else if (cdx == ncp - 1 ||
                   text.substr(uchars[cdx + 1].offset, uchars[cdx + 1].length) ==
                       " ") {
            ++word_idx;
        }
    }

    // Sentence grouping via the native splitter (replaces nltk punkt).
    const auto spans = whisperx::text::sentence_spans(text, language);

    // Each subsegment, pre-groupby.
    struct Sub {
        std::string text;
        std::optional<double> start;
        std::optional<double> end;
        json words;
        json chars;  // optional char alignments
    };
    std::vector<Sub> subs;

    for (const auto& [sstart_sz, send_sz] : spans) {
        const int sstart = static_cast<int>(sstart_sz);
        const int send = static_cast<int>(send_sz);
        // curr_chars: alignment.py uses index >= sstart & index <= send (inclusive
        // send — a benign punkt-era quirk; the extra boundary char is whitespace
        // and is filtered out below). Mirror it exactly for golden parity.
        std::vector<int> curr;
        for (int cdx = sstart; cdx <= send && cdx < ncp; ++cdx) curr.push_back(cdx);

        // sentence_text = text[sstart:send] (send exclusive codepoint slice)
        std::string sentence_text;
        if (send > sstart && sstart < ncp) {
            const std::size_t b0 = uchars[sstart].offset;
            const std::size_t b1 = (send < ncp) ? uchars[send].offset : text.size();
            sentence_text = text.substr(b0, b1 - b0);
        }

        std::vector<std::optional<double>> starts_all, ends_nonspace;
        for (int cdx : curr) {
            starts_all.push_back(cells[cdx].start);
            if (!cells[cdx].is_space) ends_nonspace.push_back(cells[cdx].end);
        }
        std::optional<double> sentence_start = min_opt(starts_all);
        std::optional<double> sentence_end = max_opt(ends_nonspace);

        // distinct word indices in appearance order (word_idx is monotonic over
        // the contiguous cdx range, so a change from the previous == a new value).
        std::vector<int> word_order;
        for (int cdx : curr)
            if (word_order.empty() || word_order.back() != cells[cdx].word_idx)
                word_order.push_back(cells[cdx].word_idx);

        json sentence_words = json::array();
        std::vector<json> char_records;  // for return_char_alignments
        for (int wi : word_order) {
            std::string word_text;
            std::vector<std::optional<double>> wstart, wend, wscore;
            for (int cdx : curr) {
                if (cells[cdx].word_idx != wi) continue;
                word_text += cells[cdx].ch;
            }
            // strip()
            auto not_space = [](unsigned char c) {
                return !(c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                         c == '\f' || c == '\v');
            };
            std::size_t a = 0, b = word_text.size();
            while (a < b && !not_space(word_text[a])) ++a;
            while (b > a && !not_space(word_text[b - 1])) --b;
            word_text = word_text.substr(a, b - a);
            if (word_text.empty()) continue;

            // dont use space character for alignment
            for (int cdx : curr) {
                if (cells[cdx].word_idx != wi || cells[cdx].is_space) continue;
                wstart.push_back(cells[cdx].start);
                wend.push_back(cells[cdx].end);
                wscore.push_back(cells[cdx].score);
            }
            std::optional<double> ws = min_opt(wstart);
            std::optional<double> we = max_opt(wend);
            std::optional<double> wsc = mean_opt(wscore);

            json w = json::object();
            w["word"] = word_text;
            if (ws) w["start"] = *ws;
            if (we) w["end"] = *we;
            if (wsc) w["score"] = round3(*wsc);
            sentence_words.push_back(std::move(w));
        }

        // Interpolate timestamps for words with no alignable characters
        // (alignment.py:365-376).
        if (!sentence_words.empty()) {
            std::vector<std::optional<double>> ws, we;
            for (const auto& w : sentence_words) {
                ws.push_back(w.contains("start")
                                 ? std::optional<double>(w["start"].get<double>())
                                 : std::nullopt);
                we.push_back(w.contains("end")
                                 ? std::optional<double>(w["end"].get<double>())
                                 : std::nullopt);
            }
            int notna = 0, na = 0;
            for (const auto& v : ws) (v ? notna : na)++;
            if (na > 0 && notna > 0) {
                auto fs = interpolate_nans(ws, interpolate_method);
                auto fe = interpolate_nans(we, interpolate_method);
                for (std::size_t i = 0; i < sentence_words.size(); ++i) {
                    if (!sentence_words[i].contains("start") && fs[i])
                        sentence_words[i]["start"] = *fs[i];
                    if (!sentence_words[i].contains("end") && fe[i])
                        sentence_words[i]["end"] = *fe[i];
                }
            }
        }

        Sub sub;
        sub.text = sentence_text;
        sub.start = sentence_start;
        sub.end = sentence_end;
        sub.words = std::move(sentence_words);
        if (return_char_alignments) {
            json chars = json::array();
            for (int cdx : curr) {
                json c = json::object();
                c["char"] = cells[cdx].ch;
                if (cells[cdx].start) c["start"] = *cells[cdx].start;
                if (cells[cdx].end) c["end"] = *cells[cdx].end;
                if (cells[cdx].score) c["score"] = *cells[cdx].score;
                chars.push_back(std::move(c));
            }
            sub.chars = std::move(chars);
        }
        subs.push_back(std::move(sub));
    }

    // interpolate subsegment start/end across sentences (alignment.py:396-397)
    {
        std::vector<std::optional<double>> ss, ee;
        for (const auto& s : subs) {
            ss.push_back(s.start);
            ee.push_back(s.end);
        }
        ss = interpolate_nans(ss, interpolate_method);
        ee = interpolate_nans(ee, interpolate_method);
        for (std::size_t i = 0; i < subs.size(); ++i) {
            subs[i].start = ss[i];
            subs[i].end = ee[i];
        }
    }

    // groupby(["start","end"]).agg — merge subsegments sharing (start,end), sorted
    // ascending by key; NaN-keyed rows dropped (pandas default). text " ".join (or
    // "" for nospaces), words/chars summed, avg_logprob first.
    struct Group {
        double start, end;
        std::vector<std::size_t> members;
    };
    std::vector<Group> groups;
    for (std::size_t i = 0; i < subs.size(); ++i) {
        if (!subs[i].start || !subs[i].end) continue;  // dropna
        double s = *subs[i].start, e = *subs[i].end;
        auto g = std::find_if(groups.begin(), groups.end(), [&](const Group& gr) {
            return gr.start == s && gr.end == e;
        });
        if (g == groups.end())
            groups.push_back({s, e, {i}});
        else
            g->members.push_back(i);
    }
    std::stable_sort(groups.begin(), groups.end(), [](const Group& a, const Group& b) {
        return a.start < b.start || (a.start == b.start && a.end < b.end);
    });

    json out = json::array();
    const std::string joiner = ns ? "" : " ";
    for (const auto& g : groups) {
        json seg = json::object();
        std::string txt;
        json words = json::array();
        json chars = json::array();
        bool any_chars = false;
        for (std::size_t k = 0; k < g.members.size(); ++k) {
            const Sub& s = subs[g.members[k]];
            if (k) txt += joiner;
            txt += s.text;
            for (const auto& w : s.words) words.push_back(w);
            if (!s.chars.is_null()) {
                any_chars = true;
                for (const auto& c : s.chars) chars.push_back(c);
            }
        }
        seg["start"] = g.start;
        seg["end"] = g.end;
        seg["text"] = txt;
        seg["words"] = std::move(words);
        if (any_chars) seg["chars"] = std::move(chars);
        if (avg_logprob) seg["avg_logprob"] = *avg_logprob;
        out.push_back(std::move(seg));
    }

    return {true, std::move(out)};
}

}  // namespace whisperx::align
