#include "text/sentence_split.hpp"

#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "text/utf8.hpp"
#include "whisperx_nonbreaking_prefixes.hpp"  // generated: kPrefixFiles[]

namespace whisperx::text {

namespace {

// Parsed Moses non-breaking-prefix data for one language.
struct PrefixSet {
    std::unordered_set<std::string> plain;    // suppress before any starter
    std::unordered_set<std::string> numeric;  // suppress only before a digit
};

// Parse one Moses nonbreaking_prefix.<lang> file body: skip blank/`#`-comment
// lines; `WORD` → plain; `WORD #NUMERIC_ONLY#` → numeric-only.
PrefixSet parse_prefixes(std::string_view body) {
    PrefixSet out;
    std::size_t i = 0;
    while (i < body.size()) {
        std::size_t eol = body.find('\n', i);
        if (eol == std::string_view::npos) eol = body.size();
        std::string_view line = body.substr(i, eol - i);
        i = eol + 1;
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (line.empty() || line.front() == '#') continue;
        // first whitespace-delimited token = prefix; "#NUMERIC_ONLY#" tag after it.
        std::size_t sp = line.find_first_of(" \t");
        std::string_view word = line.substr(0, sp);
        if (word.empty()) continue;
        bool numeric_only = sp != std::string_view::npos &&
                            line.find("#NUMERIC_ONLY#", sp) != std::string_view::npos;
        if (numeric_only)
            out.numeric.emplace(word);
        else
            out.plain.emplace(word);
    }
    return out;
}

const PrefixSet& prefixes_for(std::string_view lang) {
    static std::once_flag once;
    static std::unordered_map<std::string, PrefixSet> cache;
    std::call_once(once, [] {
        for (const auto& [l, body] : data::kPrefixFiles)
            cache.emplace(std::string(l), parse_prefixes(body));
    });
    auto it = cache.find(std::string(lang));
    if (it == cache.end()) it = cache.find("en");
    static const PrefixSet empty;
    return it == cache.end() ? empty : it->second;
}

// --- codepoint classification (fixed ranges, mirrored in Python) -------------
bool is_space(char32_t c) {
    return c == 0x09 || c == 0x0A || c == 0x0B || c == 0x0C || c == 0x0D ||
           c == 0x20 || c == 0xA0;
}
bool is_digit(char32_t c) { return c >= '0' && c <= '9'; }
bool is_upper(char32_t c) {
    return (c >= 'A' && c <= 'Z') ||                 // ASCII
           (c >= 0xC0 && c <= 0xDE && c != 0xD7) ||  // Latin-1 uppercase
           (c >= 0x400 && c <= 0x42F);               // Cyrillic Ѐ..Я (incl. Ё)
}
bool is_opening(char32_t c) {
    // quotes/brackets/inverted marks that can begin a sentence.
    return c == '"' || c == '\'' || c == '`' || c == '(' || c == '[' ||
           c == '{' || c == 0xAB /*«*/ || c == 0x2039 /*‹*/ ||
           c == 0x201C /*“*/ || c == 0x2018 /*‘*/ || c == 0xBF /*¿*/ ||
           c == 0xA1 /*¡*/;
}
bool is_starter(char32_t c) {
    return is_upper(c) || is_digit(c) || is_opening(c);
}
bool is_terminator(char32_t c) {
    return c == '.' || c == '!' || c == '?' || c == 0x2026 /*…*/;
}
bool is_closing(char32_t c) {
    return c == '"' || c == '\'' || c == ')' || c == ']' || c == '}' ||
           c == 0xBB /*»*/ || c == 0x201D /*”*/ || c == 0x2019 /*’*/ ||
           c == 0x203A /*›*/;
}

}  // namespace

std::vector<std::pair<std::size_t, std::size_t>> sentence_spans(
    std::string_view text, std::string_view lang) {
    // Decode to codepoints once; keep byte offsets so prefix tokens can be sliced
    // out of the original buffer for comparison.
    const std::vector<Utf8Char> uchars = utf8_chars(text);
    const std::size_t n = uchars.size();
    std::vector<char32_t> cp(n);
    for (std::size_t k = 0; k < n; ++k)
        cp[k] = utf8_decode(text, uchars[k].offset);

    auto tok_before = [&](std::size_t dot) -> std::string {
        // maximal non-space codepoint run ending just before index `dot`.
        std::size_t a = dot;
        while (a > 0 && !is_space(cp[a - 1])) --a;
        if (a >= dot) return {};
        std::size_t b0 = uchars[a].offset;
        std::size_t b1 = uchars[dot - 1].offset + uchars[dot - 1].length;
        return std::string(text.substr(b0, b1 - b0));
    };

    const PrefixSet& pfx = prefixes_for(lang);
    std::vector<std::pair<std::size_t, std::size_t>> spans;

    // first non-space codepoint
    std::size_t sent_start = 0;
    while (sent_start < n && is_space(cp[sent_start])) ++sent_start;
    if (sent_start >= n) return spans;  // empty / whitespace-only

    std::size_t i = sent_start;
    while (i < n) {
        if (!is_terminator(cp[i])) {
            ++i;
            continue;
        }
        // extend over consecutive terminators + closing chars: [i, k)
        std::size_t k = i + 1;
        while (k < n && (is_terminator(cp[k]) || is_closing(cp[k]))) ++k;
        // require whitespace then a sentence starter
        std::size_t w = k;
        while (w < n && is_space(cp[w])) ++w;
        const bool has_ws = w > k;
        if (has_ws && w < n && is_starter(cp[w])) {
            bool suppress = false;
            if (cp[i] == '.') {
                const std::string tok = tok_before(i);
                if (pfx.plain.count(tok)) {
                    suppress = true;
                } else if (pfx.numeric.count(tok) && is_digit(cp[w])) {
                    suppress = true;
                }
            }
            if (!suppress) {
                spans.emplace_back(sent_start, k);
                sent_start = w;
                i = w;
                continue;
            }
        }
        i = k;
    }
    // trailing sentence (trim trailing whitespace)
    std::size_t end = n;
    while (end > sent_start && is_space(cp[end - 1])) --end;
    if (end > sent_start) spans.emplace_back(sent_start, end);
    return spans;
}

}  // namespace whisperx::text
