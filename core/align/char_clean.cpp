#include "align/char_clean.hpp"

#include <utf8proc.h>

#include "text/utf8.hpp"

namespace whisperx::align {

namespace {

// ja/zh have no word spaces (alignment.py:31) — keep each char, no ' '->'|'.
bool nospaces(const std::string& lang) { return lang == "ja" || lang == "zh"; }

// Python str whitespace set (str.lstrip/rstrip with no args == Py_UNICODE_ISSPACE).
// The realistic transcript domain is ASCII spaces; the rarer Unicode spaces are
// included so num_leading/num_trailing match Python exactly regardless.
bool py_isspace(char32_t c) {
    switch (c) {
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D:
        case 0x1C: case 0x1D: case 0x1E: case 0x1F: case 0x20:
        case 0x85: case 0xA0: case 0x1680:
        case 0x2000: case 0x2001: case 0x2002: case 0x2003: case 0x2004:
        case 0x2005: case 0x2006: case 0x2007: case 0x2008: case 0x2009:
        case 0x200A: case 0x2028: case 0x2029: case 0x202F: case 0x205F:
        case 0x3000:
            return true;
        default:
            return false;
    }
}

}  // namespace

std::string lower_utf8(char32_t cp) {
    utf8proc_int32_t lo = utf8proc_tolower(static_cast<utf8proc_int32_t>(cp));
    utf8proc_uint8_t buf[4];
    utf8proc_ssize_t n = utf8proc_encode_char(lo, buf);
    if (n <= 0) return {};
    return std::string(reinterpret_cast<char*>(buf), static_cast<std::size_t>(n));
}

CleanResult clean_segment(const std::string& text, const std::string& language,
                          const std::map<std::string, int>& dictionary) {
    const std::vector<text::Utf8Char> chars = text::utf8_chars(text);
    const std::size_t n = chars.size();
    const bool no_spaces = nospaces(language);

    // num_leading / num_trailing — Python len(text) - len(text.lstrip()/rstrip()),
    // counted in codepoints.
    std::size_t num_leading = 0;
    while (num_leading < n &&
           py_isspace(text::utf8_decode(text, chars[num_leading].offset)))
        ++num_leading;
    std::size_t num_trailing = 0;
    while (num_trailing < n &&
           py_isspace(text::utf8_decode(
               text, chars[n - 1 - num_trailing].offset)))
        ++num_trailing;

    CleanResult out;
    for (std::size_t cdx = 0; cdx < n; ++cdx) {
        char32_t cp = text::utf8_decode(text, chars[cdx].offset);
        std::string ch = lower_utf8(cp);            // char.lower()
        if (!no_spaces && ch == " ") ch = "|";       // spaces -> wav2vec2 "|"

        if (cdx < num_leading) {
            // leading whitespace — skip
        } else if (cdx + num_trailing >= n) {
            // trailing whitespace — skip (cdx > n - num_trailing - 1)
        } else if (dictionary.count(ch)) {
            out.clean_char += ch;
            out.clean_cdx.push_back(static_cast<int>(cdx));
        } else if (ch != " " && ch != "|") {
            // OOV (digit/symbol/foreign script) — kept, mapped to the wildcard later
            out.clean_char += ch;
            out.clean_cdx.push_back(static_cast<int>(cdx));
        }
    }
    return out;
}

}  // namespace whisperx::align
