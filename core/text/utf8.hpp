// Minimal UTF-8 helpers shared by the alignment char indexing and the sentence
// splitter — both need to walk a byte string by *codepoint* (Python str indexing
// semantics) without pulling in ICU. Header-only.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace whisperx::text {

// Byte length of the UTF-8 codepoint starting at lead byte `c` (1–4). Returns 1
// for invalid/continuation lead bytes so callers always advance (lenient, like
// Python's surrogate-pass would never produce here — inputs are valid UTF-8).
inline std::size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x06) return 2;  // 110xxxxx
    if ((c >> 4) == 0x0E) return 3;  // 1110xxxx
    if ((c >> 3) == 0x1E) return 4;  // 11110xxx
    return 1;
}

// Split a UTF-8 string into its codepoints, each as a (offset, byte-length) pair
// over the original buffer. `chars[i]` is the i-th codepoint — indexable like a
// Python str. Offsets let callers recover byte spans (the splitter's contract).
struct Utf8Char {
    std::size_t offset;  // byte offset into the source string
    std::size_t length;  // byte length of this codepoint
};

inline std::vector<Utf8Char> utf8_chars(std::string_view s) {
    std::vector<Utf8Char> out;
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t n = utf8_len(static_cast<unsigned char>(s[i]));
        if (i + n > s.size()) n = s.size() - i;  // truncated tail — clamp
        out.push_back({i, n});
        i += n;
    }
    return out;
}

// Decode the codepoint at byte offset `i` to its Unicode scalar value (for
// classification — capital/letter/digit tests). Advances nothing; pair with
// utf8_len. Returns the lead byte itself for invalid sequences.
inline char32_t utf8_decode(std::string_view s, std::size_t i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    std::size_t n = utf8_len(c);
    if (n == 1 || i + n > s.size()) return c;
    char32_t cp = c & (0xFF >> (n + 1));
    for (std::size_t k = 1; k < n; ++k)
        cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    return cp;
}

}  // namespace whisperx::text
