// Percent-encoding (RFC 3986 application/x-www-form-urlencoded) — a standalone
// port of oatpp::encoding::Url (src/oatpp/encoding/Url.{cpp,hpp}, Apache-2.0)
// rewritten on std::string instead of oat++ streams/Hex so it has no oat++
// dependency. Config mirrors upstream: an allow-list of unescaped bytes,
// optional space<->'+' mapping, and a selectable hex alphabet.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace whisperx::server::encoding {

class Url {
public:
    // Uppercase/lowercase hex digit tables for percent-escapes.
    static const char* const ALPHABET_UPPER;  // "0123456789ABCDEF"
    static const char* const ALPHABET_LOWER;  // "0123456789abcdef"

    struct Config {
        // Map ' ' <-> '+' (form-urlencoded). When false, space is %20.
        bool spaceToPlus = false;
        // Alphabet used to render percent-escapes on encode.
        const char* hexAlphabet = ALPHABET_UPPER;
        // Bytes that pass through unescaped. Default: RFC 3986 unreserved set
        // (ALPHA / DIGIT / '-' / '.' / '_' / '~').
        bool allowedChars[256];

        Config();
        void allowChar(std::uint8_t c);
        void allowCharRange(std::uint8_t from, std::uint8_t to);
        void disallowChar(std::uint8_t c);
        void disallowCharRange(std::uint8_t from, std::uint8_t to);
    };

    // Append the percent-encoding of [data, data+size) to out.
    static void encode(std::string& out, const void* data, std::size_t size,
                       const Config& config);
    // Append the percent-decoding of [data, data+size) to out.
    static void decode(std::string& out, const void* data, std::size_t size);

    static std::string encode(const std::string& data, const Config& config);
    static std::string decode(const std::string& data);
};

}  // namespace whisperx::server::encoding
