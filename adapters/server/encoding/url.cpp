#include "encoding/url.hpp"

namespace whisperx::server::encoding {

const char* const Url::ALPHABET_UPPER = "0123456789ABCDEF";
const char* const Url::ALPHABET_LOWER = "0123456789abcdef";

namespace {

// Decode one hex nibble; returns -1 for non-hex (lenient: caller treats as 0).
int nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

}  // namespace

Url::Config::Config() {
    disallowCharRange(0, 255);
    allowCharRange('0', '9');
    allowCharRange('a', 'z');
    allowCharRange('A', 'Z');
    allowChar('-');
    allowChar('.');
    allowChar('_');
    allowChar('~');
}

void Url::Config::allowChar(std::uint8_t c) { allowedChars[c] = true; }

void Url::Config::allowCharRange(std::uint8_t from, std::uint8_t to) {
    for (int c = from; c <= to; ++c) allowedChars[c] = true;
}

void Url::Config::disallowChar(std::uint8_t c) { allowedChars[c] = false; }

void Url::Config::disallowCharRange(std::uint8_t from, std::uint8_t to) {
    for (int c = from; c <= to; ++c) allowedChars[c] = false;
}

void Url::encode(std::string& out, const void* data, std::size_t size,
                 const Config& config) {
    auto pdata = reinterpret_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        unsigned char c = pdata[i];
        if (config.allowedChars[c]) {
            out += static_cast<char>(c);
        } else if (c == ' ' && config.spaceToPlus) {
            out += '+';
        } else {
            out += '%';
            out += config.hexAlphabet[c >> 4];
            out += config.hexAlphabet[c & 0x0F];
        }
    }
}

void Url::decode(std::string& out, const void* data, std::size_t size) {
    auto pdata = reinterpret_cast<const char*>(data);
    std::size_t i = 0;
    while (i < size) {
        char c = pdata[i];
        if (c == '%') {
            if (size - i > 2) {
                int hi = nibble(pdata[i + 1]);
                int lo = nibble(pdata[i + 2]);
                if (hi < 0) hi = 0;
                if (lo < 0) lo = 0;
                out += static_cast<char>((hi << 4) | lo);
                i += 3;
            } else {
                break;
            }
        } else if (c == '+') {
            out += ' ';
            ++i;
        } else {
            out += c;
            ++i;
        }
    }
}

std::string Url::encode(const std::string& data, const Config& config) {
    std::string out;
    out.reserve(data.size());
    encode(out, data.data(), data.size(), config);
    return out;
}

std::string Url::decode(const std::string& data) {
    std::string out;
    out.reserve(data.size());
    decode(out, data.data(), data.size());
    return out;
}

}  // namespace whisperx::server::encoding
