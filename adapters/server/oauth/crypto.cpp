#include "oauth/crypto.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <bcrypt.h>
#elif defined(__APPLE__)
#include <cstdlib>  // arc4random_buf
#else
#include <sys/random.h>  // getrandom
#include <cerrno>
#endif

namespace whisperx::server::oauth {

namespace {

inline std::uint32_t rotr(std::uint32_t x, std::uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

// SHA-256 round constants (first 32 bits of the fractional parts of the cube
// roots of the first 64 primes) — FIPS 180-4.
const std::uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

}  // namespace

Sha256::Sha256()
    : h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19} {}

void Sha256::process(const std::uint8_t* block) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
               (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
               static_cast<std::uint32_t>(block[i * 4 + 3]);
    for (int i = 16; i < 64; ++i) {
        std::uint32_t s0 =
            rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        std::uint32_t s1 =
            rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4],
                  f = h_[5], g = h_[6], hh = h_[7];
    for (int i = 0; i < 64; ++i) {
        std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        std::uint32_t ch = (e & f) ^ (~e & g);
        std::uint32_t t1 = hh + S1 + ch + K[i] + w[i];
        std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        std::uint32_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
}

void Sha256::update(const std::uint8_t* data, std::size_t len) {
    bitlen_ += static_cast<std::uint64_t>(len) * 8;
    if (buflen_ > 0) {  // top up a partial block first
        std::size_t need = 64 - buflen_;
        std::size_t take = len < need ? len : need;
        std::memcpy(buf_ + buflen_, data, take);
        buflen_ += take;
        data += take;
        len -= take;
        if (buflen_ == 64) {
            process(buf_);
            buflen_ = 0;
        }
    }
    while (len >= 64) {
        process(data);
        data += 64;
        len -= 64;
    }
    if (len > 0) {
        std::memcpy(buf_, data, len);
        buflen_ = len;
    }
}

std::array<std::uint8_t, 32> Sha256::finish() {
    const std::uint64_t bitlen = bitlen_;
    std::uint8_t pad = 0x80;
    update(&pad, 1);
    std::uint8_t zero = 0;
    while (buflen_ != 56) update(&zero, 1);
    std::uint8_t lenbytes[8];
    for (int i = 0; i < 8; ++i)
        lenbytes[i] = static_cast<std::uint8_t>((bitlen >> ((7 - i) * 8)) & 0xff);
    // update() folds these into the final block; bitlen_ drift is irrelevant now.
    std::memcpy(buf_ + buflen_, lenbytes, 8);
    process(buf_);
    buflen_ = 0;

    std::array<std::uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i) {
        out[i * 4 + 0] = (h_[i] >> 24) & 0xff;
        out[i * 4 + 1] = (h_[i] >> 16) & 0xff;
        out[i * 4 + 2] = (h_[i] >> 8) & 0xff;
        out[i * 4 + 3] = h_[i] & 0xff;
    }
    return out;
}

std::array<std::uint8_t, 32> sha256(const std::string& msg) {
    Sha256 ctx;
    ctx.update(msg);
    return ctx.finish();
}

namespace {
std::string to_hex(const std::array<std::uint8_t, 32>& d) {
    static const char* H = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (std::uint8_t b : d) {
        out += H[b >> 4];
        out += H[b & 0xf];
    }
    return out;
}
}  // namespace

std::string sha256_hex(const std::string& data) { return to_hex(sha256(data)); }

std::string sha256_hex_file(const std::string& path) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("cannot open for hashing: " + path);
    Sha256 ctx;
    std::vector<std::uint8_t> buf(1024 * 1024);
    std::size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), fp)) > 0)
        ctx.update(buf.data(), n);
    bool err = std::ferror(fp) != 0;
    std::fclose(fp);
    if (err) throw std::runtime_error("read error while hashing: " + path);
    return to_hex(ctx.finish());
}

std::string base64url_nopad(const std::uint8_t* data, std::size_t len) {
    static const char* T =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    std::size_t i = 0;
    for (; i + 3 <= len; i += 3) {
        std::uint32_t n = (static_cast<std::uint32_t>(data[i]) << 16) |
                          (static_cast<std::uint32_t>(data[i + 1]) << 8) |
                          static_cast<std::uint32_t>(data[i + 2]);
        out += T[(n >> 18) & 63];
        out += T[(n >> 12) & 63];
        out += T[(n >> 6) & 63];
        out += T[n & 63];
    }
    if (len - i == 1) {
        std::uint32_t n = static_cast<std::uint32_t>(data[i]) << 16;
        out += T[(n >> 18) & 63];
        out += T[(n >> 12) & 63];
    } else if (len - i == 2) {
        std::uint32_t n = (static_cast<std::uint32_t>(data[i]) << 16) |
                          (static_cast<std::uint32_t>(data[i + 1]) << 8);
        out += T[(n >> 18) & 63];
        out += T[(n >> 12) & 63];
        out += T[(n >> 6) & 63];
    }
    return out;
}

std::string base64url_nopad(const std::string& data) {
    return base64url_nopad(reinterpret_cast<const std::uint8_t*>(data.data()),
                           data.size());
}

std::string random_bytes(std::size_t n) {
    std::string buf(n, '\0');
    auto* p = reinterpret_cast<unsigned char*>(buf.data());
#if defined(_WIN32)
    if (BCryptGenRandom(nullptr, p, static_cast<ULONG>(n),
                        BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0)
        throw std::runtime_error("BCryptGenRandom failed");
#elif defined(__APPLE__)
    arc4random_buf(p, n);
#else
    std::size_t got = 0;
    while (got < n) {
        ssize_t r = getrandom(p + got, n - got, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("getrandom failed");
        }
        got += static_cast<std::size_t>(r);
    }
#endif
    return buf;
}

}  // namespace whisperx::server::oauth
