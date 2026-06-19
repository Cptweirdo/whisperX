// Dependency-free crypto primitives for the OAuth2 PKCE flow. We do NOT couple to
// OpenSSL here: libcurl pulls in whatever TLS backend the platform ships
// (OpenSSL / GnuTLS / Secure Transport), so reaching for a specific one for a
// SHA-256 + base64url + CSPRNG would be fragile. SHA-256 is a small, self-checked
// implementation (verified against the RFC 7636 Appendix-B PKCE vector in the
// tests); randomness comes from the OS CSPRNG.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace whisperx::server::oauth {

// Incremental SHA-256 (FIPS 180-4) so large files hash without loading whole.
// Feed bytes with update(); finish() returns the 32-byte digest and consumes
// the context.
class Sha256 {
public:
    Sha256();
    void update(const std::uint8_t* data, std::size_t len);
    void update(const std::string& s) {
        update(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
    }
    std::array<std::uint8_t, 32> finish();

private:
    void process(const std::uint8_t* block);
    std::uint32_t h_[8];
    std::uint64_t bitlen_ = 0;
    std::uint8_t buf_[64];
    std::size_t buflen_ = 0;
};

// SHA-256 of an arbitrary byte string → 32 raw bytes.
std::array<std::uint8_t, 32> sha256(const std::string& data);

// SHA-256 as lowercase hex — the content-addressed object-store key shape.
std::string sha256_hex(const std::string& data);
// Streamed SHA-256 of a file → lowercase hex (1 MiB chunks). Throws
// std::runtime_error if the file can't be read.
std::string sha256_hex_file(const std::string& path);

// URL-safe base64 with no '=' padding (RFC 4648 §5) — the PKCE/JWT alphabet.
std::string base64url_nopad(const std::uint8_t* data, std::size_t len);
std::string base64url_nopad(const std::string& data);

// `n` cryptographically-random bytes from the OS CSPRNG (getrandom /
// arc4random_buf / BCryptGenRandom). Throws std::runtime_error if the OS RNG
// cannot be read — we never silently fall back to a weak source for tokens.
std::string random_bytes(std::size_t n);

// `n` CSPRNG bytes as 2n lowercase hex chars — collision-safe id material.
std::string random_hex(std::size_t n);

}  // namespace whisperx::server::oauth
