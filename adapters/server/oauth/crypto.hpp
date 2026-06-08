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

// SHA-256 of an arbitrary byte string → 32 raw bytes.
std::array<std::uint8_t, 32> sha256(const std::string& data);

// URL-safe base64 with no '=' padding (RFC 4648 §5) — the PKCE/JWT alphabet.
std::string base64url_nopad(const std::uint8_t* data, std::size_t len);
std::string base64url_nopad(const std::string& data);

// `n` cryptographically-random bytes from the OS CSPRNG (getrandom /
// arc4random_buf / BCryptGenRandom). Throws std::runtime_error if the OS RNG
// cannot be read — we never silently fall back to a weak source for tokens.
std::string random_bytes(std::size_t n);

}  // namespace whisperx::server::oauth
