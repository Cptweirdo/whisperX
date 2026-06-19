// PKCE (RFC 7636) + CSRF state helpers for the auth-code flow.
#pragma once

#include <string>

namespace whisperx::server::oauth {

// A high-entropy code_verifier — 43 url-safe chars (base64url of 32 random
// bytes), well within the RFC 7636 43–128 range and entirely in the unreserved
// set, so it never needs escaping.
std::string code_verifier();

// The S256 code_challenge for a verifier: base64url(SHA-256(verifier)), no pad.
std::string code_challenge_s256(const std::string& verifier);

// An opaque anti-CSRF `state` value (base64url of 16 random bytes).
std::string random_state();

}  // namespace whisperx::server::oauth
