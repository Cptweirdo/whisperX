#include "oauth/pkce.hpp"

#include "oauth/crypto.hpp"

namespace whisperx::server::oauth {

std::string code_verifier() { return base64url_nopad(random_bytes(32)); }

std::string code_challenge_s256(const std::string& verifier) {
    auto digest = sha256(verifier);
    return base64url_nopad(digest.data(), digest.size());
}

std::string random_state() { return base64url_nopad(random_bytes(16)); }

}  // namespace whisperx::server::oauth
