// Live HF token verification — the C++ port of secret_store.py::verify_token.
// Confirms (1) the token is valid (whoami) and (2) it can access the gated
// diarization model (its user conditions are accepted) — the most common real
// failure. CPU-sherpa diarization is token-free, so this is for onboarding UX
// parity with the Flask host, not a runtime gate. Uses net::curl_client.
#pragma once

#include <string>
#include <utility>

namespace whisperx::server::secrets {

// Returns (ok, user-facing detail). The gated model is WHISPERX_DIARIZE_MODEL
// (default pyannote/speaker-diarization-community-1).
std::pair<bool, std::string> verify_token(const std::string& token);

}  // namespace whisperx::server::secrets
