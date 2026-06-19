// Native align driver — the C++ replacement for the *whole* whisperx/alignment.py
// ::align body (alignment.py:221-560) for ONNX (mirror-language) models. Makes
// align() a true native entrypoint (no Python re-entry), the prerequisite for the
// 100%-native orchestrator. Owns char-cleaning (char_clean), the gather + batched
// ORT forward (Wav2Vec2Onnx), and the per-segment emission_post -> align_assemble
// (3A/3B pieces, reused). Model *loading* + the torch forward path stay Python.
//
// Built only under WHISPERX_CORE_AUDIO (it drives the ORT forward).
#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <span>
#include <string>

#include <nlohmann/json.hpp>

#include "align/wav2vec2_onnx.hpp"

namespace whisperx::align {

using nlohmann::json;

// Align `transcript` (a {"segments": [...]} dict, or a bare segment array) against
// `audio` (16 kHz mono f32 PCM) using the already-loaded ONNX `model` + its
// `dictionary`. Reproduces align()'s output: {"segments": [...], "word_segments":
// [...]}. `batchable` packs padded+masked batches (layer_norm models only).
// `progress`, when set, fires ((sdx+1)/N)*100 after each *successfully aligned*
// segment (matching the Python callback — stubs don't fire it).
json align_run(const json& transcript, Wav2Vec2Onnx& model,
               const std::map<std::string, int>& dictionary,
               std::span<const float> audio, const std::string& language,
               bool batchable, const std::string& interpolate_method,
               bool return_char_alignments,
               const std::function<void(double)>& progress);

}  // namespace whisperx::align
