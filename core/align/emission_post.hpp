// Post-processing of a raw wav2vec2-CTC forward (Phase 3B, "path 2"): the steps
// that in Python live at alignment.py:285,294-305, moved into C++ so the mirror-lang
// align path is torch-free end to end. Takes the model's **raw logits** (what
// model.onnx emits) and produces exactly what the 3A assembler (align_assemble)
// consumes: a log-softmax'd, optionally wildcard-extended emission + the token ids.
//
// Pure + deterministic (no ORT, no torch) so it lives in the dep-free core lib and
// is Catch2-testable in isolation. Compiled -ffp-contract=off (see CMakeLists) — the
// log_softmax is a new fp32 parity surface, budgeted by emission_atol.
#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace whisperx::align {

struct PostResult {
    std::vector<float> emission;  // (T, V') row-major, log-softmax'd (+wildcard col)
    std::size_t T;                // frames
    std::size_t V;                // V' = labels, or labels+1 when wildcard-extended
    std::vector<int> tokens;      // per-codepoint token id for each char of text_clean
};

// Reproduce the Python post-forward steps:
//   1. log_softmax over the label axis (stable max-subtract, fp32) — alignment.py:285.
//   2. if any codepoint of `text_clean` is outside `dictionary` (OOV), append the
//      max-non-blank column (over the log-softmax values) and map OOV codepoints to
//      it — alignment.py:294-302.
//   3. tokenize: each codepoint of `text_clean` -> its dictionary id (OOV -> wildcard).
// `text_clean` is already lower-cased / space->'|' (the Python char-cleaning that
// stays in the facade); each codepoint is looked up as a UTF-8 substring key.
PostResult emission_post(const float* logits, std::size_t T, std::size_t V,
                         int blank_id, const std::string& text_clean,
                         const std::map<std::string, int>& dictionary);

}  // namespace whisperx::align
