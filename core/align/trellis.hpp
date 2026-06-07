// C++ port of the forced-alignment Viterbi core from whisperx/alignment.py
// (get_trellis :425 / backtrack :455 / merge_repeats :508 / merge_words :526),
// itself from the torchaudio forced-alignment tutorial.
//
// Pure and deterministic given the CTC emission matrix: this slice (Phase 3A)
// takes the post-log_softmax, wildcard-extended emission as a *fixed* fp32 input
// (the decoupled-golden seam — the torch/ORT model forward stays out of here) and
// reproduces the integer trellis path + char-segments bit-for-bit. All arithmetic
// is float32 to match torch's float32 tensors element-for-element.
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace whisperx::align {

// One backtracked point (mirrors alignment.py::Point): which token, which audio
// frame, and the frame-wise probability (exp of the chosen emission).
struct Point {
    int token_index;
    int time_index;
    float score;
};

// One merged label run (mirrors alignment.py::Segment): a char label spanning
// [start, end) frames with the mean point score. `length() == end - start`.
struct Segment {
    std::string label;
    int start;
    int end;
    float score;
    int length() const { return end - start; }
};

// A row-major CTC emission view: `T` frames × `V` labels, log-softmax'd. Stored
// as a flat float32 buffer (data[t * V + v]) so it maps straight onto a numpy
// (T, V) array crossing the pybind seam.
struct Emission {
    const float* data;
    std::size_t T;
    std::size_t V;
    float at(std::size_t t, std::size_t v) const { return data[t * V + v]; }
};

// get_trellis — the (T+1)×(J+1) Viterbi cost matrix (J = tokens.size()), returned
// row-major. Reproduces alignment.py:425-445 exactly (cumsum blank column, ±inf
// boundaries, the maximum(stay, change) recurrence).
std::vector<float> get_trellis(const Emission& emission,
                               const std::vector<int>& tokens, int blank_id = 0);

// backtrack — walk the trellis from argmax of its last column back to the start,
// emitting one Point per frame consumed. Returns std::nullopt on failure (frames
// exhausted before all tokens consumed — alignment.py's `else: return None`).
std::optional<std::vector<Point>> backtrack(const std::vector<float>& trellis,
                                            const Emission& emission,
                                            const std::vector<int>& tokens,
                                            int blank_id = 0);

// merge_repeats — collapse consecutive same-token points into char Segments,
// labelled by transcript[token_index]. `transcript` is the cleaned char string
// (UTF-8); each token indexes a *codepoint* of it (see the .cpp note).
std::vector<Segment> merge_repeats(const std::vector<Point>& path,
                                   const std::string& transcript);

// merge_words — group char Segments into word Segments on the `separator` label
// ("|"). Not on align()'s golden path (it does its own word assembly) but ported
// + tested for completeness / reuse.
std::vector<Segment> merge_words(const std::vector<Segment>& segments,
                                 const std::string& separator = "|");

}  // namespace whisperx::align
