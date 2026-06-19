// C++ port of whisperx/vads/vad.py::Vad.merge_chunks — packs VAD speech segments
// into ≤chunk_size windows (the "merge operation" from the WhisperX paper).
//
// Pure and model-agnostic: it consumes the raw speech segments any VAD backend
// produces (silero / pyannote) and is exercised behind the `vad` token. Per the
// decoupled-golden rule the parity gate feeds it a *fixed* segment list (the raw
// pre-merge segments dumped as a golden input), so any drift is a real bug.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace whisperx::vad {

using nlohmann::json;

// One VAD speech segment (mirrors whisperx.diarize.Segment: start/end seconds +
// an optional speaker label). merge_chunks reads only start/end; speaker rides
// along for fidelity with the Python input and is not emitted.
struct VadSegment {
    double start;
    double end;
    std::optional<std::string> speaker;
};

// merge_chunks(segments, chunk_size, onset, offset) -> array of merged chunks,
// each {start:num, end:num, segments:[[s,e], …]}. `onset`/`offset` are accepted
// for signature fidelity with the Python static method but are unused by the
// algorithm. Returns [] for an empty input (Python's Silero/Pyannote wrappers
// guard this before calling; the bare Vad.merge_chunks would index [0]).
json merge_chunks(const std::vector<VadSegment>& segments, double chunk_size,
                  double onset = 0.5,
                  std::optional<double> offset = std::nullopt);

}  // namespace whisperx::vad
