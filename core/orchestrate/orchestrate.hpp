// Native end-to-end orchestrator — the C++ replacement for the *sequencing* of
// app/pipeline.py::run_job (pipeline.py:506-584). Decodes once into an AudioBuffer
// and drives every stage over it (decode -> VAD/ASR -> align -> diarize/assign)
// with no Python re-entry for the compute path: the buffer, segments and emissions
// never round-trip across the pybind seam between stages.
//
// The stage *operations* are injected as std::function callbacks (`Steps`), so this
// translation unit is pure sequencing — no ORT/sherpa/ffmpeg deps. It lives in the
// always-built whisperx_core_lib and is unit-tested dep-free with stubbed steps
// (the exact stage order, the diarizer-optional branch, on_duration-once, and
// stage-boundary cancellation). The pybind `run_job` (whisperx_core_audio) supplies
// the real closures over load_audio / silero_segments+merge_chunks / WhisperSherpa /
// align_run / SherpaDiarizer + assign_word_speakers.
#pragma once

#include <functional>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "audio/audio_buffer.hpp"

namespace whisperx::orchestrate {

using nlohmann::json;
using whisperx::audio::AudioBuffer;

// The injected per-stage compute. Each runs while the GIL is released in the pybind
// path; `load_align` is the one step that re-enters Python (model resolution at the
// loading_align boundary — I/O, not compute). `diarize` empty => no diarizer (skip
// the diarizing stage; mirrors `bundle.diarize is None`).
struct Steps {
    // "decoding": decode the audio file once -> the shared buffer.
    std::function<AudioBuffer()> decode;
    // "transcribing": VAD + merge_chunks + ASR -> (transcript segments array,
    // detected language). The pair mirrors run_job's `result["segments"]` + `lang`.
    std::function<std::pair<json, std::string>(const AudioBuffer&)> transcribe;
    // "loading_align": resolve/cache the align model for the detected language
    // (side effect — the resolved handle is captured by `align`). Optional.
    std::function<void(const std::string& language)> load_align;
    // "aligning": force-align the transcript -> {segments, word_segments}.
    std::function<json(const AudioBuffer&, const json& transcript,
                       const std::string& language)>
        align;
    // "diarizing": diarize + assign_word_speakers -> the result with speaker labels.
    // Empty std::function => skip (no diarizer).
    std::function<json(const AudioBuffer&, json aligned)> diarize;
};

// Run the pipeline. Fires `stage(name)` before each stage in run_job's exact order
// (decoding -> transcribing -> loading_align -> aligning -> [diarizing]); `stage`
// may throw to cancel at a boundary (never checked mid-stage). `on_duration` fires
// once, right after decode, with the clip duration in seconds. Returns the result
// json `{segments, word_segments, language, diarized}` (the caller adds duration /
// num_segments / artifacts).
json run_job(const Steps& steps,
             const std::function<void(const std::string&)>& stage,
             const std::function<void(double)>& on_duration);

}  // namespace whisperx::orchestrate
