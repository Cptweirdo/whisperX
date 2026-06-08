// Implementation of the native run_job sequencer. See orchestrate.hpp. Pure
// control flow over injected steps — mirrors app/pipeline.py::run_job:537-570.
#include "orchestrate/orchestrate.hpp"

#include <utility>

namespace whisperx::orchestrate {

json run_job(const Steps& steps,
             const std::function<void(const std::string&)>& stage,
             const std::function<void(double)>& on_duration) {
    auto fire = [&stage](const char* name) {
        if (stage) stage(name);  // may throw (Cancelled) -> unwinds at the boundary
    };

    fire("decoding");
    AudioBuffer buf = steps.decode();
    if (on_duration) on_duration(buf.duration_s());

    fire("transcribing");
    auto [segments, language] = steps.transcribe(buf);

    fire("loading_align");
    if (steps.load_align) steps.load_align(language);

    fire("aligning");
    json result = steps.align(buf, segments, language);

    if (steps.diarize) {
        fire("diarizing");
        result = steps.diarize(buf, std::move(result));
        result["diarized"] = true;
    } else {
        result["diarized"] = false;
    }

    result["language"] = language;
    return result;
}

}  // namespace whisperx::orchestrate
