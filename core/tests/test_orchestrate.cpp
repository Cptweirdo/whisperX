// Native unit tests for the run_job sequencer (core/orchestrate): the stage order,
// the diarizer-optional branch, on_duration-once, the result shape, and
// stage-boundary cancellation — all with STUBBED steps so the control flow is
// pinned dep-free (no ORT/sherpa). The real engine assembly is covered by the
// opt-in bindings/test/test_orchestrate_parity.py against the staged native path.
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <string>
#include <vector>

#include "orchestrate/orchestrate.hpp"

using namespace whisperx::orchestrate;
using whisperx::audio::AudioBuffer;
using nlohmann::json;

namespace {

// A buffer of `n` samples at 16 kHz (duration = n/16000 s).
AudioBuffer make_buf(std::size_t n) {
    AudioBuffer b;
    b.samples.assign(n, 0.0f);
    b.sample_rate = 16000;
    return b;
}

// Steps that record their invocation order into `log` and return canned data.
struct Recorder {
    std::vector<std::string>& log;
    json segments = json::array({{{"text", "hi"}, {"start", 0.0}, {"end", 1.0}}});
    json aligned = json{{"segments", json::array({{{"text", "hi"}}})},
                        {"word_segments", json::array({{{"word", "hi"}}})}};

    Steps steps(bool with_diarize) {
        Steps s;
        s.decode = [this] {
            log.emplace_back("decode");
            return make_buf(32000);  // 2.0 s
        };
        s.transcribe = [this](const AudioBuffer&) {
            log.emplace_back("transcribe");
            return std::make_pair(segments, std::string("en"));
        };
        s.load_align = [this](const std::string& lang) {
            log.emplace_back("load_align:" + lang);
        };
        s.align = [this](const AudioBuffer&, const json&, const std::string&) {
            log.emplace_back("align");
            return aligned;
        };
        if (with_diarize)
            s.diarize = [this](const AudioBuffer&, json a) {
                log.emplace_back("diarize");
                a["segments"][0]["speaker"] = "SPEAKER_00";
                return a;
            };
        return s;
    }
};

}  // namespace

TEST_CASE("run_job: full stage order with a diarizer") {
    std::vector<std::string> log;
    std::vector<std::string> stages;
    std::vector<double> durations;
    Recorder rec{log};

    json out = run_job(
        rec.steps(/*with_diarize=*/true),
        [&](const std::string& s) { stages.emplace_back(s); },
        [&](double d) { durations.push_back(d); });

    CHECK(stages == std::vector<std::string>{"decoding", "transcribing",
                                             "loading_align", "aligning",
                                             "diarizing"});
    CHECK(log == std::vector<std::string>{"decode", "transcribe", "load_align:en",
                                          "align", "diarize"});
    // on_duration fires exactly once, right after decode, with len/SR.
    REQUIRE(durations.size() == 1);
    CHECK(durations[0] == 2.0);
    CHECK(out["language"] == "en");
    CHECK(out["diarized"] == true);
    CHECK(out["segments"][0]["speaker"] == "SPEAKER_00");
    CHECK(out.contains("word_segments"));
}

TEST_CASE("run_job: no diarizer skips diarizing and sets diarized=false") {
    std::vector<std::string> log;
    std::vector<std::string> stages;
    Recorder rec{log};

    json out = run_job(
        rec.steps(/*with_diarize=*/false),
        [&](const std::string& s) { stages.emplace_back(s); }, nullptr);

    CHECK(stages == std::vector<std::string>{"decoding", "transcribing",
                                             "loading_align", "aligning"});
    CHECK(log.back() == "align");  // no "diarize"
    CHECK(out["diarized"] == false);
    CHECK_FALSE(out["segments"][0].contains("speaker"));
    CHECK(out["language"] == "en");
}

TEST_CASE("run_job: a throwing stage callback unwinds at the boundary") {
    std::vector<std::string> log;
    Recorder rec{log};

    // Cancel at the "transcribing" boundary: decode ran, nothing after.
    auto throw_at_transcribing = [](const std::string& s) {
        if (s == "transcribing") throw std::runtime_error("cancelled");
    };

    CHECK_THROWS_AS(run_job(rec.steps(true), throw_at_transcribing, nullptr),
                    std::runtime_error);
    // decode happened (before the transcribing boundary); transcribe did not.
    CHECK(log == std::vector<std::string>{"decode"});
}

TEST_CASE("run_job: on_duration is optional (null) and load_align is optional") {
    std::vector<std::string> log;
    Recorder rec{log};
    Steps s = rec.steps(/*with_diarize=*/false);
    s.load_align = nullptr;  // no resolver -> stage still fires, step skipped

    std::vector<std::string> stages;
    json out = run_job(
        s, [&](const std::string& st) { stages.emplace_back(st); }, nullptr);

    // loading_align stage still announced even though the resolver is absent.
    CHECK(stages == std::vector<std::string>{"decoding", "transcribing",
                                             "loading_align", "aligning"});
    CHECK(log == std::vector<std::string>{"decode", "transcribe", "align"});
    CHECK(out["diarized"] == false);
}
