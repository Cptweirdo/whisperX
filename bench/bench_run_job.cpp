// Phase-6 bench: end-to-end native run_job RTF + RSS-over-N-jobs. Drives the
// *real* orchestrate::run_job (the same sequencer the pybind `run_job` uses) over
// the native engines — decode-once AudioBuffer through silero VAD -> merge_chunks
// -> WhisperSherpa -> align_run -> (optional) SherpaDiarizer + assign_word_speakers
// — so the numbers are the production path, not a reimplementation. Pure C++ (no
// pybind/Python): `resolve_align` is a local lambda that reads the mirror's
// meta.json off disk. Models are built ONCE and reused across every run (the
// resident-models proof); each Steps closure self-times, so we report per-stage
// median wall ms + RTF (stage_ms / audio_ms). Dependency-light (std::chrono, no
// Google Benchmark) so it rides the WHISPERX_CORE_AUDIO build like bench_audio.
//
//   bench_run_job --clip a.wav --silero silero_vad.onnx \
//       --whisper-dir <sherpa-whisper-dir> --align-dir <model.onnx+meta.json dir> \
//       [--diarize-seg seg.onnx --diarize-embed emb.onnx] [--lang en] \
//       [--mode timing|rss] [--runs N] [--warmup W] [--num-clusters K] \
//       [--feature-dim 80] [--threads 4] [--json out.json]
//
//   timing: median per-stage ms + RTF over `runs` (first `warmup` discarded).
//   rss   : RSS (KB, /proc/self/statm) sampled after each of `runs` jobs — the
//           tail-flat check (bench/check_budget.py --rss) decides the std::pmr arena.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>  // sysconf(_SC_PAGESIZE)

#include <nlohmann/json.hpp>

#include "align/align_driver.hpp"
#include "align/wav2vec2_onnx.hpp"
#include "asr/whisper_sherpa.hpp"
#include "audio/audio_buffer.hpp"
#include "audio/decode.hpp"
#include "audio/vad_silero.hpp"
#include "diarize/assign_speakers.hpp"
#include "diarize/diarize_sherpa.hpp"
#include "diarize/interval_tree.hpp"
#include "orchestrate/orchestrate.hpp"
#include "vad/merge_chunks.hpp"

namespace audio = whisperx::audio;
namespace asr = whisperx::asr;
namespace al = whisperx::align;
namespace wd = whisperx::diarize;
namespace orch = whisperx::orchestrate;
using nlohmann::json;
using clk = std::chrono::steady_clock;

// VAD params — the staged/parity baseline (asr_sherpa defaults).
constexpr double kVadOnset = 0.5, kVadOffset = 0.363, kChunkSize = 30.0;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// --- transcript-assembly helpers: identical to the pybind run_job's, so the bench
// runs the same work per stage (text strip + round3 segment bounds + speaker label).
static std::string strip_ws(const std::string& s) {
    const char* ws = " \t\n\r\f\v";
    const auto b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}
static double round3(double x) { return std::nearbyint(x * 1000.0) / 1000.0; }
static std::string speaker_label(int id) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "SPEAKER_%02d", id);
    return std::string(buf);
}

// First file in `dir` whose name contains `want`, preferring names *without*
// `avoid` (e.g. the non-int8 ONNX, matching the CI asset picker).
static std::string find_in_dir(const std::string& dir, const std::string& want,
                               const std::string& avoid = "") {
    std::vector<std::string> hits;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        const std::string name = e.path().filename().string();
        if (name.find(want) != std::string::npos) hits.push_back(e.path().string());
    }
    if (hits.empty())
        throw std::runtime_error("no file matching '" + want + "' in " + dir);
    std::stable_sort(hits.begin(), hits.end(), [&](const auto& a, const auto& b) {
        if (avoid.empty()) return false;
        return (a.find(avoid) != std::string::npos) <
               (b.find(avoid) != std::string::npos);  // avoid-free first
    });
    return hits.front();
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

// Resident set size in KB (Linux /proc/self/statm: field 2 = resident pages).
static long rss_kb() {
    std::ifstream f("/proc/self/statm");
    long size_pages = 0, resident_pages = 0;
    f >> size_pages >> resident_pages;
    return resident_pages * (sysconf(_SC_PAGESIZE) / 1024);
}

// --- arg parsing -------------------------------------------------------------
struct Args {
    std::string clip, silero, whisper_dir, align_dir, diar_seg, diar_embed, lang;
    std::string mode = "timing", json_out;
    int runs = 7, warmup = 2, num_clusters = 0, feature_dim = 80, threads = 4;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    auto next = [&](int& i) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--clip") a.clip = next(i);
        else if (k == "--silero") a.silero = next(i);
        else if (k == "--whisper-dir") a.whisper_dir = next(i);
        else if (k == "--align-dir") a.align_dir = next(i);
        else if (k == "--diarize-seg") a.diar_seg = next(i);
        else if (k == "--diarize-embed") a.diar_embed = next(i);
        else if (k == "--lang") a.lang = next(i);
        else if (k == "--mode") a.mode = next(i);
        else if (k == "--json") a.json_out = next(i);
        else if (k == "--runs") a.runs = std::stoi(next(i));
        else if (k == "--warmup") a.warmup = std::stoi(next(i));
        else if (k == "--num-clusters") a.num_clusters = std::stoi(next(i));
        else if (k == "--feature-dim") a.feature_dim = std::stoi(next(i));
        else if (k == "--threads") a.threads = std::stoi(next(i));
        else throw std::runtime_error("unknown arg: " + k);
    }
    if (a.clip.empty() || a.silero.empty() || a.whisper_dir.empty() || a.align_dir.empty())
        throw std::runtime_error("--clip, --silero, --whisper-dir, --align-dir are required");
    return a;
}

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_run_job: %s\n", e.what());
        return 2;
    }

    // --- build the engines ONCE (resident across every run) ------------------
    const std::string enc = find_in_dir(args.whisper_dir, "encoder", "int8");
    const std::string dec = find_in_dir(args.whisper_dir, "decoder", "int8");
    const std::string tok = find_in_dir(args.whisper_dir, "tokens");
    asr::WhisperSherpa asr_model(enc, dec, tok, args.threads, args.feature_dim);

    const std::string align_onnx = args.align_dir + "/model.onnx";
    json meta = json::parse(std::ifstream(args.align_dir + "/meta.json"));
    al::Wav2Vec2Onnx align_model(align_onnx, args.threads);
    const auto dictionary = meta.at("dictionary").get<std::map<std::string, int>>();
    const bool batchable = meta.value("batchable", false);
    std::string lang = args.lang.empty() ? meta.value("language", std::string("en")) : args.lang;

    std::unique_ptr<wd::SherpaDiarizer> diar;
    if (!args.diar_seg.empty())
        diar = std::make_unique<wd::SherpaDiarizer>(args.diar_seg, args.diar_embed,
                                                    args.threads, "cpu");

    // --- per-run, self-timing Steps over the resident engines ----------------
    struct Timings {
        double decode = 0, transcribe = 0, load_align = 0, align = 0, diarize = 0;
    } t;

    orch::Steps steps;
    steps.decode = [&] {
        auto s = clk::now();
        auto buf = audio::load_audio(args.clip);
        t.decode = ms_since(s);
        return buf;
    };
    steps.transcribe = [&](const audio::AudioBuffer& buf) -> std::pair<json, std::string> {
        auto s = clk::now();
        auto segs = audio::silero_segments(buf, args.silero, kVadOnset, kChunkSize);
        json merged = whisperx::vad::merge_chunks(segs, kChunkSize, kVadOnset, kVadOffset);
        std::vector<std::pair<double, double>> spans;
        spans.reserve(merged.size());
        for (const auto& mc : merged)
            spans.emplace_back(mc.at("start").get<double>(), mc.at("end").get<double>());
        std::string detected = args.lang;
        if (detected.empty() && !spans.empty()) detected = asr_model.detect_language(buf);
        auto chunks = asr_model.transcribe(buf, spans, detected, "transcribe");
        json out = json::array();
        for (std::size_t i = 0; i < spans.size(); ++i) {
            json seg;
            seg["text"] = strip_ws(chunks[i].text);
            seg["start"] = round3(spans[i].first);
            seg["end"] = round3(spans[i].second);
            seg["avg_logprob"] = static_cast<double>(chunks[i].avg_logprob);
            out.push_back(std::move(seg));
        }
        t.transcribe = ms_since(s);
        return {std::move(out), detected.empty() ? std::string("en") : detected};
    };
    // Models are resident — "loading" is a no-op here (the whole point of the
    // resident-models design); we still time it to report ~0 honestly.
    steps.load_align = [&](const std::string&) {
        auto s = clk::now();
        t.load_align = ms_since(s);
    };
    steps.align = [&](const audio::AudioBuffer& buf, const json& transcript,
                      const std::string& language) {
        auto s = clk::now();
        std::span<const float> aspan(buf.samples.data(), buf.samples.size());
        json r = al::align_run(transcript, align_model, dictionary, aspan, language,
                               batchable, "nearest", false, nullptr);
        t.align = ms_since(s);
        return r;
    };
    if (diar)
        steps.diarize = [&](const audio::AudioBuffer& buf, json aligned) {
            auto s = clk::now();
            auto raw = diar->diarize(buf, args.num_clusters);
            std::vector<wd::Turn> turns;
            turns.reserve(raw.size());
            for (const auto& d : raw) turns.push_back({d.start, d.end, speaker_label(d.speaker)});
            aligned["segments"] = wd::assign_word_speakers(turns, aligned["segments"], false);
            t.diarize = ms_since(s);
            return aligned;
        };

    double duration_s = 0;
    std::function<void(double)> on_dur = [&](double d) { duration_s = d; };
    std::function<void(const std::string&)> no_stage;  // empty: closures self-time

    // --- forced language: the bench resolves it up front (CI may pin --lang) --
    if (!args.lang.empty()) lang = args.lang;

    if (args.mode == "rss") {
        std::vector<long> rss;
        rss.reserve(args.runs);
        for (int i = 0; i < args.runs; ++i) {
            t = Timings{};
            orch::run_job(steps, no_stage, on_dur);
            rss.push_back(rss_kb());
        }
        json result;
        result["clip"] = args.clip;
        result["runs"] = args.runs;
        result["rss_kb"] = rss;
        std::printf("rss over %d jobs (KB): first=%ld last=%ld\n", args.runs,
                    rss.front(), rss.back());
        if (!args.json_out.empty()) std::ofstream(args.json_out) << result.dump(2) << "\n";
        return 0;
    }

    // timing mode
    std::vector<Timings> samples;
    samples.reserve(args.runs);
    for (int i = 0; i < args.warmup + args.runs; ++i) {
        t = Timings{};
        orch::run_job(steps, no_stage, on_dur);
        if (i >= args.warmup) samples.push_back(t);
    }
    auto med = [&](double Timings::*field) {
        std::vector<double> v;
        v.reserve(samples.size());
        for (const auto& s : samples) v.push_back(s.*field);
        return median(v);
    };

    json result;
    result["clip"] = args.clip;
    result["duration_s"] = duration_s;
    result["runs"] = args.runs;
    result["warmup"] = args.warmup;
    result["diarized"] = static_cast<bool>(diar);
    auto add = [&](const char* name, double stage_ms) {
        json j;
        j["ms"] = stage_ms;
        j["rtf"] = duration_s > 0 ? stage_ms / (duration_s * 1000.0) : 0.0;
        result["stages"][name] = j;
    };
    add("decoding", med(&Timings::decode));
    add("transcribing", med(&Timings::transcribe));
    add("loading_align", med(&Timings::load_align));
    add("aligning", med(&Timings::align));
    if (diar) add("diarizing", med(&Timings::diarize));

    std::printf("clip       : %s  (%.2f s, %d runs / %d warmup)\n", args.clip.c_str(),
                duration_s, args.runs, args.warmup);
    for (const auto& name : {"decoding", "transcribing", "loading_align", "aligning", "diarizing"}) {
        if (!result["stages"].contains(name)) continue;
        std::printf("  %-14s %8.2f ms   RTF %.4f\n", name,
                    result["stages"][name]["ms"].get<double>(),
                    result["stages"][name]["rtf"].get<double>());
    }
    if (!args.json_out.empty()) std::ofstream(args.json_out) << result.dump(2) << "\n";
    return 0;
}
