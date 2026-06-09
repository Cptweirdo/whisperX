#include "jobs/runner.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "align/align_driver.hpp"
#include "audio/audio_buffer.hpp"
#include "audio/decode.hpp"
#include "audio/vad_silero.hpp"
#include "db/session_store.hpp"
#include "diarize/assign_speakers.hpp"
#include "log/log.hpp"
#include "models/model_manager.hpp"
#include "orchestrate/orchestrate.hpp"
#include "sse/broker.hpp"
#include "vad/merge_chunks.hpp"
#include "writers/writers.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

namespace wa = whisperx::audio;
namespace wal = whisperx::align;
namespace wd = whisperx::diarize;
namespace orch = whisperx::orchestrate;
namespace mm = whisperx::server::models;

namespace whisperx::server::jobs {

namespace {

// VAD hyperparameters — the asr_sherpa.py defaults (and the orchestrate path's).
constexpr double kVadOnset = 0.5;
constexpr double kVadOffset = 0.363;
constexpr double kChunkSize = 30.0;
const char* const kArtifactBasename = "transcript";
const std::vector<std::string> kOutputFormats = {"srt", "vtt", "txt", "json"};

// STAGE_RTF (pipeline.py:192) — measured CPU medians, rounded up for an ETA hint.
double stage_rtf(const std::string& stage) {
    if (stage == "transcribing") return 0.10;
    if (stage == "aligning") return 0.15;
    if (stage == "diarizing") return 0.15;
    return 0.0;
}

std::string strip_ws(const std::string& s) {
    const char* ws = " \t\n\r\f\v";
    auto b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    auto e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

double round3(double x) { return std::nearbyint(x * 1000.0) / 1000.0; }

std::string speaker_label(int cluster_id) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "SPEAKER_%02d", cluster_id);
    return std::string(buf);
}

std::optional<std::string> opt_str(const json& obj, const char* key) {
    if (obj.is_object() && obj.contains(key) && obj[key].is_string())
        return obj[key].get<std::string>();
    return std::nullopt;
}

std::optional<int> opt_int(const json& obj, const char* key) {
    if (obj.is_object() && obj.contains(key) && obj[key].is_number_integer())
        return obj[key].get<int>();
    return std::nullopt;
}

}  // namespace

RunSession make_run_session(whisperx::db::SessionStore& store,
                            mm::ModelManager& manager,
                            whisperx::server::sse::Broker& broker,
                            const whisperx::server::Config& cfg) {
    std::string data_dir = cfg.data_dir;
    return [&store, &manager, &broker, data_dir](const std::string& session_id,
                                                 const CancelFlag& cancel) {
        auto logger = whisperx::server::log::get("runner");
        const auto row = store.get(session_id);
        if (!row)
            throw std::runtime_error("session " + session_id + " disappeared");

        json opts = row->options.is_object() ? row->options : json::object();
        std::string model = row->model.value_or(manager.active());
        std::string audio_filename = row->audio_filename.value_or("");
        fs::path session_dir =
            fs::path(data_dir) / "sessions" / session_id;
        std::string audio_path = (session_dir / audio_filename).string();

        std::string forced_language = opt_str(opts, "language").value_or("");
        std::optional<int> min_speakers = opt_int(opts, "min_speakers");
        std::optional<int> max_speakers = opt_int(opts, "max_speakers");

        // --- resident engines (load on demand; shared diarizer/align) --------
        // shared_ptrs held for the whole job: a runtime device switch may evict
        // the manager's caches mid-flight; these refs keep our engines alive.
        std::shared_ptr<whisperx::asr::WhisperSherpa> asr = manager.load_asr(model);
        std::shared_ptr<wd::SherpaDiarizer> diarizer = manager.ensure_diarize();
        int num_clusters = 0;
        if (max_speakers)
            num_clusters = *max_speakers;
        else if (min_speakers)
            num_clusters = *min_speakers;

        const std::string silero = manager.silero_path();

        // --- progress (durable mark_stage + SSE delta), stage-boundary cancel -
        auto stage = [&](const std::string& s) {
            if (cancel->load()) throw Cancelled();
            store.mark_stage(session_id, whisperx::db::parse_stage(s));
            const auto row2 = store.get(session_id);
            double dur = row2 ? row2->duration.value_or(0.0) : 0.0;
            json ev = {{"stage", s}};
            double rtf = stage_rtf(s);
            if (rtf > 0.0 && dur > 0.0)
                ev["eta"] = std::llround(rtf * dur);
            broker.publish(session_id, ev);
        };

        double captured_duration = 0.0;
        auto on_duration = [&](double d) {
            captured_duration = d;
            store.mark_duration(session_id, d);
        };

        // --- Steps over the native engines (port of the pybind run_job) ------
        mm::AlignHandle align_handle{nullptr, nullptr, false};
        int blank_audio_removed = 0;

        orch::Steps steps;
        steps.decode = [&] { return wa::load_audio(audio_path); };

        steps.transcribe =
            [&](const wa::AudioBuffer& buf) -> std::pair<json, std::string> {
            auto segs = wa::silero_segments(buf, silero, kVadOnset, kChunkSize);
            json merged = whisperx::vad::merge_chunks(segs, kChunkSize, kVadOnset,
                                                      kVadOffset);
            std::vector<std::pair<double, double>> spans;
            spans.reserve(merged.size());
            for (const auto& mc : merged)
                spans.emplace_back(mc.at("start").get<double>(),
                                   mc.at("end").get<double>());

            std::string lang = forced_language;
            if (lang.empty() && !spans.empty()) lang = asr->detect_language(buf);

            auto chunks = asr->transcribe(buf, spans, lang, "transcribe");
            json out = json::array();
            for (std::size_t i = 0; i < spans.size(); ++i) {
                blank_audio_removed += chunks[i].blank_audio_removed;
                json seg;
                seg["text"] = strip_ws(chunks[i].text);
                seg["start"] = round3(spans[i].first);
                seg["end"] = round3(spans[i].second);
                seg["avg_logprob"] = static_cast<double>(chunks[i].avg_logprob);
                out.push_back(std::move(seg));
            }
            return {std::move(out), lang.empty() ? std::string("en") : lang};
        };

        steps.load_align = [&](const std::string& lang) {
            align_handle = manager.align_for(lang);
        };

        steps.align = [&](const wa::AudioBuffer& buf, const json& transcript,
                          const std::string& lang) {
            std::span<const float> aspan(buf.samples.data(), buf.samples.size());
            return wal::align_run(transcript, *align_handle.model,
                                  *align_handle.dictionary, aspan, lang,
                                  align_handle.batchable, "nearest", false,
                                  nullptr);
        };

        if (diarizer != nullptr)
            steps.diarize = [&, diarizer](const wa::AudioBuffer& buf, json aligned) {
                auto raw = diarizer->diarize(buf, num_clusters);
                std::vector<wd::Turn> turns;
                turns.reserve(raw.size());
                for (const auto& d : raw)
                    turns.push_back({d.start, d.end, speaker_label(d.speaker)});
                aligned["segments"] = wd::assign_word_speakers(
                    turns, aligned["segments"], /*fill_nearest=*/false);
                return aligned;
            };

        json result = orch::run_job(steps, stage, on_duration);
        if (blank_audio_removed)
            logger->warn(
                "Stripped {} [BLANK_AUDIO] marker(s) the sherpa Whisper backend "
                "emitted for silent audio.",
                blank_audio_removed);

        const std::string language =
            result.value("language", forced_language.empty() ? "en"
                                                             : forced_language);
        const bool diarized = result.value("diarized", false);
        long long num_segments =
            static_cast<long long>(result.value("segments", json::array()).size());
        result["duration"] = captured_duration;
        result["num_segments"] = num_segments;

        // --- write export artifacts via the native writers ------------------
        // The json writer emits transcript.json, which is the result sidecar
        // SessionStore::load_result reads back.
        const json wopts = {{"max_line_width", nullptr},
                            {"max_line_count", nullptr},
                            {"highlight_words", false}};
        fs::create_directories(session_dir);
        for (const auto& fmt : kOutputFormats) {
            std::string body;
            if (fmt == "srt")
                body = whisperx::writers::write_srt(result, wopts);
            else if (fmt == "vtt")
                body = whisperx::writers::write_vtt(result, wopts);
            else if (fmt == "txt")
                body = whisperx::writers::write_txt(result, wopts);
            else
                body = whisperx::writers::write_json(result, wopts);
            std::ofstream(session_dir / (std::string(kArtifactBasename) + "." + fmt))
                << body;
        }

        store.mark_done(session_id, std::optional<std::string>(language),
                        diarized, model, num_segments, captured_duration);
        logger->info("Job complete: {} segments (lang={})", num_segments,
                     language);
    };
}

}  // namespace whisperx::server::jobs
