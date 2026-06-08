// pybind11 adapter — the Python-facing face of the C++ engine core and the
// parity oracle for the strangler-fig migration. As each stage lands in `core/`
// it is exposed here and diffed against the live Python pipeline from pytest.
//
// Phase 0: the pure text metric + build provenance.
// Phase 1: the SQLite `SessionStore` (DB-only surface) — `app.store.SessionStore`
//          forwards its DB methods here when `WHISPERX_CORE_STAGES` contains `db`.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "align/align.hpp"
#include "align/char_clean.hpp"
#include "align/emission_post.hpp"
#include "align/trellis.hpp"
#include "audio/audio_constants.hpp"
#include "build_info.hpp"
#include "db/session_store.hpp"
#include "diarize/assign_speakers.hpp"
#include "edits/edits.hpp"
#include "orchestrate/orchestrate.hpp"
#include "text/edit_distance.hpp"
#include "text/sentence_split.hpp"
#include "text/sequence_matcher.hpp"
#include "vad/merge_chunks.hpp"
#include "writers/writers.hpp"

#ifdef WHISPERX_CORE_AUDIO
#include "align/align_driver.hpp"
#include "align/wav2vec2_onnx.hpp"
#include "asr/whisper_sherpa.hpp"
#include "audio/decode.hpp"
#include "audio/vad_silero.hpp"
#include "diarize/diarize_sherpa.hpp"
#endif

namespace py = pybind11;
using nlohmann::json;
using whisperx::db::SessionStore;

namespace {

// nlohmann::json -> native Python object (dict/list/str/int/float/bool/None), so
// row dicts cross the seam looking exactly like app.store._row_to_dict output.
py::object json_to_py(const json& j) {
    switch (j.type()) {
        case json::value_t::null:
        case json::value_t::discarded:
            return py::none();
        case json::value_t::boolean:
            return py::bool_(j.get<bool>());
        case json::value_t::number_integer:
            return py::int_(j.get<int64_t>());
        case json::value_t::number_unsigned:
            return py::int_(j.get<uint64_t>());
        case json::value_t::number_float:
            return py::float_(j.get<double>());
        case json::value_t::string:
            return py::str(j.get<std::string>());
        case json::value_t::array: {
            py::list l;
            for (const auto& e : j) l.append(json_to_py(e));
            return std::move(l);
        }
        case json::value_t::object:
        case json::value_t::binary: {
            py::dict d;
            for (auto it = j.begin(); it != j.end(); ++it) {
                d[py::str(it.key())] = json_to_py(it.value());
            }
            return std::move(d);
        }
    }
    return py::none();
}

// Python object -> nlohmann::json (for the `options` dict + translation fields).
// bool is checked before int (Python bool subclasses int).
json py_to_json(const py::handle& o) {
    if (o.is_none()) return json(nullptr);
    if (py::isinstance<py::bool_>(o)) return o.cast<bool>();
    if (py::isinstance<py::int_>(o)) return o.cast<int64_t>();
    if (py::isinstance<py::float_>(o)) return o.cast<double>();
    if (py::isinstance<py::str>(o)) return o.cast<std::string>();
    if (py::isinstance<py::list>(o) || py::isinstance<py::tuple>(o)) {
        json a = json::array();
        for (const auto& e : o) a.push_back(py_to_json(e));
        return a;
    }
    if (py::isinstance<py::dict>(o)) {
        json d = json::object();
        for (const auto& item : o.cast<py::dict>()) {
            d[py::str(item.first).cast<std::string>()] = py_to_json(item.second);
        }
        return d;
    }
    return py::str(o).cast<std::string>();  // last resort: stringify
}

void bind_session_store(py::module_& m) {
    py::class_<SessionStore>(m, "SessionStore")
        .def(py::init<const std::string&>(), py::arg("data_dir"))
        .def_property_readonly("db_path", &SessionStore::db_path)
        // backup / restore
        .def("snapshot_db", &SessionStore::snapshot_db, py::arg("dest_path"))
        .def("swap_db", &SessionStore::swap_db, py::arg("new_db_path"))
        // writes
        .def(
            "create",
            [](SessionStore& s, const std::string& session_id,
               const std::string& filename, const std::string& audio_filename,
               const py::handle& options, std::optional<std::string> model) {
                s.create(session_id, filename, audio_filename,
                         py_to_json(options), model);
            },
            py::arg("session_id"), py::arg("filename"), py::arg("audio_filename"),
            py::arg("options"), py::arg("model") = py::none())
        .def("mark_running", &SessionStore::mark_running, py::arg("session_id"))
        .def("mark_stage", &SessionStore::mark_stage, py::arg("session_id"),
             py::arg("stage"))
        .def("mark_duration", &SessionStore::mark_duration, py::arg("session_id"),
             py::arg("duration"))
        .def("mark_done", &SessionStore::mark_done, py::arg("session_id"),
             py::arg("language"), py::arg("diarized"), py::arg("model"),
             py::arg("num_segments"), py::arg("duration"))
        .def("mark_error", &SessionStore::mark_error, py::arg("session_id"),
             py::arg("message"))
        .def("rename", &SessionStore::rename, py::arg("session_id"),
             py::arg("name"))
        .def("delete", &SessionStore::remove, py::arg("session_id"))
        // speaker name overrides
        .def(
            "get_speaker_names",
            [](SessionStore& s, const std::string& session_id) {
                return json_to_py(s.get_speaker_names(session_id));
            },
            py::arg("session_id"))
        .def("set_speaker_name", &SessionStore::set_speaker_name,
             py::arg("session_id"), py::arg("speaker_key"), py::arg("name"))
        // settings
        .def("get_setting", &SessionStore::get_setting, py::arg("key"),
             py::arg("default") = py::none())
        .def("set_setting", &SessionStore::set_setting, py::arg("key"),
             py::arg("value"))
        // translations (JSON column)
        .def(
            "get_translations",
            [](SessionStore& s, const std::string& session_id) {
                return json_to_py(s.get_translations(session_id));
            },
            py::arg("session_id"))
        .def(
            "set_translation_status",
            [](SessionStore& s, const std::string& session_id,
               const std::string& lang, const std::string& status,
               std::optional<std::string> service,
               std::optional<std::string> error) {
                return json_to_py(s.set_translation_status(session_id, lang,
                                                           status, service, error));
            },
            py::arg("session_id"), py::arg("lang"), py::arg("status"),
            py::arg("service") = py::none(), py::arg("error") = py::none())
        // reads
        .def(
            "get",
            [](SessionStore& s, const std::string& session_id) {
                return json_to_py(s.get(session_id));
            },
            py::arg("session_id"))
        .def("list", [](SessionStore& s) { return json_to_py(s.list()); })
        .def("has_active_jobs", &SessionStore::has_active_jobs)
        // lifecycle
        .def("reconcile_startup", &SessionStore::reconcile_startup)
        .def("close", &SessionStore::close)
        // file-backed sidecars (edits/undo overlay + translation files)
        .def(
            "load_result",
            [](SessionStore& s, const std::string& session_id) {
                return json_to_py(s.load_result(session_id));
            },
            py::arg("session_id"))
        .def(
            "load_edits",
            [](SessionStore& s, const std::string& session_id) {
                return json_to_py(s.load_edits(session_id));
            },
            py::arg("session_id"))
        .def(
            "current_segments",
            [](SessionStore& s, const std::string& session_id,
               const py::handle& original_segments) {
                return json_to_py(
                    s.current_segments(session_id, py_to_json(original_segments)));
            },
            py::arg("session_id"), py::arg("original_segments"))
        .def("edit_history_len", &SessionStore::edit_history_len,
             py::arg("session_id"))
        .def(
            "save_turn_edit",
            [](SessionStore& s, const std::string& session_id, long turn_index,
               const std::string& new_text) {
                return json_to_py(
                    s.save_turn_edit(session_id, turn_index, new_text));
            },
            py::arg("session_id"), py::arg("turn_index"), py::arg("new_text"))
        .def(
            "save_turn_reassign",
            [](SessionStore& s, const std::string& session_id, long turn_index,
               const std::string& new_speaker) {
                return json_to_py(
                    s.save_turn_reassign(session_id, turn_index, new_speaker));
            },
            py::arg("session_id"), py::arg("turn_index"), py::arg("new_speaker"))
        .def(
            "undo_turn_edit",
            [](SessionStore& s, const std::string& session_id) {
                return json_to_py(s.undo_turn_edit(session_id));
            },
            py::arg("session_id"))
        .def(
            "load_translation",
            [](SessionStore& s, const std::string& session_id,
               const std::string& lang) {
                return json_to_py(s.load_translation(session_id, lang));
            },
            py::arg("session_id"), py::arg("lang"))
        .def(
            "save_translation",
            [](SessionStore& s, const std::string& session_id,
               const std::string& lang, const py::handle& payload) {
                s.save_translation(session_id, lang, py_to_json(payload));
            },
            py::arg("session_id"), py::arg("lang"), py::arg("payload"));
}

// The app/edits.py algorithms (turn grouping / collapse / realign / undo) and the
// difflib matching-block primitive they rely on, exposed for the app.edits facade
// and the parity oracle.
void bind_edits(py::module_& m) {
    namespace ed = whisperx::edits;

    m.attr("HISTORY_LIMIT") = ed::HISTORY_LIMIT;
    m.attr("MIN_WORD_WIDTH") = ed::MIN_WORD_WIDTH;
    m.attr("SEGMENT_MIN_DURATION") = ed::SEGMENT_MIN_DURATION;

    // app.edits.NoChange — apply_turn_reassign raises this on a no-op.
    py::register_exception<ed::NoChange>(m, "NoChange");

    m.def(
        "group_turns",
        [](const py::handle& segments) {
            return json_to_py(ed::group_turns(py_to_json(segments)));
        },
        py::arg("segments"));
    m.def(
        "distinct_speakers",
        [](const py::handle& segments) {
            return json_to_py(ed::distinct_speakers(py_to_json(segments)));
        },
        py::arg("segments"));
    m.def(
        "next_speaker_key",
        [](const py::handle& existing_keys) {
            return ed::next_speaker_key(py_to_json(existing_keys));
        },
        py::arg("existing_keys"));
    m.def(
        "coalesce_segments",
        [](const py::handle& segments, double threshold) {
            return json_to_py(
                ed::coalesce_segments(py_to_json(segments), threshold));
        },
        py::arg("segments"), py::arg("threshold") = ed::SEGMENT_MIN_DURATION);
    m.def(
        "realign_words",
        [](const py::handle& old_words, const std::string& new_text,
           const py::handle& start, const py::handle& end) {
            return json_to_py(ed::realign_words(py_to_json(old_words), new_text,
                                                py_to_json(start),
                                                py_to_json(end)));
        },
        py::arg("old_words"), py::arg("new_text"), py::arg("start") = py::none(),
        py::arg("end") = py::none());
    m.def(
        "apply_turn_edit",
        [](const py::handle& segments, long turn_index,
           const std::string& new_text) {
            auto [segs, delta] =
                ed::apply_turn_edit(py_to_json(segments), turn_index, new_text);
            return py::make_tuple(json_to_py(segs), json_to_py(delta));
        },
        py::arg("segments"), py::arg("turn_index"), py::arg("new_text"));
    m.def(
        "apply_turn_reassign",
        [](const py::handle& segments, long turn_index,
           const std::string& new_speaker) {
            auto [segs, delta] = ed::apply_turn_reassign(py_to_json(segments),
                                                         turn_index, new_speaker);
            return py::make_tuple(json_to_py(segs), json_to_py(delta));
        },
        py::arg("segments"), py::arg("turn_index"), py::arg("new_speaker"));
    m.def(
        "undo_last",
        [](const py::handle& segments, const py::handle& history) {
            auto [segs, hist] =
                ed::undo_last(py_to_json(segments), py_to_json(history));
            return py::make_tuple(json_to_py(segs), json_to_py(hist));
        },
        py::arg("segments"), py::arg("history"));

    // The difflib primitive realign_words is built on, exposed so the parity
    // oracle can diff it against difflib.SequenceMatcher(...).get_matching_blocks().
    m.def(
        "matching_blocks",
        [](const std::vector<std::string>& a, const std::vector<std::string>& b) {
            whisperx::text::SequenceMatcher sm(a, b);
            py::list out;
            for (const auto& blk : sm.get_matching_blocks()) {
                out.append(py::make_tuple(blk.a, blk.b, blk.size));
            }
            return out;
        },
        py::arg("a"), py::arg("b"),
        "difflib.SequenceMatcher(autojunk=False).get_matching_blocks() as "
        "(a, b, size) tuples.");
}

// whisperx/vads/vad.py::Vad.merge_chunks — packs VAD speech segments into
// ≤chunk_size windows. Exposed for the whisperx.vads.vad facade (`vad` token)
// and the parity oracle; the audio hyperparameters ride along as module attrs.
void bind_vad(py::module_& m) {
    namespace audio = whisperx::audio;
    m.attr("SAMPLE_RATE") = audio::kSampleRate;
    m.attr("N_FFT") = audio::kNFft;
    m.attr("HOP_LENGTH") = audio::kHopLength;
    m.attr("CHUNK_LENGTH") = audio::kChunkLength;
    m.attr("N_SAMPLES") = audio::kNSamples;
    m.attr("N_FRAMES") = audio::kNFrames;
    m.attr("N_SAMPLES_PER_TOKEN") = audio::kNSamplesPerToken;
    m.attr("FRAMES_PER_SECOND") = audio::kFramesPerSecond;
    m.attr("TOKENS_PER_SECOND") = audio::kTokensPerSecond;

    using SegTuple = std::tuple<double, double, std::optional<std::string>>;
    m.def(
        "merge_chunks",
        [](const std::vector<SegTuple>& segments, double chunk_size, double onset,
           std::optional<double> offset) {
            std::vector<whisperx::vad::VadSegment> v;
            v.reserve(segments.size());
            for (const auto& [s, e, sp] : segments) v.push_back({s, e, sp});
            return json_to_py(
                whisperx::vad::merge_chunks(v, chunk_size, onset, offset));
        },
        py::arg("segments"), py::arg("chunk_size"), py::arg("onset") = 0.5,
        py::arg("offset") = py::none(),
        "Vad.merge_chunks: segments as (start, end, speaker|None) tuples -> list "
        "of {start, end, segments:[[s,e],…]} chunks.");
}

// whisperx/alignment.py forced-alignment core (`align` token, Phase 3A): the
// Viterbi trellis + the char→word→sentence assembly + the native sentence
// splitter. The model forward + char-cleaning + wildcard extension stay Python
// (the facade); these take the fixed extended emission + precomputed tokens.
namespace {

// Wrap a (T, V) float32 numpy array as an Emission view (no copy; the array must
// outlive the call, which it does within these lambdas).
whisperx::align::Emission emission_view(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& a) {
    if (a.ndim() != 2)
        throw std::invalid_argument("emission must be a 2-D (T, V) array");
    return {a.data(), static_cast<std::size_t>(a.shape(0)),
            static_cast<std::size_t>(a.shape(1))};
}

}  // namespace

void bind_align(py::module_& m) {
    namespace al = whisperx::align;

    // Viterbi backtrack path — get_trellis + backtrack. Returns list of
    // (token_index, time_index, score) or None on failure. (Viterbi-exact golden.)
    m.def(
        "align_trellis_path",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>&
               emission,
           const std::vector<int>& tokens, int blank_id) -> py::object {
            auto emi = emission_view(emission);
            auto trellis = al::get_trellis(emi, tokens, blank_id);
            auto path = al::backtrack(trellis, emi, tokens, blank_id);
            if (!path) return py::none();
            py::list out;
            for (const auto& p : *path)
                out.append(py::make_tuple(p.token_index, p.time_index, p.score));
            return std::move(out);
        },
        py::arg("emission"), py::arg("tokens"), py::arg("blank_id") = 0);

    // merge_repeats char-segments — get_trellis + backtrack + merge_repeats.
    // Returns list of (label, start, end, score) or None.
    m.def(
        "align_char_segments",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>&
               emission,
           const std::vector<int>& tokens, int blank_id,
           const std::string& transcript) -> py::object {
            auto emi = emission_view(emission);
            auto trellis = al::get_trellis(emi, tokens, blank_id);
            auto path = al::backtrack(trellis, emi, tokens, blank_id);
            if (!path) return py::none();
            auto segs = al::merge_repeats(*path, transcript);
            py::list out;
            for (const auto& s : segs)
                out.append(py::make_tuple(s.label, s.start, s.end, s.score));
            return std::move(out);
        },
        py::arg("emission"), py::arg("tokens"), py::arg("blank_id"),
        py::arg("transcript"));

    // Native sentence splitter — replaces nltk punkt. (start, end) codepoint spans.
    m.def(
        "sentence_spans",
        [](const std::string& text, const std::string& lang) {
            py::list out;
            for (const auto& [s, e] : whisperx::text::sentence_spans(text, lang))
                out.append(py::make_tuple(s, e));
            return out;
        },
        py::arg("text"), py::arg("lang") = "en");

    // The per-segment assembly (the `align` facade entry). Returns
    // {"ok": bool, "subsegments": [...]}.
    m.def(
        "align_assemble",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>&
               emission,
           const std::vector<int>& tokens, int blank_id,
           const std::string& text_clean, const std::string& text,
           const std::vector<int>& clean_cdx, double t1, double t2,
           const std::string& language, const std::string& interpolate_method,
           bool return_char_alignments, std::optional<double> avg_logprob) {
            auto emi = emission_view(emission);
            auto res = al::align_assemble(emi, tokens, blank_id, text_clean, text,
                                          clean_cdx, t1, t2, language,
                                          interpolate_method,
                                          return_char_alignments, avg_logprob);
            py::dict d;
            d["ok"] = res.ok;
            d["subsegments"] = json_to_py(res.subsegments);
            return d;
        },
        py::arg("emission"), py::arg("tokens"), py::arg("blank_id"),
        py::arg("text_clean"), py::arg("text"), py::arg("clean_cdx"),
        py::arg("t1"), py::arg("t2"), py::arg("language"),
        py::arg("interpolate_method") = "nearest",
        py::arg("return_char_alignments") = false,
        py::arg("avg_logprob") = py::none());

    // Phase-3B "path 2": raw logits (T, V) -> log_softmax + OOV wildcard column +
    // tokens, the post-forward steps that used to be torch in the Python facade
    // (alignment.py:285,294-305). Pure (no ORT) so it's always built. Returns
    // (emission (T, V') float32, tokens). The caller then feeds align_assemble.
    m.def(
        "align_emission_post",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>&
               logits,
           int blank_id, const std::string& text_clean,
           const std::map<std::string, int>& dictionary) {
            if (logits.ndim() != 2)
                throw std::invalid_argument("logits must be a 2-D (T, V) array");
            auto r = al::emission_post(
                logits.data(), static_cast<std::size_t>(logits.shape(0)),
                static_cast<std::size_t>(logits.shape(1)), blank_id, text_clean,
                dictionary);
            py::array_t<float> emi({static_cast<py::ssize_t>(r.T),
                                    static_cast<py::ssize_t>(r.V)});
            std::memcpy(emi.mutable_data(), r.emission.data(),
                        r.emission.size() * sizeof(float));
            return py::make_tuple(std::move(emi), std::move(r.tokens));
        },
        py::arg("logits"), py::arg("blank_id"), py::arg("text_clean"),
        py::arg("dictionary"),
        "Raw CTC logits (T,V) -> (log_softmax'd, wildcard-extended emission (T,V'), "
        "token ids) — the torch-free post-forward step for the align_onnx path.");

    // Per-segment char-cleaning (alignment.py:252-280): the `align_driver`
    // preprocessing pass, exposed for the dep-free parity test. Returns
    // (clean_char, clean_cdx) — the cleaned string + the kept codepoint indices.
    m.def(
        "clean_segment",
        [](const std::string& text, const std::string& language,
           const std::map<std::string, int>& dictionary) {
            auto r = al::clean_segment(text, language, dictionary);
            return py::make_tuple(r.clean_char, r.clean_cdx);
        },
        py::arg("text"), py::arg("language"), py::arg("dictionary"),
        "Char-clean one transcript segment -> (clean_char, clean_cdx): lowercase "
        "(utf8proc), ' '->'|', drop leading/trailing whitespace, keep dict-key or "
        "non-space (OOV-wildcard) codepoints. Codepoint indexing == Python str.");
}

// whisperx/diarize.py speaker-assignment glue (`assign` token, Phase 4): the
// IntervalTree + assign_word_speakers dominant-by-overlap labelling. The pyannote
// DataFrame is extracted to (start, end, speaker) turns in the Python facade; the
// speaker_embeddings passthrough stays Python (a dict copy, no algorithm).
void bind_diarize(py::module_& m) {
    namespace di = whisperx::diarize;
    using TurnTuple = std::tuple<double, double, std::string>;
    m.def(
        "assign_word_speakers",
        [](const std::vector<TurnTuple>& turns, const py::handle& segments,
           bool fill_nearest) {
            std::vector<di::Turn> t;
            t.reserve(turns.size());
            for (const auto& [s, e, sp] : turns) t.push_back({s, e, sp});
            json out = di::assign_word_speakers(t, py_to_json(segments),
                                                fill_nearest);
            return json_to_py(out);
        },
        py::arg("turns"), py::arg("segments"), py::arg("fill_nearest") = false,
        "assign_word_speakers: diarization turns as (start, end, speaker) tuples + "
        "a list of segment dicts -> the same segments with 'speaker' set on each "
        "segment and each timed word by dominant overlap (fill_nearest assigns the "
        "nearest turn when there is no overlap).");
}

// Output writers — the six ResultWriter bodies (srt/vtt/txt/tsv/aud/json) +
// SubtitlesWriter.iterate_result. Pure JSON in / string out; the Python facade in
// whisperx/utils.py routes each write_result here under the `writers` token and
// writes the returned string to the file (naming + open stay Python).
void bind_writers(py::module_& m) {
    namespace wr = whisperx::writers;
    auto def_writer = [&](const char* name,
                          std::string (*fn)(const json&, const json&),
                          const char* doc) {
        m.def(
            name,
            [fn](const py::handle& result, const py::handle& options) {
                return fn(py_to_json(result), py_to_json(options));
            },
            py::arg("result"), py::arg("options"), doc);
    };
    def_writer("write_srt", &wr::write_srt, "SRT file bytes from a result dict.");
    def_writer("write_vtt", &wr::write_vtt, "WebVTT file bytes from a result dict.");
    def_writer("write_txt", &wr::write_txt, "Plain-text file bytes from a result dict.");
    def_writer("write_tsv", &wr::write_tsv, "TSV file bytes from a result dict.");
    def_writer("write_aud", &wr::write_aud, "Audacity-label file bytes from a result dict.");
    def_writer("write_json", &wr::write_json,
               "JSON file bytes (compact dump; semantic round-trip parity).");
    m.def(
        "format_timestamp",
        [](double seconds, bool always_include_hours, const std::string& marker) {
            return wr::format_timestamp(seconds, always_include_hours,
                                        marker.empty() ? '.' : marker[0]);
        },
        py::arg("seconds"), py::arg("always_include_hours") = false,
        py::arg("decimal_marker") = ".",
        "SRT/VTT timecode from seconds (integer-ms, exact).");
}

#ifdef WHISPERX_CORE_AUDIO
// --- run_job (orchestrate token) helpers — mirror asr_sherpa.py's per-segment
// assembly so the native orchestrator's transcript matches the staged path.
// Python str.strip(): trim leading/trailing ASCII whitespace (sherpa Whisper
// text never carries exotic Unicode spaces at the ends).
std::string strip_ws(const std::string& s) {
    const char* ws = " \t\n\r\f\v";
    const auto b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

// Python round(x, 3): round-half-to-even at 3 decimals (banker's, via nearbyint
// under the default FE_TONEAREST), matching asr_sherpa's round(start/end, 3).
double round3(double x) { return std::nearbyint(x * 1000.0) / 1000.0; }

// Cluster id -> pyannote-style "SPEAKER_xx" (diarize_sherpa._speaker_label).
std::string speaker_label(int cluster_id) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "SPEAKER_%02d", cluster_id);
    return std::string(buf);
}

// In-process libav* decode (replaces the ffmpeg subprocess; `decode` token).
// Built only with WHISPERX_CORE_AUDIO; the Python facade hasattr-guards so a
// dep-free module degrades to the subprocess oracle.
void bind_audio(py::module_& m) {
    namespace audio = whisperx::audio;
    m.def(
        "load_audio",
        [](const std::string& path, int sr) {
            audio::AudioBuffer buf = audio::load_audio(path, sr);
            // Copy out to a 1-D float32 numpy array (today's load_audio contract).
            py::array_t<float> arr(static_cast<py::ssize_t>(buf.samples.size()));
            std::memcpy(arr.mutable_data(), buf.samples.data(),
                        buf.samples.size() * sizeof(float));
            return arr;
        },
        py::arg("path"), py::arg("sr") = audio::kSampleRate,
        "Decode an audio file in-process (libav*) to mono float32 PCM at sr Hz "
        "in [-1, 1) — sample-for-sample equivalent to whisperx.load_audio.");

    m.def(
        "probe_duration", &audio::probe_duration, py::arg("path"),
        "Container duration in seconds from the header (libav*, no decode). "
        "Returns < 0 when the container reports no usable duration.");

    using SegOut = std::tuple<double, double, std::string>;
    m.def(
        "silero_segments",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> waveform,
           const std::string& model_path, int sr, double onset,
           double chunk_size) {
            audio::AudioBuffer buf;
            buf.sample_rate = sr;
            buf.samples.assign(waveform.data(), waveform.data() + waveform.size());
            auto segs =
                audio::silero_segments(buf, model_path, onset, chunk_size);
            std::vector<SegOut> out;
            out.reserve(segs.size());
            for (const auto& s : segs)
                out.emplace_back(s.start, s.end,
                                 s.speaker.value_or("UNKNOWN"));
            return out;
        },
        py::arg("waveform"), py::arg("model_path"),
        py::arg("sr") = audio::kSampleRate, py::arg("onset") = 0.5,
        py::arg("chunk_size") = 30.0,
        "Silero VAD (sherpa-onnx / ORT) over a float32 waveform -> list of "
        "(start, end, 'UNKNOWN') speech segments in seconds (pre-merge_chunks).");

    // Native wav2vec2 forward under ORT (Phase 3B). Replaces the torch model
    // forward; the consumer feeds the raw logits through align_emission_post.
    using WaveArray =
        py::array_t<float, py::array::c_style | py::array::forcecast>;
    py::class_<whisperx::align::Wav2Vec2Onnx>(m, "Wav2Vec2Onnx")
        .def(py::init<const std::string&, int>(), py::arg("onnx_path"),
             py::arg("num_threads") = 1)
        .def(
            "forward",
            [](whisperx::align::Wav2Vec2Onnx& self,
               const std::vector<WaveArray>& waveforms, bool batched) {
                std::vector<std::span<const float>> spans;
                spans.reserve(waveforms.size());
                for (const auto& w : waveforms) {
                    if (w.ndim() != 1)
                        throw std::invalid_argument(
                            "each waveform must be a 1-D float32 array");
                    spans.emplace_back(w.data(),
                                       static_cast<std::size_t>(w.size()));
                }
                std::vector<std::pair<std::size_t, std::size_t>> shapes;
                auto out = self.forward(spans, shapes, batched);
                py::list result;
                for (std::size_t i = 0; i < out.size(); ++i) {
                    const auto [t, v] = shapes[i];
                    py::array_t<float> arr({static_cast<py::ssize_t>(t),
                                            static_cast<py::ssize_t>(v)});
                    if (!out[i].empty())
                        std::memcpy(arr.mutable_data(), out[i].data(),
                                    out[i].size() * sizeof(float));
                    result.append(std::move(arr));
                }
                return result;
            },
            py::arg("waveforms"), py::arg("batched") = true,
            "Run the wav2vec2 forward over a list of 1-D float32 waveforms -> list "
            "of (T_i, V) raw-logit arrays. batched=True packs padded+masked batches "
            "(layer_norm models only); False runs each segment alone.");

    // Native align driver (`align_driver` token, Phase 5). Owns the whole align()
    // body for ONNX models: char-clean -> gather+forward -> emission_post ->
    // align_assemble -> word_segments. Releases the GIL for the native compute and
    // re-acquires it per progress callback (fired after each aligned segment).
    m.def(
        "align_run",
        [](const py::handle& transcript, whisperx::align::Wav2Vec2Onnx& model,
           const std::map<std::string, int>& dictionary,
           const py::array_t<float, py::array::c_style | py::array::forcecast>&
               audio,
           const std::string& language, bool batchable,
           const std::string& interpolate_method, bool return_char_alignments,
           const py::object& progress) {
            if (audio.ndim() != 1)
                throw std::invalid_argument("audio must be a 1-D float32 array");
            std::span<const float> aspan(
                audio.data(), static_cast<std::size_t>(audio.size()));
            json tr = py_to_json(transcript);
            std::function<void(double)> prog;
            if (!progress.is_none())
                prog = [&progress](double p) {
                    py::gil_scoped_acquire gil;
                    progress(p);
                };
            json out;
            {
                py::gil_scoped_release rel;
                out = whisperx::align::align_run(
                    tr, model, dictionary, aspan, language, batchable,
                    interpolate_method, return_char_alignments, prog);
            }
            return json_to_py(out);
        },
        py::arg("transcript"), py::arg("model"), py::arg("dictionary"),
        py::arg("audio"), py::arg("language"), py::arg("batchable") = false,
        py::arg("interpolate_method") = "nearest",
        py::arg("return_char_alignments") = false, py::arg("progress") = py::none(),
        "Native align() for ONNX models: transcript dict + Wav2Vec2Onnx model + "
        "dictionary + 16 kHz mono f32 audio -> {segments, word_segments}.");

    // Native Whisper ASR backend under ORT (Phase 4 / 4a). Wraps sherpa-onnx's
    // OfflineRecognizer; the Python asr_sherpa facade hands it the VAD spans +
    // the decoded waveform. Judged by WER/CER (decoupled), not exact text.
    using Span = std::pair<double, double>;
    py::class_<whisperx::asr::WhisperSherpa>(m, "WhisperSherpa")
        .def(py::init<const std::string&, const std::string&,
                      const std::string&, int, int, const std::string&,
                      const std::string&>(),
             py::arg("encoder"), py::arg("decoder"), py::arg("tokens"),
             py::arg("num_threads") = 1, py::arg("feature_dim") = 80,
             py::arg("language") = "", py::arg("task") = "transcribe")
        .def(
            "transcribe",
            [](whisperx::asr::WhisperSherpa& self,
               const py::array_t<float, py::array::c_style |
                                            py::array::forcecast>& audio,
               const std::vector<Span>& spans, const std::string& language,
               const std::string& task) {
                if (audio.ndim() != 1)
                    throw std::invalid_argument(
                        "audio must be a 1-D float32 array");
                whisperx::audio::AudioBuffer buf;
                buf.samples.assign(audio.data(), audio.data() + audio.size());
                auto chunks = self.transcribe(buf, spans, language, task);
                py::list out;
                for (const auto& c : chunks) {
                    py::dict d;
                    d["text"] = c.text;
                    d["avg_logprob"] = c.avg_logprob;
                    out.append(std::move(d));
                }
                return out;
            },
            py::arg("audio"), py::arg("spans"), py::arg("language") = "",
            py::arg("task") = "",
            "Transcribe each (start_s, end_s) VAD span of a float32 waveform with "
            "sherpa-onnx Whisper -> list of {text, avg_logprob} (one per span).")
        .def(
            "detect_language",
            [](whisperx::asr::WhisperSherpa& self,
               const py::array_t<float, py::array::c_style |
                                            py::array::forcecast>& audio) {
                if (audio.ndim() != 1)
                    throw std::invalid_argument(
                        "audio must be a 1-D float32 array");
                whisperx::audio::AudioBuffer buf;
                buf.samples.assign(audio.data(), audio.data() + audio.size());
                return self.detect_language(buf);
            },
            py::arg("audio"),
            "Whisper language ID over the first 30 s -> bare code (e.g. 'en').");

    // Native speaker diarization under ORT (Phase 4 / 4b). Wraps sherpa-onnx's
    // OfflineSpeakerDiarization (pyannote-seg-3.0 + CAM++ + FastClustering); the
    // Python diarize_sherpa facade builds the same DataFrame the pyannote path
    // returns. A/B (not parity) with community-1 — judged by speaker-count + DER.
    py::class_<whisperx::diarize::SherpaDiarizer>(m, "SherpaDiarizer")
        .def(py::init<const std::string&, const std::string&, int,
                      const std::string&, float, float, float>(),
             py::arg("segmentation"), py::arg("embedding"),
             py::arg("num_threads") = 1, py::arg("provider") = "cpu",
             py::arg("threshold") = 0.5f, py::arg("min_duration_on") = 0.3f,
             py::arg("min_duration_off") = 0.5f)
        .def(
            "diarize",
            [](whisperx::diarize::SherpaDiarizer& self,
               const py::array_t<float, py::array::c_style |
                                            py::array::forcecast>& audio,
               int num_clusters) {
                if (audio.ndim() != 1)
                    throw std::invalid_argument(
                        "audio must be a 1-D float32 array");
                whisperx::audio::AudioBuffer buf;
                buf.samples.assign(audio.data(), audio.data() + audio.size());
                auto segs = self.diarize(buf, num_clusters);
                py::list out;
                for (const auto& s : segs) {
                    py::dict d;
                    d["start"] = s.start;
                    d["end"] = s.end;
                    d["speaker"] = s.speaker;
                    out.append(std::move(d));
                }
                return out;
            },
            py::arg("audio"), py::arg("num_clusters") = 0,
            "Diarize a float32 waveform -> list of {start, end, speaker:int}. "
            "num_clusters>0 forces exactly that many speakers; <=0 = auto.")
        .def(
            "embeddings",
            [](whisperx::diarize::SherpaDiarizer& self,
               const py::array_t<float, py::array::c_style |
                                            py::array::forcecast>& audio,
               const py::list& segments) {
                if (audio.ndim() != 1)
                    throw std::invalid_argument(
                        "audio must be a 1-D float32 array");
                whisperx::audio::AudioBuffer buf;
                buf.samples.assign(audio.data(), audio.data() + audio.size());
                std::vector<whisperx::diarize::DiarSegment> segs;
                for (const auto& item : segments) {
                    auto d = item.cast<py::dict>();
                    segs.push_back({d["start"].cast<double>(),
                                    d["end"].cast<double>(),
                                    d["speaker"].cast<int>()});
                }
                auto emb = self.embeddings(buf, segs);
                py::dict out;
                for (auto& [spk, vec] : emb) out[py::int_(spk)] = vec;
                return out;
            },
            py::arg("audio"), py::arg("segments"),
            "Per-speaker mean embedding over its turns -> {speaker:int: list[float]}.")
        .def("embedding_dim", &whisperx::diarize::SherpaDiarizer::embedding_dim,
             "Embedding dimensionality of the extractor model.");

    // Native end-to-end orchestrator (`orchestrate` token, Phase 5). Decodes once
    // and drives the whole compute chain over one AudioBuffer — silero VAD ->
    // merge_chunks -> WhisperSherpa -> align_run -> SherpaDiarizer +
    // assign_word_speakers — with no Python re-entry except the align-model
    // resolver at the loading_align boundary (model load is HF/Python I/O). The
    // GIL is released for the native compute and re-acquired inside each callback
    // (progress / on_duration / resolve_align); a raising progress cb propagates
    // (stage-boundary cancellation). Returns the run_job result dict.
    m.def(
        "run_job",
        [](const std::string& audio_path, whisperx::asr::WhisperSherpa& asr,
           const std::string& silero_model_path, double vad_onset,
           double vad_offset, double chunk_size,
           const std::string& forced_language, const std::string& task,
           const py::object& resolve_align,
           const std::string& interpolate_method, bool return_char_alignments,
           const py::object& diarizer_obj, int num_clusters,
           const py::object& progress, const py::object& on_duration_cb) {
            namespace wa = whisperx::audio;
            namespace wal = whisperx::align;
            namespace wd = whisperx::diarize;
            using whisperx::orchestrate::Steps;

            wd::SherpaDiarizer* diarizer =
                diarizer_obj.is_none() ? nullptr
                                       : diarizer_obj.cast<wd::SherpaDiarizer*>();

            // The align model resolved (under the GIL) at the loading_align
            // boundary, then consumed by the aligning step.
            struct AlignHandle {
                wal::Wav2Vec2Onnx* model = nullptr;
                std::map<std::string, int> dictionary;
                bool batchable = false;
            } align_handle;

            Steps steps;

            steps.decode = [&] { return wa::load_audio(audio_path); };

            steps.transcribe =
                [&](const wa::AudioBuffer& buf) -> std::pair<json, std::string> {
                auto segs = wa::silero_segments(buf, silero_model_path, vad_onset,
                                                chunk_size);
                json merged = whisperx::vad::merge_chunks(segs, chunk_size,
                                                          vad_onset, vad_offset);
                std::vector<std::pair<double, double>> spans;
                spans.reserve(merged.size());
                for (const auto& mc : merged)
                    spans.emplace_back(mc.at("start").get<double>(),
                                       mc.at("end").get<double>());

                std::string lang = forced_language;
                if (lang.empty() && !spans.empty())
                    lang = asr.detect_language(buf);

                auto chunks = asr.transcribe(buf, spans, lang, task);
                json out = json::array();
                for (std::size_t i = 0; i < spans.size(); ++i) {
                    json seg;
                    seg["text"] = strip_ws(chunks[i].text);
                    seg["start"] = round3(spans[i].first);
                    seg["end"] = round3(spans[i].second);
                    seg["avg_logprob"] =
                        static_cast<double>(chunks[i].avg_logprob);
                    out.push_back(std::move(seg));
                }
                return {std::move(out), lang.empty() ? std::string("en") : lang};
            };

            steps.load_align = [&](const std::string& lang) {
                py::gil_scoped_acquire gil;
                py::tuple t = resolve_align(lang).cast<py::tuple>();
                align_handle.model = t[0].cast<wal::Wav2Vec2Onnx*>();
                align_handle.dictionary =
                    t[1].cast<std::map<std::string, int>>();
                align_handle.batchable = t[2].cast<bool>();
            };

            steps.align = [&](const wa::AudioBuffer& buf, const json& transcript,
                              const std::string& lang) {
                std::span<const float> aspan(buf.samples.data(),
                                             buf.samples.size());
                return wal::align_run(transcript, *align_handle.model,
                                      align_handle.dictionary, aspan, lang,
                                      align_handle.batchable, interpolate_method,
                                      return_char_alignments, nullptr);
            };

            if (diarizer != nullptr)
                steps.diarize = [&](const wa::AudioBuffer& buf, json aligned) {
                    auto raw = diarizer->diarize(buf, num_clusters);
                    std::vector<wd::Turn> turns;
                    turns.reserve(raw.size());
                    for (const auto& d : raw)
                        turns.push_back(
                            {d.start, d.end, speaker_label(d.speaker)});
                    aligned["segments"] = wd::assign_word_speakers(
                        turns, aligned["segments"], /*fill_nearest=*/false);
                    return aligned;
                };

            std::function<void(const std::string&)> stage;
            if (!progress.is_none())
                stage = [&progress](const std::string& s) {
                    py::gil_scoped_acquire gil;
                    progress(s);
                };
            std::function<void(double)> on_duration;
            if (!on_duration_cb.is_none())
                on_duration = [&on_duration_cb](double d) {
                    py::gil_scoped_acquire gil;
                    on_duration_cb(d);
                };

            json out;
            {
                py::gil_scoped_release rel;
                out = whisperx::orchestrate::run_job(steps, stage, on_duration);
            }
            return json_to_py(out);
        },
        py::arg("audio_path"), py::arg("asr"), py::arg("silero_model_path"),
        py::arg("vad_onset") = 0.5, py::arg("vad_offset") = 0.363,
        py::arg("chunk_size") = 30.0, py::arg("language") = "",
        py::arg("task") = "transcribe", py::arg("resolve_align"),
        py::arg("interpolate_method") = "nearest",
        py::arg("return_char_alignments") = false,
        py::arg("diarizer") = py::none(), py::arg("num_clusters") = 0,
        py::arg("progress") = py::none(), py::arg("on_duration") = py::none(),
        "Native end-to-end run_job for the sherpa-ASR device: decode-once "
        "AudioBuffer driven through silero VAD -> merge_chunks -> WhisperSherpa "
        "-> align_run -> (optional) SherpaDiarizer + assign_word_speakers. "
        "resolve_align(lang) -> (Wav2Vec2Onnx, dict, batchable) is called once at "
        "the loading_align boundary. Returns {segments, word_segments, language, "
        "diarized}.");
}
#endif

}  // namespace

PYBIND11_MODULE(whisperx_core, m) {
    m.doc() = "WhisperX C++ engine core (pybind11 adapter / parity oracle)";
    m.attr("__version__") = whisperx::kVersion;

    m.def("build_info_json", &whisperx::build_info_json,
          "JSON string: version + linked dependency versions.");

    m.def(
        "edit_distance",
        [](const std::vector<std::string>& a, const std::vector<std::string>& b) {
            return whisperx::text::edit_distance(a, b);
        },
        py::arg("a"), py::arg("b"),
        "Levenshtein distance between two token sequences (matches the Python "
        "golden baseline's _edit_distance).");

    m.def(
        "char_edit_distance",
        [](const std::string& a, const std::string& b) {
            return whisperx::text::edit_distance(a, b);
        },
        py::arg("a"), py::arg("b"),
        "Character-level Levenshtein distance (CER numerator).");

    bind_session_store(m);
    bind_edits(m);
    bind_vad(m);
    bind_align(m);
    bind_diarize(m);
    bind_writers(m);
#ifdef WHISPERX_CORE_AUDIO
    bind_audio(m);
#endif
}
