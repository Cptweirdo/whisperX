// pybind11 adapter — the Python-facing face of the C++ engine core and the
// parity oracle for the strangler-fig migration. As each stage lands in `core/`
// it is exposed here and diffed against the live Python pipeline from pytest.
//
// Phase 0: the pure text metric + build provenance.
// Phase 1: the SQLite `SessionStore` (DB-only surface) — `app.store.SessionStore`
//          forwards its DB methods here when `WHISPERX_CORE_STAGES` contains `db`.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "audio/audio_constants.hpp"
#include "build_info.hpp"
#include "db/session_store.hpp"
#include "edits/edits.hpp"
#include "text/edit_distance.hpp"
#include "text/sequence_matcher.hpp"
#include "vad/merge_chunks.hpp"

#ifdef WHISPERX_CORE_AUDIO
#include <pybind11/numpy.h>

#include "audio/decode.hpp"
#include "audio/vad_silero.hpp"
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

#ifdef WHISPERX_CORE_AUDIO
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
#ifdef WHISPERX_CORE_AUDIO
    bind_audio(m);
#endif
}
