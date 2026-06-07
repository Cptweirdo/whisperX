// pybind11 adapter — the Python-facing face of the C++ engine core and the
// parity oracle for the strangler-fig migration. As each stage lands in `core/`
// it is exposed here and diffed against the live Python pipeline from pytest.
//
// Phase 0: the pure text metric + build provenance.
// Phase 1: the SQLite `SessionStore` (DB-only surface) — `app.store.SessionStore`
//          forwards its DB methods here when `WHISPERX_CORE_STAGES` contains `db`.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "build_info.hpp"
#include "db/session_store.hpp"
#include "text/edit_distance.hpp"

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
        .def("close", &SessionStore::close);
}

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
}
