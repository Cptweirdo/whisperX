// pybind11 adapter — the Python-facing face of the C++ engine core and the
// parity oracle for the strangler-fig migration. As each stage lands in `core/`
// it is exposed here and diffed against the live Python pipeline from pytest.
//
// Phase 0 exposes only the pure text metric + build provenance; that is enough
// to prove the oracle loop (`import whisperx_core`, C++ result == Python result).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "build_info.hpp"
#include "text/edit_distance.hpp"

namespace py = pybind11;

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
}
