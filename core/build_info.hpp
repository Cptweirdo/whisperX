// Provenance for the engine core — proves nlohmann/json links end-to-end
// (Phase-0 "pull and link at least one dependency") and gives the Python oracle
// a version handle to pin goldens against.
#pragma once

#include <string>

namespace whisperx {

inline constexpr const char* kVersion = "0.0.0";  // bumped alongside pyproject

// Serialized {version, cxx_standard, nlohmann_json, ...} as a JSON string.
std::string build_info_json();

}  // namespace whisperx
