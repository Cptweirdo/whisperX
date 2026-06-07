#include "build_info.hpp"

#include <nlohmann/json.hpp>

namespace whisperx {

std::string build_info_json() {
    nlohmann::json j{
        {"version", kVersion},
        {"cxx_standard", __cplusplus / 100 % 100},  // e.g. 20 for C++20
        {"nlohmann_json",
         std::to_string(NLOHMANN_JSON_VERSION_MAJOR) + "." +
             std::to_string(NLOHMANN_JSON_VERSION_MINOR) + "." +
             std::to_string(NLOHMANN_JSON_VERSION_PATCH)},
    };
    return j.dump();
}

}  // namespace whisperx
