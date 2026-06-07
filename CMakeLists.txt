# WhisperX C++ engine core — Phase 0 scaffold (see docs/cpp-core-handoff.md).
#
# Builds:
#   * whisperx_core_lib   — static lib of pure, ported algorithms (core/)
#   * whisperx_core       — pybind11 Python module (adapters/py/), the oracle
#   * whisperx_core_tests — Catch2 + CTest, built with ASan/LeakSanitizer
#
# Dependencies: light deps (pybind11, Catch2, nlohmann/json) are pulled via
# FetchContent so a fresh checkout builds with no extra setup. The production
# toolchain is **vcpkg** (see vcpkg.json) — pass its toolchain file and the same
# find_package() calls resolve against vcpkg instead (FIND_PACKAGE_ARGS below).
# The heavy runtime deps (ONNX Runtime, sherpa-onnx, SQLiteCpp, Eigen, Google
# Benchmark) are declared in vcpkg.json and wired in as their stages land.
cmake_minimum_required(VERSION 3.24)
# C is enabled for SQLiteCpp's bundled SQLite amalgamation (sqlite3.c).
project(whisperx_core LANGUAGES CXX C VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(WHISPERX_CORE_SANITIZE "Build native tests with ASan + LeakSanitizer" ON)
option(WHISPERX_CORE_PYMODULE "Build the pybind11 Python module" ON)
option(WHISPERX_CORE_TESTS "Build the Catch2 test suite" ON)

include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG v3.11.3
  FIND_PACKAGE_ARGS NAMES nlohmann_json)
FetchContent_MakeAvailable(nlohmann_json)

# SQLiteCpp — RAII wrapper over the genuine SQLite C API (Phase 1 DB layer).
# Bundles the SQLite amalgamation (SQLITECPP_INTERNAL_SQLITE, default ON) so the
# on-disk file format + WAL + Online Backup API are guaranteed identical to what
# Python's sqlite3 produced, with no system-SQLite dependency.
set(SQLITECPP_INTERNAL_SQLITE ON CACHE BOOL "" FORCE)
set(SQLITECPP_RUN_CPPLINT OFF CACHE BOOL "" FORCE)
set(SQLITECPP_RUN_CPPCHECK OFF CACHE BOOL "" FORCE)
set(SQLITECPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SQLITECPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  SQLiteCpp
  GIT_REPOSITORY https://github.com/SRombauts/SQLiteCpp
  GIT_TAG 3.3.2
  FIND_PACKAGE_ARGS NAMES SQLiteCpp)
FetchContent_MakeAvailable(SQLiteCpp)

# --- pure core library -------------------------------------------------------
add_library(whisperx_core_lib STATIC
  core/text/edit_distance.cpp
  core/text/sequence_matcher.cpp
  core/edits/edits.cpp
  core/db/session_store.cpp
  core/vad/merge_chunks.cpp
  core/build_info.cpp)
target_include_directories(whisperx_core_lib PUBLIC core)
target_link_libraries(whisperx_core_lib PUBLIC
  nlohmann_json::nlohmann_json SQLiteCpp)
target_compile_options(whisperx_core_lib PRIVATE -Wall -Wextra)
set_target_properties(whisperx_core_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# edits.cpp's gap-interpolation borrow arithmetic must match CPython's
# never-fused IEEE-754 doubles bit-for-bit (the parity tests assert exact ==).
# Disable FP contraction (FMA) on this TU only; guard to GCC/Clang.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(core/edits/edits.cpp
    PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

# --- pybind11 oracle module --------------------------------------------------
if(WHISPERX_CORE_PYMODULE)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.13.6
    FIND_PACKAGE_ARGS NAMES pybind11)
  FetchContent_MakeAvailable(pybind11)
  pybind11_add_module(whisperx_core adapters/py/whisperx_core.cpp)
  target_link_libraries(whisperx_core PRIVATE whisperx_core_lib)
endif()

# --- Catch2 tests (CTest) ----------------------------------------------------
if(WHISPERX_CORE_TESTS)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2
    GIT_TAG v3.7.1
    FIND_PACKAGE_ARGS NAMES Catch2)
  FetchContent_MakeAvailable(Catch2)

  enable_testing()
  add_executable(whisperx_core_tests
    core/tests/test_edit_distance.cpp
    core/tests/test_session_store.cpp
    core/tests/test_sequence_matcher.cpp
    core/tests/test_edits.cpp
    core/tests/test_merge_chunks.cpp)
  target_link_libraries(whisperx_core_tests
    PRIVATE whisperx_core_lib Catch2::Catch2WithMain)

  # ASan + LeakSanitizer on the native test target from the first build (the
  # memory-safety gate). The pybind module is left clean: ASan inside a dlopened
  # Python extension needs libasan LD_PRELOAD'd, which we avoid here.
  if(WHISPERX_CORE_SANITIZE)
    target_compile_options(whisperx_core_tests PRIVATE
      -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(whisperx_core_tests PRIVATE
      -fsanitize=address,undefined)
  endif()

  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(Catch)
  catch_discover_tests(whisperx_core_tests)
endif()
