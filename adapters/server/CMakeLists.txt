# Native oat++ HTTP/SSE server — the C++ replacement for the Python Flask app/.
# Links whisperx_core_audio (decode/VAD/ASR/align/diarize/assign + SessionStore +
# writers, all native) + oat++ (transport) + spdlog (logging). No Python.
#
# Sources are added incrementally as the host-swap slices land (see
# docs/cpp-core-spa-architecture.md §3). main + log + config are the foundation.
add_executable(whisperx_server
  main.cpp
  log/log.cpp
  config.cpp
  sse/broker.cpp
  sse/sse_response.cpp
  jobs/jobs.cpp
  jobs/runner.cpp
  models/assets.cpp
  models/model_manager.cpp
  assets/downloader.cpp
  secrets/keyring.cpp
  secrets/hf_verify.cpp
  encoding/url.cpp
  oauth/crypto.cpp
  oauth/pkce.cpp
  oauth/credentials.cpp
  oauth/client.cpp
  oauth/flow.cpp
  oauth/backup_service.cpp
  drive/drive_client.cpp
  translate/google.cpp
  translate/overlay.cpp
  translate/queue.cpp
  http/curl_client.cpp
  http/http_util.cpp
  http/views.cpp)

target_include_directories(whisperx_server PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(whisperx_server PRIVATE
  whisperx_core_audio
  oatpp
  spdlog::spdlog
  nlohmann_json::nlohmann_json
  CURL::libcurl
  LibArchive::LibArchive
  keychain)

target_compile_options(whisperx_server PRIVATE -Wall -Wextra)
set_target_properties(whisperx_server PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# --- server unit tests (Catch2 + CTest, ASan/UBSan) --------------------------
# The transport-agnostic pieces (broker, config) are unit-tested in isolation.
# Catch2 is already FetchContent-available from the root tests block.
if(WHISPERX_CORE_TESTS)
  add_executable(whisperx_server_tests
    tests/test_broker.cpp
    tests/test_config.cpp
    tests/test_jobs.cpp
    tests/test_url.cpp
    tests/test_downloader.cpp
    tests/test_keyring.cpp
    tests/test_oauth.cpp
    tests/test_drive.cpp
    tests/test_translate.cpp
    sse/broker.cpp
    jobs/jobs.cpp
    log/log.cpp
    config.cpp
    encoding/url.cpp
    oauth/crypto.cpp
    oauth/pkce.cpp
    oauth/credentials.cpp
    oauth/client.cpp
    drive/drive_client.cpp
    assets/downloader.cpp
    secrets/keyring.cpp
    translate/google.cpp
    translate/overlay.cpp
    http/curl_client.cpp)
  target_include_directories(whisperx_server_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(whisperx_server_tests PRIVATE
    Catch2::Catch2WithMain nlohmann_json::nlohmann_json
    whisperx_core_lib spdlog::spdlog
    CURL::libcurl LibArchive::LibArchive keychain)
  if(WHISPERX_CORE_SANITIZE)
    target_compile_options(whisperx_server_tests PRIVATE
      -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(whisperx_server_tests PRIVATE
      -fsanitize=address,undefined)
  endif()
  catch_discover_tests(whisperx_server_tests)
endif()
