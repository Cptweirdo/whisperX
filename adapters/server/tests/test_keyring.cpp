// Catch2 tests for the OS keyring wrapper. The roundtrip needs a real keyring
// backend, which a headless CI box (no Secret Service) won't have — so the case
// skips gracefully when available() is false rather than failing. Uses a key
// unique to the test so it never clobbers a real stored hf_token.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <string>

#include "secrets/keyring.hpp"

namespace ks = whisperx::server::secrets;

TEST_CASE("resolve_hf_token honours the env override first", "[keyring]") {
    setenv("HF_TOKEN", "env-wins-token", 1);
    auto t = ks::resolve_hf_token();
    REQUIRE(t.has_value());
    REQUIRE(*t == "env-wins-token");
    unsetenv("HF_TOKEN");
}

TEST_CASE("keyring set/get/erase roundtrip", "[keyring]") {
    if (!ks::available()) {
        SKIP("no OS keyring backend on this host");
    }
    const std::string key = "__wx_test_roundtrip__";
    const std::string val = "s3cr3t-value";

    ks::set(key, val);
    auto got = ks::get(key);
    REQUIRE(got.has_value());
    REQUIRE(*got == val);

    ks::erase(key);
    REQUIRE_FALSE(ks::get(key).has_value());
}
