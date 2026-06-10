// Catch2 tests for config.cpp — the hand-rolled .env loader (port of
// server.py::_load_dotenv) and the typed env readers. The parser lives in an
// anonymous namespace, so we exercise it through the public load_dotenv_chain:
// write a .env beside the (fake) exe_dir and assert what lands in the
// environment. Real-env-wins precedence is checked via WHISPERX_DATA_DIR.
//
// NB: setenv mutations persist for the whole test process, so every case uses
// keys unique to that case (no cross-test bleed) and unsets what it pre-seeds.
#include <catch2/catch_test_macros.hpp>

#include "posix_compat.hpp"

#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include "config.hpp"

namespace fs = std::filesystem;
using namespace whisperx::server;

namespace {

fs::path temp_dir() {
    auto dir = fs::temp_directory_path() /
               ("wxcfg-" + std::to_string(::getpid()) + "-" +
                std::to_string(std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count()));
    fs::create_directories(dir);
    return dir;
}

void write_file(const fs::path& p, const std::string& body) {
    std::ofstream(p) << body;
}

// getenv as optional-ish string; empty marker for "unset".
std::string env_or(const char* key, const std::string& miss) {
    const char* v = std::getenv(key);
    return v ? std::string(v) : miss;
}

}  // namespace

TEST_CASE("load_dotenv parses KEY=VALUE, strips quotes/comments, trims",
          "[config]") {
    auto dir = temp_dir();
    write_file(dir / ".env",
               "WXT_PLAIN=hello\n"
               "  WXT_TRIM_KEY  =  spaced  \n"
               "WXT_DQUOTE=\"with spaces\"\n"
               "WXT_SQUOTE='single'\n"
               "WXT_INLINE=value # trailing comment\n"
               "WXT_HASH_IN_QUOTES=\"keep # this\"\n"
               "# a full comment line\n"
               "\n"
               "   \n"
               "WXT_NO_EQ_LINE\n"
               "WXT_EMPTY=\n"
               "WXT_EQ_IN_VAL=a=b=c\n");

    for (auto k : {"WXT_PLAIN", "WXT_TRIM_KEY", "WXT_DQUOTE", "WXT_SQUOTE",
                   "WXT_INLINE", "WXT_HASH_IN_QUOTES", "WXT_NO_EQ_LINE",
                   "WXT_EMPTY", "WXT_EQ_IN_VAL"})
        ::unsetenv(k);

    load_dotenv_chain(dir.string());

    CHECK(env_or("WXT_PLAIN", "<unset>") == "hello");
    CHECK(env_or("WXT_TRIM_KEY", "<unset>") == "spaced");  // key + value trimmed
    CHECK(env_or("WXT_DQUOTE", "<unset>") == "with spaces");
    CHECK(env_or("WXT_SQUOTE", "<unset>") == "single");
    CHECK(env_or("WXT_INLINE", "<unset>") == "value");
    CHECK(env_or("WXT_HASH_IN_QUOTES", "<unset>") == "keep # this");
    CHECK(env_or("WXT_NO_EQ_LINE", "<unset>") == "<unset>");  // skipped, no '='
#if defined(_WIN32)
    // The Windows environment cannot hold an empty-valued variable (setting
    // one to "" removes it), so an empty .env value reads back as unset.
    CHECK(env_or("WXT_EMPTY", "<unset>") == "<unset>");
#else
    CHECK(env_or("WXT_EMPTY", "<unset>") == "");
#endif
    CHECK(env_or("WXT_EQ_IN_VAL", "<unset>") == "a=b=c");  // only first '='

    fs::remove_all(dir);
}

TEST_CASE("real env wins over .env (setenv overwrite=0)", "[config]") {
    auto dir = temp_dir();
    write_file(dir / ".env", "WXT_PRESET=from_dotenv\n");
    ::setenv("WXT_PRESET", "from_real_env", /*overwrite=*/1);

    load_dotenv_chain(dir.string());

    CHECK(env_or("WXT_PRESET", "<unset>") == "from_real_env");
    ::unsetenv("WXT_PRESET");
    fs::remove_all(dir);
}

TEST_CASE("exe/.env wins over data_dir/.env wins over defaults.env",
          "[config]") {
    auto exe = temp_dir();
    auto data = temp_dir();

    // All three set WXT_PREC; first-set wins (setenv overwrite=0), and the chain
    // loads exe/.env, then data_dir/.env, then exe/defaults.env.
    write_file(exe / ".env", "WXT_PREC=exe\n");
    write_file(data / ".env", "WXT_PREC=data\nWXT_DATA_ONLY=data\n");
    write_file(exe / "defaults.env", "WXT_PREC=defaults\nWXT_DEFAULT_ONLY=def\n");

    ::unsetenv("WXT_PREC");
    ::unsetenv("WXT_DATA_ONLY");
    ::unsetenv("WXT_DEFAULT_ONLY");
    ::setenv("WHISPERX_DATA_DIR", data.string().c_str(), 1);

    load_dotenv_chain(exe.string());

    CHECK(env_or("WXT_PREC", "<unset>") == "exe");       // highest .env tier
    CHECK(env_or("WXT_DATA_ONLY", "<unset>") == "data");  // data tier still loads
    CHECK(env_or("WXT_DEFAULT_ONLY", "<unset>") == "def");  // defaults still load

    ::unsetenv("WXT_PREC");
    ::unsetenv("WXT_DATA_ONLY");
    ::unsetenv("WXT_DEFAULT_ONLY");
    ::unsetenv("WHISPERX_DATA_DIR");
    fs::remove_all(exe);
    fs::remove_all(data);
}

TEST_CASE("load_dotenv_chain is a no-op for a missing .env", "[config]") {
    auto dir = temp_dir();  // empty: no .env / defaults.env
    ::unsetenv("WXT_ABSENT");
    load_dotenv_chain(dir.string());  // must not throw
    CHECK(env_or("WXT_ABSENT", "<unset>") == "<unset>");
    fs::remove_all(dir);
}

TEST_CASE("env_str returns value when set/non-empty, else default", "[config]") {
    ::setenv("WXT_STR", "x", 1);
    CHECK(env_str("WXT_STR", "def") == "x");
    ::setenv("WXT_STR", "", 1);  // empty treated as unset
    CHECK(env_str("WXT_STR", "def") == "def");
    ::unsetenv("WXT_STR");
    CHECK(env_str("WXT_STR", "def") == "def");
}

TEST_CASE("env_opt is nullopt when unset/empty", "[config]") {
    ::unsetenv("WXT_OPT");
    CHECK_FALSE(env_opt("WXT_OPT").has_value());
    ::setenv("WXT_OPT", "", 1);
    CHECK_FALSE(env_opt("WXT_OPT").has_value());
    ::setenv("WXT_OPT", "v", 1);
    REQUIRE(env_opt("WXT_OPT").has_value());
    CHECK(*env_opt("WXT_OPT") == "v");
    ::unsetenv("WXT_OPT");
}

TEST_CASE("env_long parses, falls back on garbage/unset", "[config]") {
    ::setenv("WXT_LONG", "42", 1);
    CHECK(env_long("WXT_LONG", 7) == 42);
    ::setenv("WXT_LONG", "notanumber", 1);
    CHECK(env_long("WXT_LONG", 7) == 7);  // stol throws -> default
    ::unsetenv("WXT_LONG");
    CHECK(env_long("WXT_LONG", 7) == 7);
}

TEST_CASE("env_double parses, falls back on garbage/unset", "[config]") {
    ::setenv("WXT_DBL", "3.5", 1);
    CHECK(env_double("WXT_DBL", 1.0) == 3.5);
    ::setenv("WXT_DBL", "xyz", 1);
    CHECK(env_double("WXT_DBL", 1.0) == 1.0);
    ::unsetenv("WXT_DBL");
    CHECK(env_double("WXT_DBL", 1.0) == 1.0);
}

TEST_CASE("data_dir honors WHISPERX_DATA_DIR override", "[config]") {
    ::setenv("WHISPERX_DATA_DIR", "/tmp/wxt-custom-data", 1);
    CHECK(data_dir() == "/tmp/wxt-custom-data");
    ::setenv("WHISPERX_DATA_DIR", "", 1);  // empty override ignored
    CHECK(data_dir() != "/tmp/wxt-custom-data");
    ::unsetenv("WHISPERX_DATA_DIR");
}

TEST_CASE("parse_device accepts cpu/cuda/coreml (any case), rejects junk",
          "[config]") {
    CHECK(parse_device("cpu") == Device::Cpu);
    CHECK(parse_device("cuda") == Device::Cuda);
    CHECK(parse_device("coreml") == Device::CoreML);
    CHECK(parse_device("CoreML") == Device::CoreML);
    CHECK(parse_device("CUDA") == Device::Cuda);
    CHECK_FALSE(parse_device("").has_value());
    CHECK_FALSE(parse_device("metal").has_value());
    CHECK_FALSE(parse_device("mlx").has_value());        // not a Device (yet)
    CHECK_FALSE(parse_device("whispercpp").has_value());  // backend, not device
}

TEST_CASE("to_string(Device) round-trips through parse_device", "[config]") {
    for (auto d : {Device::Cpu, Device::Cuda, Device::CoreML})
        CHECK(parse_device(to_string(d)) == d);
    CHECK(std::string(to_string(Device::CoreML)) == "coreml");  // sherpa provider
}

TEST_CASE("load_config reads typed knobs from the environment", "[config]") {
    ::setenv("WHISPERX_HOST", "0.0.0.0", 1);
    ::setenv("WHISPERX_PORT", "9090", 1);
    ::setenv("WHISPERX_MAX_AUDIO_HOURS", "2.5", 1);
    ::setenv("WHISPERX_MODEL", "large-v2", 1);
    ::setenv("WHISPERX_DATA_DIR", "/tmp/wxt-cfg", 1);

    Config c = load_config();
    CHECK(c.host == "0.0.0.0");
    CHECK(c.port == 9090);
    CHECK(c.max_audio_hours == 2.5);
    CHECK(c.active_model == "large-v2");
    CHECK(c.data_dir == "/tmp/wxt-cfg");

    for (auto k : {"WHISPERX_HOST", "WHISPERX_PORT", "WHISPERX_MAX_AUDIO_HOURS",
                   "WHISPERX_MODEL", "WHISPERX_DATA_DIR"})
        ::unsetenv(k);
}
