#include "log/log.hpp"

#include <filesystem>
#include <mutex>
#include <vector>

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace fs = std::filesystem;

namespace whisperx::server::log {

namespace {
std::vector<spdlog::sink_ptr> g_sinks;
std::once_flag g_once;
spdlog::level::level_enum g_level = spdlog::level::info;

void do_init(const std::string& data_dir, const std::string& level) {
    g_level = spdlog::level::from_str(level);
    if (g_level == spdlog::level::off && level != "off")
        g_level = spdlog::level::info;  // unknown name -> info

    auto console = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();

    std::error_code ec;
    fs::path log_dir = fs::path(data_dir) / "logs";
    fs::create_directories(log_dir, ec);
    // 5 MB x 3 rotating files.
    auto file = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        (log_dir / "whisperx-server.log").string(), 5 * 1024 * 1024, 3);

    g_sinks = {console, file};
    for (auto& s : g_sinks) s->set_level(g_level);

    auto def = std::make_shared<spdlog::logger>("server", g_sinks.begin(),
                                                g_sinks.end());
    def->set_level(g_level);
    def->set_pattern("%Y-%m-%d %H:%M:%S.%e %^%l%$ %n: %v");
    spdlog::set_default_logger(def);
    spdlog::set_level(g_level);
    spdlog::flush_on(spdlog::level::warn);
    // Info lines (stage progress, "Listening on …") would otherwise sit in the
    // stdio buffer until exit, leaving the log file looking dead mid-run.
    spdlog::flush_every(std::chrono::seconds(3));
}
}  // namespace

void init(const std::string& data_dir, const std::string& level) {
    std::call_once(g_once, do_init, data_dir, level);
}

std::shared_ptr<spdlog::logger> get(const std::string& name) {
    if (auto existing = spdlog::get(name)) return existing;
    if (g_sinks.empty())  // init() not called yet — fall back to default
        return spdlog::default_logger();
    auto lg = std::make_shared<spdlog::logger>(name, g_sinks.begin(),
                                               g_sinks.end());
    lg->set_level(g_level);
    lg->set_pattern("%Y-%m-%d %H:%M:%S.%e %^%l%$ %n: %v");
    spdlog::register_logger(lg);
    return lg;
}

}  // namespace whisperx::server::log
