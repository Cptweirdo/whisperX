// Native oat++ HTTP/SSE server entrypoint — the C++ replacement for the Python
// Flask app/. Boots config + logging, builds the engine collaborators (session
// store, model manager, single-worker job queue, SSE broker), wires the routes,
// reconciles interrupted jobs, warms the active model, and runs the server.
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>

#include <curl/curl.h>

#include "oatpp/network/Server.hpp"
#include "oatpp/network/tcp/server/ConnectionProvider.hpp"
#include "oatpp/parser/json/mapping/ObjectMapper.hpp"
#include "oatpp/web/server/HttpConnectionHandler.hpp"
#include "oatpp/web/server/HttpRouter.hpp"

#include "config.hpp"
#include "db/session_store.hpp"
#include "http/api_controller.hpp"
#include "http/app_state.hpp"
#include "http/spa_controller.hpp"
#include "http/views.hpp"
#include "jobs/jobs.hpp"
#include "jobs/runner.hpp"
#include "log/log.hpp"
#include "models/model_manager.hpp"
#include "oauth/backup_service.hpp"
#include "sse/broker.hpp"
#include "translate/queue.hpp"

namespace fs = std::filesystem;
namespace ws = whisperx::server;

namespace {

// Resolve the built-SPA + static dirs: explicit env wins, else look beside the
// exe (<exe>/static/spa) then the dev tree (<cwd>/app/static/spa).
void resolve_spa_dirs(ws::Config& cfg, const std::string& exe_dir) {
    if (cfg.static_dir.empty()) {
        for (auto cand : {fs::path(exe_dir) / "static",
                          fs::current_path() / "app" / "static"}) {
            std::error_code ec;
            if (fs::is_directory(cand / "spa", ec)) {
                cfg.static_dir = cand.string();
                break;
            }
        }
        if (cfg.static_dir.empty())
            cfg.static_dir = (fs::path(exe_dir) / "static").string();
    }
    if (cfg.spa_dir.empty())
        cfg.spa_dir = (fs::path(cfg.static_dir) / "spa").string();
}

}  // namespace

int main(int argc, char** argv) {
    std::string exe_dir = ".";
    if (argc > 0) {
        std::error_code ec;
        fs::path p = fs::absolute(argv[0], ec);
        if (!ec) exe_dir = p.parent_path().string();
    }

    ws::load_dotenv_chain(exe_dir);
    auto cfg = ws::load_config();
    resolve_spa_dirs(cfg, exe_dir);
    ws::log::init(cfg.data_dir, cfg.log_level);
    auto logger = ws::log::get("server");
    logger->info("WhisperX native server starting (data_dir={}, spa={})",
                 cfg.data_dir, cfg.spa_dir);

    // libcurl global init — the HF-mirror downloader + token verifier use it.
    curl_global_init(CURL_GLOBAL_DEFAULT);

#ifdef __APPLE__
    // Default the CoreML EP compile cache (read by make_options in core/) so a
    // device switch's evict-and-rebuild doesn't recompile every model. Env wins.
    if (const char* v = std::getenv("WHISPERX_COREML_CACHE_DIR"); !v || !v[0]) {
        std::string coreml_cache = (fs::path(cfg.data_dir) / "coreml-cache").string();
        setenv("WHISPERX_COREML_CACHE_DIR", coreml_cache.c_str(), /*overwrite=*/0);
    }
#endif

    // --- engine collaborators (the app/server.py module globals) ------------
    ws::sse::Broker broker;
    whisperx::db::SessionStore store(cfg.data_dir);

    std::string active =
        store.get_setting("active_model", cfg.active_model).value_or(cfg.active_model);
    // Device precedence (mirrors active_model): persisted setting > WHISPERX_DEVICE
    // env (already on cfg.device) > "cpu". An unknown persisted value falls back to
    // the env-resolved default rather than failing boot.
    ws::Device device =
        ws::parse_device(store.get_setting("device", ws::to_string(cfg.device))
                             .value_or(ws::to_string(cfg.device)))
            .value_or(cfg.device);
    ws::models::ModelManager manager(
        active, device,
        [&broker](const auto& status) {
            broker.publish(ws::kModelsChannel, ws::views::models_event(status));
        },
        ws::models::DiarizeTuning{
            static_cast<float>(cfg.diarize_threshold),
            static_cast<float>(cfg.diarize_min_on),
            static_cast<float>(cfg.diarize_min_off),
            static_cast<float>(cfg.diarize_merge_threshold),
        });

    auto run_session = ws::jobs::make_run_session(store, manager, broker, cfg);
    ws::jobs::JobQueue queue(store, run_session, &broker);

    // Network-bound translation runs on its own single-worker executor so it
    // never blocks the CPU transcription queue (app/translate_job.py).
    ws::translate::TranslationQueue translate_queue(store, &broker);

    // Requeue sessions that were mid-flight before a restart.
    for (const auto& sid : store.reconcile_startup()) {
        logger->info("Requeuing session {} from before restart", sid);
        queue.submit(sid);
    }

    // Warm the active model + diarizer in the background (boot before ready).
    std::thread([&manager, logger] {
        try {
            manager.load_asr(manager.active());
            auto d = manager.ensure_diarize();
            logger->info("Models ready (active={}, diarization {}).",
                         manager.active(), d ? "ON" : "OFF");
        } catch (const std::exception& e) {
            logger->error("Active model warm failed: {}", e.what());
        }
    }).detach();

    ws::BackupService backup(cfg, broker, store);
    ws::AppState app{store, manager, queue, translate_queue, broker, backup, cfg};

    // --- oat++ server -------------------------------------------------------
    oatpp::base::Environment::init();
    {
        auto objectMapper =
            oatpp::parser::json::mapping::ObjectMapper::createShared();
        auto router = oatpp::web::server::HttpRouter::createShared();
        // Concrete routes first; the SPA catch-all ("*") LAST so it only takes
        // genuine misses (the router matches in insertion order).
        router->addController(std::make_shared<ws::ApiController>(app, objectMapper));
        router->addController(std::make_shared<ws::SpaController>(app, objectMapper));

        auto connectionHandler =
            oatpp::web::server::HttpConnectionHandler::createShared(router);
        auto connectionProvider =
            oatpp::network::tcp::server::ConnectionProvider::createShared(
                {cfg.host, static_cast<v_uint16>(cfg.port),
                 oatpp::network::Address::IP_4});

        oatpp::network::Server server(connectionProvider, connectionHandler);
        logger->info("Listening on http://{}:{}", cfg.host, cfg.port);
        server.run();
    }
    oatpp::base::Environment::destroy();

    queue.shutdown();
    translate_queue.shutdown();
    store.close();
    curl_global_cleanup();
    return 0;
}
