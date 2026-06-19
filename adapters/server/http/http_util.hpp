// Small HTTP helpers shared by the controllers: raw-JSON responses, request-body
// parsing (JSON or form-urlencoded, the _body() tolerance from server.py:781), a
// streaming file-download body, and a secure_filename port.
#pragma once

#include <fstream>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "oatpp/core/data/stream/Stream.hpp"
#include "oatpp/web/protocol/http/incoming/Request.hpp"
#include "oatpp/web/protocol/http/outgoing/Response.hpp"
#include "oatpp/web/protocol/http/outgoing/ResponseFactory.hpp"
#include "oatpp/web/protocol/http/outgoing/StreamingBody.hpp"

namespace whisperx::server::http_util {

using nlohmann::json;
namespace http = oatpp::web::protocol::http;

inline std::shared_ptr<http::outgoing::Response> json_response(
    const http::Status& status, const json& body) {
    auto resp = http::outgoing::ResponseFactory::createResponse(
        status, oatpp::String(body.dump()));
    resp->putHeader("Content-Type", "application/json");
    return resp;
}

// Parse a POST body as JSON, falling back to application/x-www-form-urlencoded
// (returns a flat object of string values). Always returns an object.
json parse_body(
    const std::shared_ptr<http::incoming::Request>& request);

// Stream a file from disk as a download (chunked); attachment toggles
// Content-Disposition. Returns 404 response if the file is missing.
std::shared_ptr<http::outgoing::Response> file_response(
    const std::string& path, const std::string& content_type, bool attachment,
    const std::string& download_name = "");

// werkzeug.secure_filename-ish: keep [A-Za-z0-9._-], collapse the rest to '_',
// strip leading dots/dashes; "" -> "" (caller substitutes a default).
std::string secure_filename(const std::string& name);

}  // namespace whisperx::server::http_util
