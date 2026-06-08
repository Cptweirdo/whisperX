#include "oauth/credentials.hpp"

#include <array>
#include <cstdio>
#include <ctime>
#include <sstream>

namespace whisperx::server::oauth {

std::string format_expiry(std::time_t t) {
    if (t == 0) return "";
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[40];
    std::snprintf(buf, sizeof(buf),
                  "%04d-%02d-%02dT%02d:%02d:%02d.000000Z", tm.tm_year + 1900,
                  tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

std::time_t parse_expiry(const std::string& s) {
    if (s.empty()) return 0;
    std::tm tm{};
    int y = 0, mo = 0, d = 0, h = 0, mi = 0, se = 0;
    if (std::sscanf(s.c_str(), "%d-%d-%dT%d:%d:%d", &y, &mo, &d, &h, &mi, &se) !=
        6)
        return 0;
    tm.tm_year = y - 1900;
    tm.tm_mon = mo - 1;
    tm.tm_mday = d;
    tm.tm_hour = h;
    tm.tm_min = mi;
    tm.tm_sec = se;
#if defined(_WIN32)
    return _mkgmtime(&tm);
#else
    return timegm(&tm);
#endif
}

namespace {
std::string str_or(const json& j, const char* k) {
    auto it = j.find(k);
    if (it != j.end() && it->is_string()) return it->get<std::string>();
    return "";
}
}  // namespace

Credentials Credentials::from_json(const json& j) {
    Credentials c;
    c.token = str_or(j, "token");
    c.refresh_token = str_or(j, "refresh_token");
    c.token_uri = str_or(j, "token_uri");
    c.client_id = str_or(j, "client_id");
    c.client_secret = str_or(j, "client_secret");
    auto sc = j.find("scopes");
    if (sc != j.end() && sc->is_array())
        for (const auto& s : *sc)
            if (s.is_string()) c.scopes.push_back(s.get<std::string>());
    auto ex = j.find("expiry");
    if (ex != j.end() && ex->is_string()) c.expiry = parse_expiry(*ex);
    return c;
}

json Credentials::to_json() const {
    json j;
    j["token"] = token;
    j["refresh_token"] = refresh_token;
    j["token_uri"] = token_uri;
    j["client_id"] = client_id;
    j["client_secret"] = client_secret;
    j["scopes"] = scopes;
    j["expiry"] = expiry ? json(format_expiry(expiry)) : json(nullptr);
    return j;
}

}  // namespace whisperx::server::oauth
