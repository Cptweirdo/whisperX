#include "oauth/flow.hpp"

#include <chrono>
#include <thread>

#include "log/log.hpp"
#include "oauth/pkce.hpp"

#if defined(_WIN32)
#include <windows.h>
#include <shellapi.h>
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace whisperx::server::oauth {

namespace {
constexpr auto kConsentTimeout = std::chrono::seconds(300);
auto logger() { return log::get("oauth"); }
}  // namespace

void open_in_browser(const std::string& url) {
#if defined(_WIN32)
    ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
#if defined(__APPLE__)
    const char* opener = "open";
#else
    const char* opener = "xdg-open";
#endif
    // fork+exec (not system()) — no shell, so the URL can't be interpreted even
    // though we only ever build it from percent-encoded safe characters.
    pid_t pid = fork();
    if (pid == 0) {
        // Detach stdio so the child can't scribble on the server's terminal.
        execlp(opener, opener, url.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    } else if (pid > 0) {
        // Reap asynchronously; we don't care about the opener's exit.
        std::thread([pid] { int s = 0; waitpid(pid, &s, 0); }).detach();
    }
#endif
}

bool LinkFlow::start(DoneFn on_done) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (running_) return false;
        running_ = true;
    }
    std::thread([this, on_done = std::move(on_done)]() mutable {
        run(std::move(on_done));
    }).detach();
    return true;
}

bool LinkFlow::in_progress() const {
    std::lock_guard<std::mutex> lk(mu_);
    return running_;
}

void LinkFlow::run(DoneFn on_done) {
    std::string state = random_state();
    std::string verifier = code_verifier();
    std::string challenge = code_challenge_s256(verifier);

    std::future<CallbackData> fut;
    {
        std::lock_guard<std::mutex> lk(mu_);
        expected_state_ = state;
        promise_ = std::make_shared<std::promise<CallbackData>>();
        fut = promise_->get_future();
    }

    bool ok = false;
    std::string error;
    try {
        std::string url = client_.build_authorize_url(state, challenge);
        open_in_browser(url);
        logger()->info("Opened consent URL; awaiting callback (300s).");

        if (fut.wait_for(kConsentTimeout) != std::future_status::ready) {
            error = "Timed out waiting for authorization.";
        } else {
            CallbackData cb = fut.get();
            if (!cb.error.empty()) {
                error = cb.error;
            } else {
                client_.exchange_code(cb.code, verifier);
                ok = true;
            }
        }
    } catch (const std::exception& e) {
        error = e.what();
    }

    {
        std::lock_guard<std::mutex> lk(mu_);
        running_ = false;
        promise_.reset();
        expected_state_.clear();
    }
    if (ok)
        logger()->info("Google Drive linked.");
    else
        logger()->warn("Link flow failed: {}", error);
    if (on_done) on_done(ok, error);
}

void LinkFlow::on_callback(const std::string& code, const std::string& state,
                           const std::string& error) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!promise_) return;  // no flow pending / already settled
    CallbackData cb;
    if (!error.empty())
        cb.error = "Authorization was denied or cancelled (" + error + ").";
    else if (state != expected_state_)
        cb.error = "State mismatch — possible CSRF; authorization rejected.";
    else if (code.empty())
        cb.error = "Authorization returned no code.";
    else
        cb.code = code;
    promise_->set_value(cb);
    promise_.reset();  // ensures we only fulfil once
}

const char* LinkFlow::success_html() {
    return R"HTML(<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Connected</title>
<style>
  body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
    font-family:ui-sans-serif,system-ui,sans-serif;background:#fbf9f4;color:#1b1c19;}
  .cb{max-width:440px;text-align:center;padding:32px;}
  .badge{width:64px;height:64px;border-radius:8px;background:#1a1a18;color:#fbf9f4;
    display:flex;align-items:center;justify-content:center;margin:0 auto 24px;}
  .badge svg{width:30px;height:30px;}
  h1{font-size:26px;margin:0 0 12px;}
  p{font-size:15px;line-height:1.55;color:#5a5b56;margin:0 0 8px;}
  .note{font-size:13px;color:#8a8a83;margin-top:18px;}
</style></head><body>
  <div class="cb">
    <div class="badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
      stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="20 6 9 17 4 12"/></svg></div>
    <h1>Google Drive is connected</h1>
    <p>Authorization complete. You can close this tab and return to the app.</p>
    <p class="note">This tab will close automatically in <b id="count">5</b> seconds.</p>
  </div>
  <script>(function(){var n=5,c=document.getElementById('count');
    var t=setInterval(function(){n--;if(n<=0){clearInterval(t);window.close();}
    else if(c)c.textContent=n;},1000);})();</script>
</body></html>)HTML";
}

const char* LinkFlow::error_html() {
    return R"HTML(<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Authorization didn't complete</title>
<style>
  body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
    font-family:ui-sans-serif,system-ui,sans-serif;background:#fbf9f4;color:#1b1c19;}
  .cb{max-width:440px;text-align:center;padding:32px;}
  .badge{width:64px;height:64px;border-radius:8px;background:#f5f3ee;
    border:1px solid #cdccc4;color:#5a5b56;
    display:flex;align-items:center;justify-content:center;margin:0 auto 24px;}
  .badge svg{width:28px;height:28px;}
  h1{font-size:26px;margin:0 0 12px;}
  p{font-size:15px;line-height:1.55;color:#5a5b56;margin:0;}
</style></head><body>
  <div class="cb">
    <div class="badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
      stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></div>
    <h1>Authorization didn't complete</h1>
    <p>Google Drive was not connected. Close this tab, return to the app, and try again.</p>
  </div>
</body></html>)HTML";
}

}  // namespace whisperx::server::oauth
