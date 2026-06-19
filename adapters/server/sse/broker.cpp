#include "sse/broker.hpp"

namespace whisperx::server::sse {

void Subscription::push(json event) {
    std::lock_guard<std::mutex> lk(mu_);
    if (closed_) return;
    if (q_.size() >= max_size_) return;  // drop on full (sse.py queue.Full: pass)
    q_.push(std::move(event));
    cv_.notify_one();
}

std::optional<json> Subscription::pop(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!cv_.wait_for(lk, timeout, [this] { return !q_.empty() || closed_; }))
        return std::nullopt;  // timed out — caller emits a keepalive
    if (q_.empty()) return std::nullopt;  // closed and drained
    json ev = std::move(q_.front());
    q_.pop();
    return ev;
}

void Subscription::close() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
    }
    cv_.notify_all();
}

std::shared_ptr<Subscription> Broker::subscribe(const std::string& channel) {
    auto sub = std::make_shared<Subscription>();
    std::lock_guard<std::mutex> lk(mu_);
    subs_[channel].insert(sub);
    return sub;
}

void Broker::unsubscribe(const std::string& channel,
                         const std::shared_ptr<Subscription>& sub) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = subs_.find(channel);
    if (it == subs_.end()) return;
    it->second.erase(sub);
    if (it->second.empty()) subs_.erase(it);
}

void Broker::publish(const std::string& channel, json event) {
    std::vector<std::shared_ptr<Subscription>> targets;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = subs_.find(channel);
        if (it == subs_.end()) return;
        targets.assign(it->second.begin(), it->second.end());
    }
    for (auto& sub : targets) sub->push(event);
}

std::size_t Broker::subscriber_count(const std::string& channel) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = subs_.find(channel);
    return it == subs_.end() ? 0 : it->second.size();
}

}  // namespace whisperx::server::sse
