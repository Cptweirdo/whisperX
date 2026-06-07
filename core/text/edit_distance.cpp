#include "text/edit_distance.hpp"

#include <algorithm>
#include <numeric>

namespace whisperx::text {

std::size_t edit_distance(const std::vector<std::string>& a,
                          const std::vector<std::string>& b) {
    std::vector<std::size_t> prev(b.size() + 1);
    std::iota(prev.begin(), prev.end(), std::size_t{0});
    std::vector<std::size_t> cur(b.size() + 1);
    for (std::size_t i = 1; i <= a.size(); ++i) {
        cur[0] = i;
        for (std::size_t j = 1; j <= b.size(); ++j) {
            const std::size_t sub = prev[j - 1] + (a[i - 1] != b[j - 1] ? 1 : 0);
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, sub});
        }
        std::swap(prev, cur);
    }
    return prev[b.size()];
}

std::size_t edit_distance(const std::string& a, const std::string& b) {
    std::vector<std::string> ta, tb;
    ta.reserve(a.size());
    tb.reserve(b.size());
    for (char c : a) ta.emplace_back(1, c);
    for (char c : b) tb.emplace_back(1, c);
    return edit_distance(ta, tb);
}

}  // namespace whisperx::text
