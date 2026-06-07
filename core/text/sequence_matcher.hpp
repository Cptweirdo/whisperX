// A verbatim port of CPython's difflib.SequenceMatcher for the
// autojunk=False, isjunk=None case ONLY — the exact configuration app/edits.py's
// realign_words uses (`difflib.SequenceMatcher(a=..., b=..., autojunk=False)`).
//
// We need only the *matching blocks* (greedy longest-contiguous matches with
// difflib's tie-break), not get_opcodes: realign_words distinguishes "equal"
// (token survives with its timing) from everything else (untimed/dropped), and
// the equal index-pairs come straight from the blocks. The autojunk
// popularity-pruning branch is omitted deliberately — its absence is observable
// through realign_words (a token repeated >200 times must still match).
//
// Reference: CPython Lib/difflib.py (find_longest_match / get_matching_blocks).
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace whisperx::text {

// difflib.Match(a, b, size): a match of `size` elements starting at a[a], b[b].
struct Match {
    int a;
    int b;
    int size;
};

class SequenceMatcher {
public:
    SequenceMatcher(std::vector<std::string> a, std::vector<std::string> b);

    // difflib.find_longest_match: the longest matching block in a[alo:ahi] vs
    // b[blo:bhi], breaking ties toward the earliest a, then earliest b.
    Match find_longest_match(int alo, int ahi, int blo, int bhi) const;

    // difflib.get_matching_blocks: the full, coalesced list of matching blocks,
    // terminated by the (len(a), len(b), 0) sentinel. Memoized.
    const std::vector<Match>& get_matching_blocks();

private:
    std::vector<std::string> a_;
    std::vector<std::string> b_;
    // __chain_b: element -> ascending list of its indices in b (no junk purge).
    std::unordered_map<std::string, std::vector<int>> b2j_;
    std::vector<Match> matching_blocks_;
    bool computed_ = false;
};

}  // namespace whisperx::text
