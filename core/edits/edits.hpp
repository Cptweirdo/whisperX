// C++ port of app/edits.py — turn-level transcript editing (pure, Flask-free).
//
// Everything is carried as nlohmann::json so arbitrary segment/word keys
// (score, avg_logprob, …) survive untouched and a value-copy == Python deepcopy.
// Two contracts are load-bearing for parity and are enforced throughout:
//   * Key presence, not null: a word marks "untimed" by the ABSENCE of its
//     start/end keys (Python `"start" in word`). We read with .contains()/.find()
//     (never operator[], which would auto-insert) and emit untimed words carrying
//     NEITHER key. realign_words copies timing only when both keys are present.
//   * Float bit-parity: _interpolate_gaps is transcribed statement-for-statement
//     and this TU is compiled with -ffp-contract=off (see CMakeLists.txt) so the
//     borrow arithmetic matches CPython's never-fused IEEE-754 doubles exactly.
#pragma once

#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace whisperx::edits {

using nlohmann::json;

// app.edits module constants.
constexpr int HISTORY_LIMIT = 100;
constexpr double MIN_WORD_WIDTH = 0.1;
constexpr double SEGMENT_MIN_DURATION = 0.2;

// app.edits.NoChange — an edit that would be a no-op (e.g. reassigning a turn to
// the speaker it already has). The pybind layer registers this as a Python
// exception; the C++ store catches it internally for save_turn_reassign.
struct NoChange : std::runtime_error {
    NoChange() : std::runtime_error("edit is a no-op") {}
};

// group_turns(segments) -> array of turn objects, each:
//   {index:int, speaker:str|null, start:num|null, end:num|null,
//    seg_indices:[int], text:str}
json group_turns(const json& segments);

// turn_atoms(segments, seg_indices) -> the ordered word atoms a turn renders as:
// trimmed word tokens (raw start/end carried only when BOTH are numeric), or one
// fallback atom per wordless segment (its trimmed text + segment timing, plus a
// `stale` flag passthrough). Empty tokens are dropped. This is the SINGLE source
// for the char-offset contract: server-side views (views::turn_words) and
// apply_turn_split both derive from it, so the offsets the SPA sends never drift
// from the tokens it rendered. Each atom: {word:str, start?:num, end?:num, stale?:true}.
json turn_atoms(const json& segments, const json& seg_indices);

// Ordered unique non-null speaker keys, first-appearance order.
json distinct_speakers(const json& segments);

// Mint the next "SPEAKER_<n>" key (zero-padded to 2), one past the highest.
std::string next_speaker_key(const json& existing_keys);

// Merge consecutive same-speaker segments under `threshold` (pure; new list).
json coalesce_segments(const json& segments,
                       double threshold = SEGMENT_MIN_DURATION);

// Map edited text back onto a turn's words, keeping survivors' timing and
// interpolating typed words. `start`/`end` are turn bounds (num or null).
// Returns [] when no token keeps real timing.
json realign_words(const json& old_words, const std::string& new_text,
                   const json& start, const json& end);

// apply_turn_edit / apply_turn_reassign -> (new_segments, delta).
// A negative turn_index throws std::out_of_range (-> Python IndexError); so does
// an index past the end. apply_turn_reassign throws NoChange on a no-op.
std::pair<json, json> apply_turn_edit(const json& segments, long turn_index,
                                      const std::string& new_text);
std::pair<json, json> apply_turn_reassign(const json& segments, long turn_index,
                                          const std::string& new_speaker);

// apply_turn_split: reassign chars [sel_start, sel_end) of a turn to new_speaker,
// rebuilding it as up to 3 segments — head(orig)/middle(new)/tail(orig). Offsets
// are UTF-16 code units into the turn's turn_atoms joined by a single space (the
// SPA's DOM-selection contract); a partial-word selection snaps outward to whole
// words. Returns (new_segments, delta) like the other ops (delta carries new_len
// so undo_last restores the run). Throws std::out_of_range on a bad turn_index,
// NoChange on an empty selection or when new_speaker is the turn's current speaker.
std::pair<json, json> apply_turn_split(const json& segments, long turn_index,
                                       long sel_start, long sel_end,
                                       const std::string& new_speaker);

// undo_last(segments, history) -> (new_segments, new_history). Strict LIFO.
std::pair<json, json> undo_last(const json& segments, const json& history);

}  // namespace whisperx::edits
