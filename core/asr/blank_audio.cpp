// strip_blank_audio — remove the "[BLANK_AUDIO]" text sherpa-onnx Whisper emits for
// silent input (declared in asr/whisper_sherpa.hpp). A pure string util with no
// sherpa/ORT/ffmpeg deps, so it lives in whisperx_core_lib and is unit-testable
// (core/tests/test_blank_audio.cpp) without the audio stage. See the header for the
// rationale (the Python faster-whisper path dropped these via no_speech_threshold).
#include "asr/whisper_sherpa.hpp"

#include <algorithm>
#include <cctype>
#include <regex>
#include <string>

namespace whisperx::asr {

namespace {
// "[BLANK_AUDIO]" with optional inner whitespace, case-insensitive.
const std::regex kBlankAudioRe(R"(\[\s*BLANK_AUDIO\s*\])",
                               std::regex_constants::icase);
// Two-or-more whitespace runs left behind after a marker is removed.
const std::regex kMultiSpaceRe(R"(\s{2,})");
}  // namespace

int strip_blank_audio(std::string& text) {
    if (text.find('[') == std::string::npos) return 0;  // fast path: no marker
    int removed = 0;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), kBlankAudioRe);
         it != std::sregex_iterator(); ++it)
        ++removed;
    if (removed == 0) return 0;
    text = std::regex_replace(text, kBlankAudioRe, " ");
    text = std::regex_replace(text, kMultiSpaceRe, " ");
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    text.erase(text.begin(),
               std::find_if(text.begin(), text.end(), not_space));
    text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(),
               text.end());
    return removed;
}

}  // namespace whisperx::asr
