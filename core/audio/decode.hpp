// In-process audio decode via the ffmpeg libraries (libav*), replacing the
// ffmpeg *subprocess* in whisperx/audio.py::load_audio. Decodes any container/
// codec to mono / sr Hz / float32 PCM in [-1, 1), matching the subprocess flags
// (-ac 1 -ar <sr> -f s16le -acodec pcm_s16le, then / 32768.0) sample-for-sample.
#pragma once

#include <stdexcept>
#include <string>

#include "audio/audio_buffer.hpp"
#include "audio/audio_constants.hpp"

namespace whisperx::audio {

// Thrown on any decode failure; the message mirrors the Python RuntimeError
// ("Failed to load audio: ...") so the host contract is unchanged.
class DecodeError : public std::runtime_error {
public:
    explicit DecodeError(const std::string& what)
        : std::runtime_error("Failed to load audio: " + what) {}
};

// Open `path`, demux the best audio stream, decode + downmix to mono + resample
// to `sr` Hz + convert to s16, returning float32 samples / 32768.0.
AudioBuffer load_audio(const std::string& path, int sr = kSampleRate);

}  // namespace whisperx::audio
