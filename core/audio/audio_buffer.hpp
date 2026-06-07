// The shared "decode once" buffer — float32 PCM at a fixed sample rate (16 kHz
// mono by the audio contract). Phase 2B decodes audio once into an AudioBuffer;
// every later C++ stage slices it zero-copy instead of re-decoding.
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "audio/audio_constants.hpp"

namespace whisperx::audio {

struct AudioBuffer {
    std::vector<float> samples;            // mono PCM, range ~[-1, 1)
    int sample_rate = kSampleRate;

    // Zero-copy view of samples [f0, f1) (frame == sample index for mono).
    // Clamped to the buffer bounds so callers can over-ask the tail safely.
    std::span<const float> slice(std::size_t f0, std::size_t f1) const {
        const std::size_t n = samples.size();
        if (f0 > n) f0 = n;
        if (f1 > n) f1 = n;
        if (f1 < f0) f1 = f0;
        return std::span<const float>(samples.data() + f0, f1 - f0);
    }

    std::size_t size() const { return samples.size(); }
    double duration_s() const {
        return sample_rate > 0
                   ? static_cast<double>(samples.size()) / sample_rate
                   : 0.0;
    }
};

}  // namespace whisperx::audio
