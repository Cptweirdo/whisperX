#include "audio/decode.hpp"

#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

namespace whisperx::audio {

namespace {

std::string av_err(int errnum) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(errnum, buf, sizeof(buf));
    return buf;
}

// RAII guards so every ffmpeg context is released on any error path (ASan-clean).
struct FormatCtx {
    AVFormatContext* p = nullptr;
    ~FormatCtx() { if (p) avformat_close_input(&p); }
};
struct CodecCtx {
    AVCodecContext* p = nullptr;
    ~CodecCtx() { if (p) avcodec_free_context(&p); }
};
struct SwrCtx {
    SwrContext* p = nullptr;
    ~SwrCtx() { if (p) swr_free(&p); }
};
struct Packet {
    AVPacket* p = av_packet_alloc();
    ~Packet() { if (p) av_packet_free(&p); }
};
struct Frame {
    AVFrame* p = av_frame_alloc();
    ~Frame() { if (p) av_frame_free(&p); }
};

}  // namespace

// Mirrors the subprocess in whisperx/audio.py:44 — one universal libav path:
// demux best audio stream, decode, then libswresample downmix->mono /
// resample-> sr / convert-> s16 (swresample defaults: SWR engine, no dither,
// identical to the ffmpeg CLI of the same version), then s / 32768.0 -> float32.
AudioBuffer load_audio(const std::string& path, int sr) {
    FormatCtx fmt;
    int ret = avformat_open_input(&fmt.p, path.c_str(), nullptr, nullptr);
    if (ret < 0) throw DecodeError(av_err(ret));

    ret = avformat_find_stream_info(fmt.p, nullptr);
    if (ret < 0) throw DecodeError(av_err(ret));

    const AVCodec* decoder = nullptr;
    int stream_index =
        av_find_best_stream(fmt.p, AVMEDIA_TYPE_AUDIO, -1, -1, &decoder, 0);
    if (stream_index < 0)
        throw DecodeError("no audio stream: " + av_err(stream_index));

    CodecCtx dec;
    dec.p = avcodec_alloc_context3(decoder);
    if (!dec.p) throw DecodeError("avcodec_alloc_context3 failed");

    ret = avcodec_parameters_to_context(
        dec.p, fmt.p->streams[stream_index]->codecpar);
    if (ret < 0) throw DecodeError(av_err(ret));
    dec.p->pkt_timebase = fmt.p->streams[stream_index]->time_base;

    ret = avcodec_open2(dec.p, decoder, nullptr);
    if (ret < 0) throw DecodeError(av_err(ret));

    // Output contract: mono / sr Hz / packed s16.
    AVChannelLayout out_layout = AV_CHANNEL_LAYOUT_MONO;
    SwrCtx swr;
    Packet pkt;
    Frame frame;
    if (!pkt.p || !frame.p) throw DecodeError("alloc failed");

    AudioBuffer out;
    out.sample_rate = sr;
    bool swr_ready = false;

    // Reserve the whole output up front from the container duration so the per-frame
    // appends never reallocate. The old per-frame reserve(size()+got) grew capacity
    // by exactly `got` each frame (libstdc++ reserve allocates no headroom), so every
    // frame reallocated + copied the entire buffer — O(n^2) in length. Measured: 16x
    // the audio took 150x the decode time; a ~100 MB wav spent minutes here.
    if (fmt.p->duration > 0) {
        const double secs = static_cast<double>(fmt.p->duration) / AV_TIME_BASE;
        out.samples.reserve(static_cast<std::size_t>(secs * sr) + sr);  // +1 s margin
    }

    auto convert_frame = [&](const AVFrame* fr) {
        if (!swr_ready) {
            ret = swr_alloc_set_opts2(
                &swr.p, &out_layout, AV_SAMPLE_FMT_S16, sr, &fr->ch_layout,
                static_cast<AVSampleFormat>(fr->format), fr->sample_rate, 0,
                nullptr);
            if (ret < 0) throw DecodeError(av_err(ret));
            ret = swr_init(swr.p);
            if (ret < 0) throw DecodeError(av_err(ret));
            swr_ready = true;
        }
        // Worst-case output sample count for this input frame (+ buffered delay).
        int max_out = swr_get_out_samples(swr.p, fr->nb_samples);
        if (max_out < 0) throw DecodeError(av_err(max_out));
        int16_t* buf = nullptr;
        int linesize = 0;
        ret = av_samples_alloc(reinterpret_cast<uint8_t**>(&buf), &linesize, 1,
                               max_out, AV_SAMPLE_FMT_S16, 0);
        if (ret < 0) throw DecodeError(av_err(ret));
        int got = swr_convert(swr.p, reinterpret_cast<uint8_t**>(&buf), max_out,
                              const_cast<const uint8_t**>(fr->extended_data),
                              fr->nb_samples);
        if (got < 0) {
            av_freep(&buf);
            throw DecodeError(av_err(got));
        }
        // No per-frame reserve: the up-front reserve covers the whole stream, and
        // push_back's geometric growth absorbs any duration under-estimate in O(n).
        for (int i = 0; i < got; ++i)
            out.samples.push_back(static_cast<float>(buf[i]) / 32768.0f);
        av_freep(&buf);
    };

    auto drain_decoder = [&]() {
        while (true) {
            ret = avcodec_receive_frame(dec.p, frame.p);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) throw DecodeError(av_err(ret));
            convert_frame(frame.p);
            av_frame_unref(frame.p);
        }
    };

    // Demux -> decode loop (only the chosen audio stream).
    while ((ret = av_read_frame(fmt.p, pkt.p)) >= 0) {
        if (pkt.p->stream_index == stream_index) {
            ret = avcodec_send_packet(dec.p, pkt.p);
            if (ret < 0 && ret != AVERROR(EAGAIN)) {
                av_packet_unref(pkt.p);
                throw DecodeError(av_err(ret));
            }
            drain_decoder();
        }
        av_packet_unref(pkt.p);
    }
    if (ret != AVERROR_EOF && ret < 0) throw DecodeError(av_err(ret));

    // Flush the decoder, then flush any samples buffered in the resampler.
    avcodec_send_packet(dec.p, nullptr);
    drain_decoder();
    if (swr_ready) {
        while (true) {
            int pending = swr_get_out_samples(swr.p, 0);
            if (pending <= 0) break;
            int16_t* buf = nullptr;
            int linesize = 0;
            ret = av_samples_alloc(reinterpret_cast<uint8_t**>(&buf), &linesize,
                                   1, pending, AV_SAMPLE_FMT_S16, 0);
            if (ret < 0) throw DecodeError(av_err(ret));
            int got = swr_convert(swr.p, reinterpret_cast<uint8_t**>(&buf),
                                  pending, nullptr, 0);
            if (got < 0) {
                av_freep(&buf);
                throw DecodeError(av_err(got));
            }
            for (int i = 0; i < got; ++i)
                out.samples.push_back(static_cast<float>(buf[i]) / 32768.0f);
            av_freep(&buf);
            if (got == 0) break;
        }
    }

    return out;
}

}  // namespace whisperx::audio
