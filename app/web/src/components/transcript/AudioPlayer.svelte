<script lang="ts">
  import { fmtClock } from "../../lib/format";

  let {
    src,
    currentTime = $bindable(0),
  }: { src: string; currentTime?: number } = $props();

  let audio = $state<HTMLAudioElement | null>(null);
  let track = $state<HTMLElement | null>(null);
  let duration = $state(0);
  let playing = $state(false);
  let dragging = $state(false);

  const RATES = [1, 1.25, 1.5, 2, 0.75];
  let rateIdx = $state(0);
  let volume = $state(1);
  let muted = $state(false);
  let lastVol = 1;

  const pct = $derived(duration ? (currentTime / duration) * 100 : 0);
  const rate = $derived(RATES[rateIdx]);
  const volIcon = $derived(
    muted || volume === 0 ? "volume-mute" : volume < 0.5 ? "volume-down" : "volume-up",
  );

  let raf = 0;
  function loop() {
    if (audio) currentTime = audio.currentTime;
    raf = requestAnimationFrame(loop);
  }

  export function seek(t: number) {
    if (!audio) return;
    audio.currentTime = Math.max(0, Math.min(duration || 0, t));
    currentTime = audio.currentTime;
  }

  function toggle() {
    if (!audio) return;
    audio.paused ? audio.play() : audio.pause();
  }
  function cycleRate() {
    rateIdx = (rateIdx + 1) % RATES.length;
    if (audio) audio.playbackRate = rate;
  }
  function toggleMute() {
    if (!audio) return;
    if (muted || volume === 0) {
      muted = false;
      volume = lastVol > 0 ? lastVol : 1;
    } else {
      lastVol = volume;
      muted = true;
    }
    audio.muted = muted;
    audio.volume = volume;
  }
  function onVol(v: number) {
    volume = v;
    muted = v === 0;
    if (v > 0) lastVol = v;
    if (audio) {
      audio.volume = v;
      audio.muted = muted;
    }
  }
  function scrub(clientX: number) {
    if (!track || !duration) return;
    const r = track.getBoundingClientRect();
    seek(Math.min(1, Math.max(0, (clientX - r.left) / r.width)) * duration);
  }
</script>

<div class="player">
  <audio
    bind:this={audio}
    {src}
    preload="metadata"
    onplay={() => {
      playing = true;
      cancelAnimationFrame(raf);
      loop();
    }}
    onpause={() => {
      playing = false;
      cancelAnimationFrame(raf);
    }}
    onloadedmetadata={() => (duration = audio?.duration || 0)}
    ontimeupdate={() => {
      if (audio?.paused) currentTime = audio.currentTime;
    }}
  ></audio>

  <div class="player__controls">
    <button class="pbtn" title="Back 10s" onclick={() => seek(currentTime - 10)}>
      <sl-icon name="arrow-counterclockwise"></sl-icon>
    </button>
    <button class="pbtn pbtn--main" title="Play / pause" onclick={toggle}>
      <sl-icon name={playing ? "pause-fill" : "play-fill"}></sl-icon>
    </button>
    <button class="pbtn" title="Forward 10s" onclick={() => seek(currentTime + 10)}>
      <sl-icon name="arrow-clockwise"></sl-icon>
    </button>
  </div>

  <span class="player__time">{fmtClock(currentTime)}</span>
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    class="player__track"
    bind:this={track}
    onpointerdown={(e) => {
      dragging = true;
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      scrub(e.clientX);
    }}
    onpointermove={(e) => dragging && scrub(e.clientX)}
    onpointerup={() => (dragging = false)}
  >
    <div class="player__rail"></div>
    <div class="player__fill" style={`width:${pct}%`}></div>
    <div class="player__knob" style={`left:${pct}%`}></div>
  </div>
  <span class="player__total">{fmtClock(duration)}</span>

  <div class="player__sep"></div>
  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
  <div class="player__rate" title="Playback speed" onclick={cycleRate}>
    <sl-icon name="speedometer2"></sl-icon>
    <span>{rate % 1 === 0 ? rate.toFixed(1) : rate}x</span>
  </div>
  <div class="player__volume">
    <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
    <span class="icon-btn" title="Mute / unmute" onclick={toggleMute}>
      <sl-icon name={volIcon}></sl-icon>
    </span>
    <div class="player__volpanel">
      <sl-range
        min="0"
        max="100"
        value={muted ? 0 : Math.round(volume * 100)}
        tooltip="none"
        onsl-input={(e: any) => onVol(e.target.value / 100)}
      ></sl-range>
    </div>
  </div>
</div>
