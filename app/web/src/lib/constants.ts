// Mirrors of small server-side label maps the API doesn't otherwise expose.

export const DEVICE_LABELS: Record<string, string> = {
  cpu: "CPU",
  cuda: "GPU (CUDA)",
  mlx: "Apple GPU (MLX)",
  whispercpp: "whisper.cpp (Metal)",
};

export const DEVICE_NOTES: Record<string, string> = {
  cpu: "Always available. Slowest; fine for short clips.",
  cuda: "NVIDIA GPU acceleration.",
  mlx: "Apple Silicon GPU via MLX.",
  whispercpp: "Apple Silicon Metal via whisper.cpp — fastest on Mac.",
};

// Pipeline stage → label for in-progress rows (mirrors base.html STAGE_LABELS).
export const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  decoding: "Decoding audio",
  transcribing: "Transcribing",
  loading_align: "Loading alignment model",
  aligning: "Aligning words",
  diarizing: "Identifying speakers",
};

/** "~1m 39s left" / "~42s left" — a loose estimate. */
export function fmtEta(s: number): string {
  s = Math.round(s);
  if (s < 60) return `~${s}s left`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  return r ? `~${m}m ${r}s left` : `~${m}m left`;
}
