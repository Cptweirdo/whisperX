// Display formatting, ported from the server's Jinja/Python helpers so the SPA
// renders the same strings the old templates did.

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** Human duration for cards/metadata: "1h 05m" with hours, else "23m 45s".
 *  Mirrors server.py::_fmt_duration. */
export function fmtDuration(sec: number | null | undefined): string {
  let s = Math.floor(sec || 0);
  const h = Math.floor(s / 3600);
  s -= h * 3600;
  const m = Math.floor(s / 60);
  s -= m * 60;
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}m ${String(s).padStart(2, "0")}s`;
}

/** Clock for the audio player: "m:ss" or "h:mm:ss". */
export function fmtClock(sec: number | null | undefined): string {
  let s = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(s / 3600);
  s -= h * 3600;
  const m = Math.floor(s / 60);
  s -= m * 60;
  const ss = String(s).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

/** Transcript timestamp, mirrors render.py::_fmt_ts: "m:ss" or "h:mm:ss". */
export function fmtTs(sec: number | null | undefined): string {
  if (sec === null || sec === undefined) return "--:--";
  return fmtClock(sec);
}

/** "Jun 07, 2026 · 14:30" from an ISO timestamp. */
export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const mon = MONTHS[d.getMonth()];
  const day = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${mon} ${day}, ${d.getFullYear()} · ${hh}:${mm}`;
}

/** "SPEAKER_00" -> "Speaker 1"; pass anything else through. Mirrors
 *  render.py::_speaker_label (the server already resolves labels for `turns`,
 *  but the client needs this for the reassign menu / live edits). */
export function speakerLabel(raw: string | null | undefined): string {
  if (!raw) return "Speaker";
  const m = /^SPEAKER_(\d+)$/.exec(String(raw));
  return m ? `Speaker ${parseInt(m[1], 10) + 1}` : String(raw);
}
