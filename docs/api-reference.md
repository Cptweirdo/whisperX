# WhisperX App — HTTP API Reference

> The complete HTTP surface of the Flask backend (`app/server.py`), the contract
> between the **Svelte SPA** (`app/web/`, via `src/lib/api.ts`) and the server.
> This is the **transport-layer migration contract** a future C++ HTTP/SSE server
> must reproduce (see [`cpp-core-spa-architecture.md`](./cpp-core-spa-architecture.md)
> §3) — the analogue of the session-DB contract in
> [`cpp-core-migration-plan.md`](./cpp-core-migration-plan.md) §2.
>
> Type hints are **TypeScript-flavored** (the consumer is TS). `?` = optional/
> may be absent; `| null` = present-but-nullable. Source of truth is `app/server.py`;
> route line numbers are given for traceability.

## Conventions

- **Base URL:** same origin. Dev: Vite (`:5173`) proxies `/api`, `/sessions/<id>/<sub>`,
  `/healthz`, `/models`, `/backup`, `/oauth` to Flask (`:5000`); prod: Flask serves
  both the API and the built SPA single-origin.
- **Two route families:**
  - **`/api/*`** — the JSON API the SPA fetches as data.
  - **Root routes** — things the browser hits as *URLs*: `/healthz`, the SSE
    streams, and binary endpoints (`/sessions/<id>/{audio,download,export.md,…}`).
    Everything else falls through to the **SPA shell** (client-side routing).
- **Request bodies** (`_body()`, `server.py:757`): POST handlers accept **either**
  `application/json` **or** `application/x-www-form-urlencoded` — keys are read the
  same way. Uploads (`POST /api/sessions`) use `multipart/form-data`. Examples below
  show JSON.
- **Responses:** `application/json` unless noted (binary/file or `text/event-stream`).
- **Error shape:** `{ "error": string, ...extra? }` with a non-2xx status. A few
  endpoints use bespoke keys (`store_error`, `notice`/`notice_ok`) — noted inline.
- **Auth:** none at the HTTP layer (loopback/desktop app). Secrets (HF token, Google
  key, OAuth creds) live in the OS keyring, never in requests except when *setting*
  them.
- **Status codes:** `200` ok · `201` created · `202` accepted (async) · `400`
  bad input · `404` missing · `409` conflict/busy · `413` upload too large · `500`
  server/secret-store error · `503` models not ready.

---

## Shared types

```ts
// Whisper checkpoints the client may select (pipeline.WhisperModel).
type ModelName =
  | "tiny" | "tiny.en" | "base" | "base.en"
  | "small" | "small.en" | "medium" | "medium.en"
  | "large-v2" | "large-v3" | "large-v3-turbo" | "distil-large-v3";

// Compute backends (pipeline.DEVICES).
type Device = "cpu" | "cuda" | "mlx" | "whispercpp";

// Export formats (pipeline.OUTPUT_FORMATS).
type Format = "srt" | "vtt" | "txt" | "json";

type Status = "queued" | "running" | "done" | "error";
type Stage  = "decoding" | "transcribing" | "loading_align" | "aligning" | "diarizing";

// Per-language translation state (stored as JSON in sessions.translations).
type TranslationEntry = { status: Status; service?: string; error?: string };
type Translations = Record<string, TranslationEntry>;   // keyed by BCP-47 lang tag

// A dashboard card / session header (server.py::_card).
interface Card {
  id: string;
  name: string;                 // display filename, "Untitled recording" fallback
  chip_label: string;           // "Done" | "Processing" | "Queued" | "Error" | "Unprocessed"
  chip_class: string;           // CSS hint: "chip--ok" | "chip--run" | "chip--err" | ""
  viewable: boolean;            // true only when status === "done"
  dur: string;                  // humanized duration, e.g. "1h 02m" / "5m 03s"
  date: string;                 // "Jun 07, 2026 · 14:30"
  sub: string;                  // one-line status summary
  model: ModelName | null;
  language: string | null;
  diarized: boolean;
  num_segments: number;
  status: Status;
  stage: Stage | null;
  error: string | null;
  translations: Translations;
}

// Word inside a turn (server.py::_turn_words). Times rounded to 3dp; absent when
// the aligner produced no word/segment timing.
interface TurnWord { text: string; start?: number; end?: number; stale?: boolean }

// Speaker-grouped turn the SPA renders (server.py::_build_turns). `index` matches
// the edit/reassign endpoints (from edits.group_turns) and is stable across edits.
interface Turn {
  index: number;
  speaker: string | null;       // raw speaker key, e.g. "SPEAKER_00"
  label: string;                // resolved display name
  start: number | null;
  end: number | null;
  words: TurnWord[];
  text: string;
}

// Raw aligned segment (whisperx/schema.py::SingleAlignedSegment), surfaced
// alongside turns for clients that want pre-grouping data.
interface Segment {
  start: number;
  end: number;
  text: string;
  avg_logprob?: number;
  words?: { word: string; start?: number; end?: number; score?: number; speaker?: string }[];
  speaker?: string;
  stale?: boolean;              // translation overlay: original-text fallback
}

// Whisper model-manager status (pipeline.ModelManager.status()).
interface ModelStatus {
  active: ModelName;
  device: Device;
  cuda_available: boolean;
  mlx_available: boolean;
  whispercpp_available: boolean;
  diarize: boolean;             // diarizer object loaded right now
  diarize_error: string | null;
  diarize_available: boolean;   // can diarize (vendored model or HF token) — drive UI off this
  diarize_source: string;       // "local" | "hf" | "none"
  diarize_version: object | null;
  diarize_token: boolean;       // an HF token is in effect
  models: { name: ModelName; loaded: boolean; loading: boolean; error: string | null }[];
}

// Backup card (server.py::_backup_json = BackupService.status() + view fields).
interface BackupStatus {
  state: string;                // "idle" | "backing_up" | "restoring" | "conflict" | "error" | …
  linked: boolean;
  backend: string | null;       // "gdrive" | "local" | null
  dirty: boolean | null;        // local changes pending push (null when unlinked)
  last_root: string | null;
  last_backup_at: string | null;   // ISO-8601
  last_error: string | null;
  interval: number | null;         // periodic push seconds
  provider_label: string;          // "Google Drive" | "Local folder" | "Cloud backup"
  last_human: string | null;       // "5m ago" / "Jun 7, 14:30"
  folder: string | null;           // gdrive folder name (gdrive backend only)
  remote: RemoteState | null;
  notice?: string | null;          // transient banner text
  notice_ok?: boolean;
}

interface RemoteState {
  exists: true;
  entries: number;
  total_size: number;
  size_human: string;
  created_at: string | null;
}
```

---

## Health

### `GET /healthz` · `server.py:304`
Liveness/readiness probe (Tauri shell polls this before loading the UI).
```ts
200 → { status: "ok"; models_ready: boolean }   // models_ready flips true once the active model warms
```

---

## Sessions

### `GET /api/sessions` · `server.py:838`
List all sessions (newest first) plus a dashboard summary.
```ts
200 → {
  sessions: Card[];
  summary: { count: number; transcribed: string; total_audio: string; pct: number };
}
```

### `POST /api/sessions` · `server.py:844`
Upload audio and enqueue transcription. **`multipart/form-data`.**
```ts
// Form fields:
audio:        File          // required
model?:       ModelName     // defaults to the active model; invalid → 400
name?:        string        // display name; defaults to the uploaded filename
language?:    string        // "" / omitted = auto-detect
min_speakers?: string       // integer-as-string; non-digits ignored
max_speakers?: string

201 → { id: string; status: "queued" }
400 → { error: "No audio file uploaded." } | { error: "Unknown model: <x>" }
413 → { error: "File too large."; max_mb: number }   // exceeds WHISPERX_MAX_UPLOAD_MB
503 → { error: "loading_models" }                     // active model not warmed yet
```

### `GET /api/sessions/<session_id>` · `server.py:888`
Full session detail: the card plus transcript + edit/speaker overlays.
```ts
200 → Card & {
  result: (AlignedTranscriptionResult & { segments: Segment[] }) | null;  // null until done
  turns?: Turn[];                 // present when result exists
  speaker_names: Record<string,string>;
  can_undo: boolean;
  created_at: string;             // ISO-8601
  updated_at: string;
  options: { language?: string | null; min_speakers?: number | null; max_speakers?: number | null };
  formats: Format[];              // which export files exist on disk
}
404 → (no body)
// note: audio_filename is stripped from the response.
```

### `POST /api/sessions/<session_id>/rename` · `server.py:911`
```ts
{ name: string }   // trimmed; empty → 400
200 → { id: string; filename: string }
400 → { error: "Name cannot be empty." }
404
```

### `POST /api/sessions/<session_id>/delete` · `server.py:922`
Cancels any running/queued job, deletes the row (cascades `speaker_names`) and the
session directory.
```ts
200 → { deleted: true }
404
```

---

## Transcript editing

All return the shared **transcript payload**:
```ts
type TranscriptPayload = { turns: Turn[]; segments: Segment[]; can_undo: boolean };
```

### `POST /api/sessions/<session_id>/turns/<turn_index>` · `server.py:930`
Edit a turn's text. `turn_index` is the `Turn.index`.
```ts
{ text: string }                 // default "" (clears)
200 → TranscriptPayload
400 → { error: "Unknown turn." } // index out of range
404
```

### `POST /api/sessions/<session_id>/undo` · `server.py:941`
Undo the last edit (text or speaker reassignment). No-op if history empty.
```ts
200 → TranscriptPayload
404
```

---

## Speakers

### `GET /api/sessions/<session_id>/speakers` · `server.py:949`
Distinct speakers in the current (overlaid) segments.
```ts
200 → { key: string; label: string }[]
404
```

### `POST /api/sessions/<session_id>/speakers` · `server.py:961`
Rename a speaker (sets/clears the display name for a speaker key).
```ts
{ speaker: string; name?: string }   // name "" clears the override
200 → { key: string; label: string }
400 → { error: "Missing speaker key." }
404
```

### `POST /api/sessions/<session_id>/turns/<turn_index>/speaker` · `server.py:974`
Reassign a turn to a speaker. Provide an existing `speaker` key, **or** a `name`
to mint a new speaker (a fresh key is allocated). Providing both renames+assigns.
```ts
{ speaker?: string; name?: string }
200 → TranscriptPayload
400 → { error: "Provide a speaker key or a name for a new speaker." }
       | { error: "Unknown turn." }
409 → { error: "A speaker named '<name>' already exists." }
404
```

---

## Models & device

### `GET /api/models` · `server.py:1002`
```ts
200 → ModelStatus
```

### `POST /api/models/active` · `server.py:1007`
Switch the active model (persisted; warms in the background).
```ts
{ model: ModelName }
200 → ModelStatus
400 → { error: "Unknown model: <x>" }
```

### `POST /api/device` · `server.py:1019`
Switch the compute device (persisted). Rejected while jobs are active.
```ts
{ device: Device }
200 → ModelStatus
400 → { error: "Unknown device: <x>" } | { error: "<reason>" }
409 → { error: "busy", ...ModelStatus }   // jobs running
```

---

## Translation

### `POST /api/sessions/<session_id>/translate` · `server.py:1035`
Enqueue a translation of a finished transcript into `target_language`.
```ts
{ target_language: string }   // BCP-47-ish; validated
200 → { lang: string; status: "running"; service: string }
400 → { error: "Invalid target language." }
       | { error: "Add a Google Translation API key in Settings first." }
409 → { error: "Transcript is not ready to translate." }   // status !== "done"
404
```

### `GET /api/sessions/<session_id>/translation/<lang>` · `server.py:1054`
The translated transcript (overlay applied over current segments).
```ts
200 → { target_language: string; turns: Turn[]; segments: Segment[] }
404   // unknown session, invalid lang, or no overlay for that lang
```

---

## Onboarding (first-run)

### `GET /api/onboarding` · `server.py:1072`
```ts
200 → {
  token: string;                 // current HF token ("" if none)
  sizes: { id: string; name: string; meta: string; note: string }[];   // model size cards (HTML in note)
  selected_size: string;         // preselected size id
  models: ModelStatus;
  diarize_model: string;
  backup: BackupStatus;
}
```

### `POST /api/onboarding/verify` · `server.py:1085`
Verify an HF token without storing it.
```ts
{ token: string }
200 → { ok: boolean; detail: string }
```

### `POST /api/onboarding` · `server.py:1092`
Finish onboarding: store token (if given+valid), set model+device, mark onboarded.
```ts
{ token?: string; model: ModelName; device: Device }
200 → { ok: true }
400 → { error: "Unknown model: <x>" } | { error: "Unknown device: <x>" }
       | { error: "<token did not verify>" }
500 → { store_error: string }    // keyring unavailable
```

---

## Settings

### `GET /api/settings` · `server.py:1147`
```ts
200 → {
  default_language: string;
  languages: { code: string; label: string }[];        // transcribe-language picker
  models: ModelStatus;
  translation_service: string;
  translation_services: { id: string; label: string }[];
  translation_languages: { code: string; name: string; native: string }[];
  google_key: { key_set: boolean };
  diarize: { version: object | null; model_name: string; token_set: boolean };
  backup: BackupStatus;
  onboarded: boolean;
}
```

### `POST /api/settings` · `server.py:1152`
```ts
{ default_language: string }
200 → { ok: true; default_language: string }
```

The secret/notice endpoints share a payload shape — `{ ...fields, notice: string, notice_ok: boolean }`:

### `POST /api/settings/hf-token` · `server.py:1164`
```ts
{ hf_token: string }
200 → { token_set: boolean; notice: string; notice_ok: true }
400 → { token_set: boolean; notice: string; notice_ok: false }   // failed verify
500 → { token_set: boolean; notice: string; notice_ok: false }   // keyring unavailable
```

### `POST /api/settings/hf-token/clear` · `server.py:1178`
```ts
200 → { token_set: boolean; notice: string; notice_ok: true }
```

### `POST /api/settings/google-key` · `server.py:1193`
```ts
{ google_key: string }
200 → { key_set: boolean; notice: string; notice_ok: true }
400 / 500 → { key_set: boolean; notice: string; notice_ok: false }
```

### `POST /api/settings/google-key/clear` · `server.py:1206`
```ts
200 → { key_set: boolean; notice: string; notice_ok: true }
```

### `POST /api/settings/translation-service` · `server.py:1215`
```ts
{ translation_service: string }
200 → { ok: true }
400 → { error: "Unknown translation service." }
```

### `POST /api/settings/diarize-model/refresh` · `server.py:1224`
Re-vendor the diarization checkpoint (needs an HF token).
```ts
200 → { version: object | null; model_name: string; token_set: boolean; notice: string; notice_ok: true }
400 → { ...same; notice_ok: false }   // no token
500 → { ...same; notice_ok: false }   // refresh failed
```

---

## Backup & restore

JSON surface; live state is pushed on the SSE streams below.

### `GET /api/backup/status` · `server.py:1247`
```ts
200 → BackupStatus
```

### `POST /api/backup/connect` · `server.py:1252`
Start the (background) Google OAuth consent flow. Watch `/backup/events` for the
terminal result.
```ts
{ backup_folder?: string }      // optional Drive folder; blank keeps stored/default
200 → BackupStatus              // when already linked (no-op)
200 → { connecting: true }      // consent flow started
```

### `POST /api/backup/disconnect` · `server.py:1263`
```ts
200 → BackupStatus   // notice: "Disconnected. Local data is untouched."
```

### `POST /api/backup/now` · `server.py:1270`
Kick a push (async).
```ts
202 → BackupStatus
409 → { error: "Backup backend is not linked." }
```

### `POST /api/backup/restore` · `server.py:1278`
Restore from the remote backup.
```ts
200 → { restored: number; backup: BackupStatus }
409 → { error: string }
```

### `POST /api/backup/bootstrap/adopt` · `server.py:1287`
Adopt the remote (pull it down) when local and remote diverge.
```ts
200 → { restored: number; backup: BackupStatus }
409 → { error: string }
```

### `POST /api/backup/bootstrap/overwrite` · `server.py:1296`
Overwrite the remote with local data.
```ts
200 → { uploaded: number; skipped: number; backup: BackupStatus }
409 → { error: string }
```

### `GET /api/backup/remote-info` · `server.py:1305`
Probe the remote (for restore/conflict prompts).
```ts
200 → { remote: RemoteState | null }
409 → { error: string }
```

---

## Binary & file downloads (root routes)

### `GET /sessions/<session_id>/audio` · `server.py:614`
Streams the stored audio inline. `200` (binary) · `404`.

### `GET /sessions/<session_id>/download/<fmt>` · `server.py:622`
Pre-generated export file as an attachment. `fmt ∈ Format`. `200` (binary) · `404`
(unknown format or file not built).

### `GET /sessions/<session_id>/translation/<lang>/download/<fmt>` · `server.py:685`
Translation export, **generated on demand** (reflects current speakers + edited-
segment fallback). `200` (binary, `transcript.translation.<lang>.<fmt>`) · `404`.

### `GET /sessions/<session_id>/export.md` · `server.py:719`
Markdown transcript of the current (edit-overlaid) segments.
`200 text/markdown` (attachment) · `404`.

---

## Server-Sent Events (root routes)

All are `text/event-stream` (`Cache-Control: no-cache`, `X-Accel-Buffering: no`),
emit JSON in each `data:` frame, send a `: keepalive` comment every 15s, and (where
applicable) replay current state on connect so reconnecting clients are correct.
Transport details: `app/sse.py::sse_response`. Client: `app/web/src/lib/sse.ts`.

### `GET /sessions/<session_id>/events` · `server.py:572`
Per-session job progress. **Terminal** on `done`/`error` (stream closes).
```ts
// initial (on connect): one of —
{ status: "done" | "error" }
{ stage: Stage; eta?: number }            // when running with a known stage
{ status: "queued" | Status }             // queued / no stage yet
// deltas: { stage: Stage; eta?: number }
// terminal: { status: "done" | "error" }
404   // unknown session
```

### `GET /sessions/<session_id>/translate/events` · `server.py:663`
Per-session translation progress. **Terminal** on `done`/`error` (per language).
```ts
// initial: { translations: Translations }
// deltas:  { lang: string; status: Status; error?: string }
404
```

### `GET /models/events` · `server.py:601`
Global model-load state. **Persistent** (no terminal).
```ts
// initial + deltas:
{
  type: "models";
  models_ready: boolean;        // the *active* model is loaded
  active: ModelName;
  bundle_error: string | null;
  diarize_available: boolean;
  diarize_error: string | null;
  models: ModelStatus["models"];
}
```

### `GET /backup/status/events` · `server.py:414`
Persistent backup sync-state stream.
```ts
// initial + deltas: { type: "backup"; state: string; status: BackupStatus }
```

### `GET /backup/events` · `server.py:523`
One-shot OAuth consent result. Replays the stored result for late subscribers, else
relays the live event; **terminal** on `linked`/`error`.
```ts
{ status: "linked"; backup: BackupStatus }
{ status: "error"; message: string; backup: BackupStatus }
```

---

## SPA shell (catch-all)

### `GET /` and `GET /<path>` · `server.py:1323`
Serves the built SPA (`app/static/spa/index.html`) for any non-reserved path so
client-side routing + deep links work.
- Reserved roots (`api`, `healthz`, `backup`, `models`, `static`, `oauth`) → `404`
  (never return the shell).
- A path with a file extension that no asset/static route served → `404`.
- SPA not built → `500` ("SPA not built — run: cd app/web && bun run build").

---

## Notes for the C++ migration

- This JSON/SSE surface is **stable and transport-agnostic** — the SPA already
  depends only on it, not on Flask. A future C++ HTTP/SSE server adapter must
  reproduce it route-for-route (paths, methods, status codes, JSON shapes, SSE
  channel semantics + terminal rules).
- The SSE design is the one with sharp edges to preserve: **DB row is the durable
  source of truth**, the broker carries live deltas; every stream replays current
  state on connect; only the per-session job and translation streams have terminal
  events. See `app/sse.py` + CLAUDE.md.
- The **multipart upload** (`POST /api/sessions`) and the **on-demand translation
  export** (`/translation/<lang>/download/<fmt>`) are the two non-trivial handlers
  (file spooling; writer invocation) — port deliberately.
</content>
</invoke>
