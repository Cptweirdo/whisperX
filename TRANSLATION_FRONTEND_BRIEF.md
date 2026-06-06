# Frontend Brief — Transcript Translation

Audience: frontend engineer building the translation UI on top of the new
translation backend (PR #16). This is a scoping/handoff doc — it describes the
backend contract and the app conventions you must follow, then the UI work to do.

## What exists now (backend)

A user can translate a **finished** transcript into one or more target languages
on demand. Translation runs in the background and progress is pushed over SSE.
Output is stored **non-destructively per language** (the original transcript is
never modified), so a recording can have multiple translations at once.

Provider is pluggable; only **Google Translate** is wired today. It needs a
Google Cloud Translation API key, stored in the OS keyring (like the HF token).

The **Settings** UI is already partly built (Translation card: API-key control +
service selector). The main remaining work is the **per-recording translate UI**
on the transcript page.

## API contract

### Trigger a translation
`POST /sessions/<id>/translate`  — form field `target_language` (BCP-47-ish tag,
e.g. `es`, `pt-BR`).

- `200` → JSON `{ "lang": "es", "status": "running", "service": "google" }`
- `400` invalid language, or no API key configured
- `409` transcript not finished (`status != "done"`)
- `404` unknown session

Fire-and-forget: the work runs in the background; subscribe to the SSE stream
(below) for completion.

### Progress stream (SSE)
`GET /sessions/<id>/translate/events`

- On connect, emits the current state:
  `{ "translations": { "es": {"status":"done","service":"google"},
  "fr": {"status":"running","service":"google"} } }`
- Then per-language deltas: `{ "lang": "es", "status": "running" }` →
  `{ "lang": "es", "status": "done" }` (or `{"status":"error","error":"…"}`).
- The stream **closes** after a terminal (`done`/`error`) event.

Use the shared `openSSE(url, onData)` from `static/sse.js` (do **not** hand-roll
`new EventSource`). Reconnect/late-connect is handled — the initial event always
carries current state.

### Read a translation
`GET /sessions/<id>/translation/<lang>` → rendered transcript **HTML fragment**
(identical markup/structure to the original transcript body, speaker turns +
timed word spans). Drop it into a container and call `htmx.process(el)`.

`GET /sessions/<id>/translation/<lang>?format=json` → the raw overlay:
```json
{ "version": 1, "target_language": "es", "service": "google",
  "created_at": "…",
  "segments": [ { "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "…" } ] }
```
Note: translated segments carry `start/end/speaker/text` but **no word-level
timing** (word order differs across languages) — so word-by-word audio highlight
is not available on a translation, only segment-level.

### Download artifacts
`GET /sessions/<id>/translation/<lang>/download/<fmt>` — `fmt` ∈
`srt | vtt | txt | json`, served as an attachment.

### Which translations exist?
The session row (`GET /sessions/<id>`, and the row used to render the page) now
includes a `translations` field: `{ lang: {status, service, error?} }`. Use it
to render existing-translation chips/tabs on page load.

### Settings (already wired — for reference)
- `POST /settings/google-key` (form `google_key`) → returns `#google-key-card`
  fragment (swap `outerHTML`). Verifies the key before saving.
- `POST /settings/google-key/clear` → same fragment.
- `POST /settings/translation-service` (form `translation_service`) → small
  "Saved" fragment.
- `GET /settings` context now provides `google_key.key_set`,
  `translation_service`, and `translation_services` (`[{id,label}]`).

## App conventions you must follow (see CLAUDE.md)

- **Stack:** Flask + htmx + Shoelace web components. No SPA framework.
- **Shoelace + htmx:** htmx skips form-associated custom elements; a global
  `htmx:configRequest` listener in `base.html` merges native `FormData` so any
  **named** `sl-input`/`sl-select` "just works". Don't add hidden mirror inputs.
- **`sl-button` is not a native submit** — trigger the form explicitly
  (`htmx.trigger(form, 'submit')`), and **delegate** click handlers so they
  survive `outerHTML` swaps. Copy the pattern in `settings.html`'s `<script>`
  (the `data-google-save` / `data-google-clear` handlers).
- **`hx-*` binds at process time, not submit time.** A per-row/per-id target URL
  can't live on a fixed `hx-post`; either render the URL into the element before
  `htmx.process`, or `fetch()` the endpoint directly (see the rename-modal note
  in CLAUDE.md).
- **SSE client:** reuse `openSSE` / `sseSwap` from `static/sse.js`. The job-status
  consumer in `base.html` (a `htmx:load` listener keyed off `[data-sse-session]`)
  is the model to follow.

## UI to build (transcript page — `templates/transcript.html`)

1. **Translate control.** A language picker (`sl-select` of target languages) +
   a "Translate" button that `POST`s `/sessions/<id>/translate`. Disable it (with
   a hint linking to Settings) when no API key is configured — surface this from
   the page context, e.g. expose `google_key.key_set` to `transcript.html` like
   `/settings` does.
2. **Live progress.** On trigger, `openSSE('/sessions/<id>/translate/events')`;
   show a spinner/stage chip per language; on `done` fetch and show the rendered
   translation, on `error` show the message.
3. **View switcher.** Tabs/segmented control to switch the transcript body
   between **Original** and each completed translation. Lazy-load each via
   `GET /sessions/<id>/translation/<lang>` into the body container, then
   `htmx.process`. Remember word-level highlight only applies to Original.
4. **Existing translations on load.** Read the `translations` map to render the
   available tabs/chips immediately (no need to re-translate).
5. **Downloads.** Per active translation, offer srt/vtt/txt/json links pointing
   at the download endpoint.

## Out of scope / notes

- Backend adds **no new dependencies** (stdlib `urllib`); no `bun` rebuild needed
  for `static/sse.js` (classic script).
- Only `google` is selectable today; the service dropdown is future-proofing.
- Tests for the backend live in `tests/test_translation.py` and
  `tests/test_translation_server.py`. If you add frontend unit tests for new
  `sse.js`-based helpers, follow `app/tests/sse.test.ts` (`cd app && bun test`).
