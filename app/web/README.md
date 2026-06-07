# Manuscript web client

The WhisperX web UI: a standalone **Bun + Vite + Svelte 5 (runes)** single-page
app that talks to the Flask JSON API in `app/server.py`.

## Develop

```bash
bun install
bun run dev      # Vite dev server on :5173, proxies /api + SSE to Flask :5000
```

Run the backend in another terminal: `python -m app.server` (binds :5000).
Open http://localhost:5173.

## Build

```bash
bun run build    # → ../static/spa  (served by Flask at / in production)
```

`bun run test` runs the Vitest unit tests; `bun run check` runs svelte-check.

## Layout

- `src/lib/api.ts` — typed fetch wrapper (`/api` base; XHR upload for progress).
- `src/lib/sse.ts` — `openSSE` / `sseStream` Server-Sent Events helpers.
- `src/lib/router.svelte.ts` — history-API router.
- `src/lib/stores/*.svelte.ts` — runes state (sessions, session, models, backup,
  settings, ui, toast).
- `src/routes/` — the five screens (Dashboard, Transcript, Settings, Onboarding).
- `src/components/` — UI building blocks (dashboard / transcript / settings).
- `src/app.css` — the Manuscript design system; `src/vendor.css` — Shoelace theme
  + self-hosted fonts; `src/shoelace.ts` — cherry-picked Shoelace components.

Shoelace is used directly as custom elements; bind values via `onsl-input` /
`onsl-change`. Built assets are served by Flask under `/static/spa/`, so reference
bundled public assets through `import.meta.env.BASE_URL`.
