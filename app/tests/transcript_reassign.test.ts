// Reproduction + regression for the cross-view speaker desync bug.
//
// The transcript view caches the Original body HTML in a JS var (`origBody`) and
// restores it when you switch back from a translation. Reassigning a speaker
// *while viewing a translation* updates the server and the translation view, but
// historically never refreshed that cache — so switching back to the Original
// showed the stale (pre-reassign) speaker attribution. This drives the real
// inline `<script>` IIFEs from app/templates/transcript.html in happy-dom and
// asserts the Original reflects the reassignment after a round-trip.
//
// The IIFEs are inline (not modules), so — like app/tests/load-sse.ts — we read
// the template, pull out the two relevant blocks, and eval them against the DOM
// the setup preload installed.

import { test, expect, beforeEach } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const HTML = readFileSync(
  join(import.meta.dir, "../templates/transcript.html"),
  "utf8",
);
const BLOCKS = [...HTML.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
const TRANS_IIFE = BLOCKS.find((b) => b.includes("let origBody"))!;
const REASSIGN_IIFE = BLOCKS.find((b) => b.includes("Speaker reassignment"))!;

const tick = () => new Promise((r) => setTimeout(r, 0));
const flush = async () => { await tick(); await tick(); await tick(); };

// One diarized turn; `speaker`/`label` reflect the current server state.
function turnHTML(key: string, label: string, text: string) {
  return `
    <div class="turn" data-turn="0">
      <div class="turn__who">
        <div class="turn__speaker" data-speaker="${key}">${label}</div>
        <span class="swap-pick">
          <button class="turn__swap" type="button" data-turn="0" data-speaker="${key}"
                  aria-haspopup="true" aria-expanded="false">swap</button>
        </span>
      </div>
      <div class="turn__text">${text}</div>
    </div>`;
}

// The minimal DOM the translation IIFE wires itself to at boot.
function fixture() {
  document.body.innerHTML = `
    <script id="tx-data" type="application/json">${JSON.stringify({
      sessionId: "sess-1",
      sourceLabel: "Russian",
      langNames: { en: "English" },
      formats: ["txt"],
      translations: { en: { status: "done", service: "google" } },
      googleKeySet: true,
    })}</script>
    <button id="trans-chip" aria-expanded="false"><span id="trans-chip-label">Translate</span></button>
    <div id="lang-pick">
      <div id="lang-menu" hidden><div id="lang-menu-items"></div></div>
    </div>
    <div id="tx-note" hidden>English · <span id="tx-note-lang"></span></div>
    <div id="tx-downloads" hidden></div>
    <div id="add-translation-modal"></div>
    <button id="add-translation"></button>
    <button id="add-confirm"></button>
    <button id="add-cancel"></button>
    <div id="enroll-list"></div>
    <div id="tr-body">${turnHTML("SPEAKER_00", "Speaker 1", "original")}</div>`;
}

// Server speaker state the mock fetch reflects (a reassign mutates it).
let serverKey: string;
let serverLabel: string;

function installFetch() {
  serverKey = "SPEAKER_00";
  serverLabel = "Speaker 1";
  // @ts-expect-error override global fetch with a route-aware mock
  globalThis.fetch = async (url: string, opts?: any) => {
    if (url.includes("/translation/")) {
      // The translation render (structure/speaker come from the current original).
      return resp(turnHTML(serverKey, serverLabel, "translated"));
    }
    if (url.includes("/speakers")) {
      return json([
        { key: "SPEAKER_00", label: "Speaker 1" },
        { key: "SPEAKER_01", label: "Speaker 2" },
      ]);
    }
    if (/\/turns\/\d+\/speaker$/.test(url)) {
      const params = new URLSearchParams(opts?.body || "");
      serverKey = params.get("speaker") || serverKey;
      serverLabel = serverKey === "SPEAKER_01" ? "Speaker 2" : "Speaker 1";
      // _body_response renders the *Original* body — what origBody should become.
      return resp(turnHTML(serverKey, serverLabel, "original"));
    }
    throw new Error(`unexpected fetch: ${url}`);
  };
}

function resp(text: string) {
  return { ok: true, headers: { get: () => "1" }, text: async () => text, json: async () => ({}) };
}
function json(obj: any) {
  return { ok: true, headers: { get: () => "1" }, text: async () => JSON.stringify(obj), json: async () => obj };
}

function boot() {
  // Globals the IIFEs reference but that live elsewhere on the page.
  (globalThis as any).notify = () => {};
  (globalThis as any).openSSE = () => ({ close() {} });
  (window as any).__applyBodySwap = (_resp: any, html: string) => {
    document.getElementById("tr-body")!.innerHTML = html;
  };
  new Function(TRANS_IIFE)();
  new Function(REASSIGN_IIFE)();
}

function fire(el: Element) {
  el.dispatchEvent(new window.Event("click", { bubbles: true, cancelable: true }));
}

beforeEach(() => {
  fixture();
  installFetch();
  boot();
});

test("reassigning from a translation view updates the Original view too", async () => {
  const body = document.getElementById("tr-body")!;

  // Switch to the English translation (caches nothing new; origBody == Speaker 1).
  await (window as any).__reloadTranslation("en");
  expect(body.dataset.transLang).toBe("en");
  expect(body.textContent).toContain("Speaker 1");

  // Open the per-turn speaker picker and reassign to Speaker 2.
  fire(body.querySelector(".turn__swap")!);
  await flush(); // openMenu() awaits /speakers, then builds the menu
  const opt = [...body.querySelectorAll(".spk-opt")].find((o) =>
    o.textContent!.includes("Speaker 2"),
  )!;
  expect(opt).toBeTruthy();
  fire(opt);
  await flush(); // reassign() POSTs, then re-fetches the translation

  // Translation view reflects the reassignment (sanity: the edit went through).
  expect(body.textContent).toContain("Speaker 2");

  // Switch back to the Original via the language menu.
  fire(document.querySelector('#lang-menu-items [data-code=""]')!);
  await flush();

  // The bug: the Original shows the stale cached speaker. After the fix the
  // reassign refreshes origBody, so the Original reflects the new speaker.
  expect(body.dataset.transLang).toBeUndefined();
  expect(body.textContent).toContain("Speaker 2");
  expect(body.textContent).not.toContain("Speaker 1");
});
