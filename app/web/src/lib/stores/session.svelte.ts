// Current transcript: row + server-grouped turns, edits/undo, speaker rename +
// reassign, and translation (live SSE). Ports transcript.html's controllers into
// reactive state. The audio player + word highlight live in components and read
// the `viewTurns`/`words` derived from here.
import { api, ApiError, urls } from "../api";
import { openSSE } from "../sse";
import { notify } from "./toast.svelte";
import { settings } from "./settings.svelte";
import type {
  ReassignBody,
  SessionDetail,
  TranscriptPayload,
  Translations,
  Turn,
} from "../types";

// Re-exported for the transcript components that render turns.
export type { Turn, TurnWord } from "../types";

class SessionStore {
  id = $state("");
  row = $state<SessionDetail | null>(null);
  turns = $state<Turn[]>([]);
  canUndo = $state(false);
  speakerNames = $state<Record<string, string>>({});
  translations = $state<Translations>({});
  formats = $state<string[]>([]);

  // Translation view: null = Original, else a language code with its own turns.
  activeLang = $state<string | null>(null);
  translationTurns = $state<Turn[]>([]);
  // Edit mode: gates the select→right-click→reassign passage flow (Original only).
  editMode = $state(false);
  #es: EventSource | null = null;

  get loaded() {
    return !!this.row;
  }
  get viewTurns(): Turn[] {
    return this.activeLang ? this.translationTurns : this.turns;
  }
  get readonly(): boolean {
    return this.activeLang !== null;
  }
  // Edit mode is meaningful only on the Original (editable) transcript.
  get editing(): boolean {
    return this.editMode && !this.readonly;
  }
  get googleKeySet(): boolean {
    return !!settings.data?.google_key?.key_set;
  }
  get translationServiceId(): string {
    return settings.data?.translation_service || "google";
  }
  get translationServiceLabel(): string {
    const svcs = settings.data?.translation_services ?? [];
    return svcs.find((s: any) => s.id === this.translationServiceId)?.label || "The translation service";
  }
  // Google (the only provider today) needs an API key; a future keyless
  // provider would be ready unconditionally.
  get translationReady(): boolean {
    return this.translationServiceId === "google" ? this.googleKeySet : true;
  }
  get translationDisabledReason(): string {
    return `${this.translationServiceLabel} needs an API key. Add one in Settings → Translation.`;
  }
  get translationLanguages(): { code: string; name: string; native: string }[] {
    return settings.data?.translation_languages ?? [];
  }
  get langNames(): Record<string, string> {
    return Object.fromEntries(this.translationLanguages.map((l) => [l.code, l.native]));
  }
  nativeOf(code: string): string {
    return this.langNames[code] || (code || "").toUpperCase();
  }
  get sourceLang(): string {
    return this.row?.language || "";
  }
  get sourceLabel(): string {
    return this.sourceLang ? this.nativeOf(this.sourceLang) : "Original";
  }
  get targetLanguages() {
    return this.translationLanguages.filter((l) => l.code !== this.sourceLang);
  }
  get isTranslating(): boolean {
    return Object.values(this.translations).some((t: any) => t && t.status === "running");
  }

  async load(id: string) {
    this.reset();
    this.id = id;
    const data = await api.sessions.get(id);
    this.row = data;
    this.turns = data.turns ?? [];
    this.canUndo = !!data.can_undo;
    this.speakerNames = data.speaker_names ?? {};
    this.translations = data.translations ?? {};
    this.formats = data.formats ?? [];
    if (this.isTranslating) this.#ensureStream();
  }

  reset() {
    this.#es?.close();
    this.#es = null;
    this.activeLang = null;
    this.translationTurns = [];
    this.turns = [];
    this.row = null;
    this.editMode = false;
  }

  toggleEditMode() {
    this.editMode = !this.editMode;
  }

  // --- editing -------------------------------------------------------------
  #applyPayload(p: TranscriptPayload) {
    this.turns = p.turns;
    this.canUndo = !!p.can_undo;
  }

  async editTurn(index: number, text: string) {
    try {
      this.#applyPayload(await api.sessions.editTurn(this.id, index, text));
    } catch {
      notify("Could not save the edit.", "danger");
    }
  }

  async undo() {
    this.#applyPayload(await api.sessions.undo(this.id));
  }

  async renameRecording(name: string) {
    const r = await api.sessions.rename(this.id, name);
    if (this.row) this.row.name = r.filename;
    return r.filename;
  }

  // --- speakers ------------------------------------------------------------
  speakers() {
    return api.sessions.speakers(this.id);
  }

  async renameSpeaker(key: string, name: string) {
    const r = await api.sessions.renameSpeaker(this.id, key, name);
    if (name) this.speakerNames[key] = name;
    else delete this.speakerNames[key];
    // Patch labels in both views without a full reload.
    for (const list of [this.turns, this.translationTurns]) {
      for (const t of list) if (t.speaker === key) t.label = r.label;
    }
    return r.label;
  }

  async reassign(turn: number, params: ReassignBody) {
    const r = await api.sessions.reassign(this.id, turn, params);
    // The POST re-rendered the Original; apply it there.
    this.turns = r.turns;
    this.canUndo = !!r.can_undo;
    // In a translation view, re-fetch so the new speaker shows with translated text.
    if (this.activeLang) await this.selectTranslation(this.activeLang);
  }

  /** Reassign a selected passage inside a turn (the edit-mode flow): the turn
   *  splits in three, the `[start,end)` middle moving to `target`. Waits for the
   *  server and applies its `TranscriptPayload`; no optimistic update. Failures
   *  surface as a toast and leave turns unchanged. */
  async splitReassign(
    turnIndex: number,
    sel: { start: number; end: number },
    target: { speaker?: string; name?: string },
  ) {
    if (sel.end <= sel.start) return;
    try {
      this.#applyPayload(
        await api.sessions.splitReassign(this.id, turnIndex, {
          start: sel.start,
          end: sel.end,
          speaker: target.speaker,
          name: target.name,
        }),
      );
      if (this.activeLang) await this.selectTranslation(this.activeLang);
    } catch (e) {
      const msg = e instanceof ApiError && typeof e.body?.error === "string"
        ? e.body.error
        : "Could not reassign the selection.";
      notify(msg, "danger");
    }
  }

  // --- translation ---------------------------------------------------------
  async startTranslation(code: string) {
    const out = await api.sessions.translate(this.id, code);
    this.translations[code] = { status: out.status, service: out.service };
    this.activeLang = code; // auto-switch once SSE 'done' arrives
    this.#ensureStream();
    notify(`Translating into ${this.nativeOf(code)}…`, "primary");
  }

  async selectTranslation(code: string) {
    try {
      const data = await api.sessions.translation(this.id, code);
      this.translationTurns = data.turns;
      this.activeLang = code;
    } catch {
      notify("Could not load that translation.", "danger");
    }
  }

  showOriginal() {
    this.activeLang = null;
    this.translationTurns = [];
  }

  downloadUrl(fmt: string): string {
    return this.activeLang
      ? urls.translationDownload(this.id, this.activeLang, fmt)
      : urls.download(this.id, fmt);
  }

  #ensureStream() {
    if (this.#es) return;
    this.#es = openSSE(urls.translateEvents(this.id), (d, src) => {
      if (d.translations) {
        this.translations = { ...this.translations, ...d.translations };
      } else if (d.lang) {
        const prev = this.translations[d.lang] || {};
        const next = { ...prev, ...d };
        delete next.lang;
        this.translations[d.lang] = next;
        if (d.status === "done") {
          notify(`${this.nativeOf(d.lang)} translation ready.`, "success");
          if (this.activeLang === d.lang) this.selectTranslation(d.lang);
        } else if (d.status === "error") {
          notify(`${this.nativeOf(d.lang)} translation failed.`, "danger");
          if (this.activeLang === d.lang) this.showOriginal();
        }
      }
      if (!this.isTranslating) {
        src.close();
        this.#es = null;
      }
    });
  }
}

export const session = new SessionStore();
