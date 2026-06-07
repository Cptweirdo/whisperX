// Current transcript: row + server-grouped turns, edits/undo, speaker rename +
// reassign, and translation (live SSE). Ports transcript.html's controllers into
// reactive state. The audio player + word highlight live in components and read
// the `viewTurns`/`words` derived from here.
import { api, urls } from "../api";
import { openSSE } from "../sse";
import { notify } from "./toast.svelte";
import { settings } from "./settings.svelte";

export interface TurnWord {
  text: string;
  start?: number;
  end?: number;
  stale?: boolean;
}
export interface Turn {
  index: number;
  speaker: string | null;
  label: string;
  start: number | null;
  end: number | null;
  words: TurnWord[];
  text: string;
}

class SessionStore {
  id = $state("");
  row = $state<any>(null);
  turns = $state<Turn[]>([]);
  canUndo = $state(false);
  speakerNames = $state<Record<string, string>>({});
  translations = $state<Record<string, any>>({});
  formats = $state<string[]>([]);

  // Translation view: null = Original, else a language code with its own turns.
  activeLang = $state<string | null>(null);
  translationTurns = $state<Turn[]>([]);
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
  get googleKeySet(): boolean {
    return !!settings.data?.google_key?.key_set;
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
    const data = await api.get(`/sessions/${id}`);
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
  }

  // --- editing -------------------------------------------------------------
  #applyPayload(p: any) {
    this.turns = p.turns;
    this.canUndo = !!p.can_undo;
  }

  async editTurn(index: number, text: string) {
    try {
      this.#applyPayload(await api.post(`/sessions/${this.id}/turns/${index}`, { text }));
    } catch {
      notify("Could not save the edit.", "danger");
    }
  }

  async undo() {
    this.#applyPayload(await api.post(`/sessions/${this.id}/undo`));
  }

  async renameRecording(name: string) {
    const r = await api.post(`/sessions/${this.id}/rename`, { name });
    if (this.row) this.row.name = r.filename;
    return r.filename;
  }

  // --- speakers ------------------------------------------------------------
  speakers() {
    return api.get(`/sessions/${this.id}/speakers`);
  }

  async renameSpeaker(key: string, name: string) {
    const r = await api.post(`/sessions/${this.id}/speakers`, { speaker: key, name });
    if (name) this.speakerNames[key] = name;
    else delete this.speakerNames[key];
    // Patch labels in both views without a full reload.
    for (const list of [this.turns, this.translationTurns]) {
      for (const t of list) if (t.speaker === key) t.label = r.label;
    }
    return r.label;
  }

  async reassign(turn: number, params: { speaker?: string; name?: string }) {
    const r = await api.post(`/sessions/${this.id}/turns/${turn}/speaker`, params);
    // The POST re-rendered the Original; apply it there.
    this.turns = r.turns;
    this.canUndo = !!r.can_undo;
    // In a translation view, re-fetch so the new speaker shows with translated text.
    if (this.activeLang) await this.selectTranslation(this.activeLang);
  }

  // --- translation ---------------------------------------------------------
  async startTranslation(code: string) {
    const out = await api.post(`/sessions/${this.id}/translate`, { target_language: code });
    this.translations[code] = { status: out.status, service: out.service };
    this.activeLang = code; // auto-switch once SSE 'done' arrives
    this.#ensureStream();
    notify(`Translating into ${this.nativeOf(code)}…`, "primary");
  }

  async selectTranslation(code: string) {
    try {
      const data = await api.get(`/sessions/${this.id}/translation/${code}`);
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
