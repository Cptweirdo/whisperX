// Dashboard data: the recordings list + library summary, kept live by per-row
// job SSE streams (replacing index.html's location.reload() on completion).
import { api, urls } from "../api";
import { openSSE } from "../sse";

export interface SessionCard {
  id: string;
  name: string;
  status: string;
  stage: string | null;
  chip_label: string;
  chip_class: string;
  viewable: boolean;
  dur: string;
  date: string;
  sub: string;
  language: string | null;
  num_segments: number;
  error: string | null;
  [k: string]: any;
}

class SessionsStore {
  list = $state<SessionCard[]>([]);
  summary = $state<any>({ count: 0, transcribed: "0m", total_audio: "0m", pct: 0 });
  loaded = $state(false);
  #streams = new Map<string, EventSource>();

  get featured(): SessionCard | null {
    return this.list[0] ?? null;
  }
  count(status: "pending" | "failed" | "done" | "all"): number {
    if (status === "all") return this.list.length;
    return this.list.filter((c) => this.bucket(c) === status).length;
  }
  bucket(c: SessionCard): "pending" | "failed" | "done" {
    if (c.status === "queued" || c.status === "running") return "pending";
    if (c.status === "error") return "failed";
    return "done";
  }

  async load() {
    const data = await api.get("/sessions");
    this.list = data.sessions;
    this.summary = data.summary;
    this.loaded = true;
    this.#reconcileStreams();
  }

  async rename(id: string, name: string) {
    const res = await api.post(`/sessions/${id}/rename`, { name });
    const row = this.list.find((c) => c.id === id);
    if (row) row.name = res.filename;
    return res.filename;
  }

  async remove(id: string) {
    await api.post(`/sessions/${id}/delete`);
    this.#closeStream(id);
    // Reload so a deleted featured card promotes the next + summary shifts.
    await this.load();
  }

  /** Upload a new recording; resolves with the new session id. */
  async create(form: FormData, onProgress?: (f: number) => void): Promise<string> {
    const res = await api.upload("/sessions", form, onProgress);
    await this.load();
    return res.id;
  }

  #reconcileStreams() {
    const live = new Set<string>();
    for (const c of this.list) {
      if (c.status === "queued" || c.status === "running") {
        live.add(c.id);
        if (!this.#streams.has(c.id)) this.#open(c.id);
      }
    }
    for (const id of [...this.#streams.keys()]) {
      if (!live.has(id)) this.#closeStream(id);
    }
  }

  #open(id: string) {
    const es = openSSE(urls.sessionEvents(id), (d, src) => {
      if (d.status === "done" || d.status === "error") {
        src.close();
        this.#streams.delete(id);
        this.load(); // refresh row to terminal state (+ featured/summary)
        return;
      }
      if (d.stage) {
        const row = this.list.find((c) => c.id === id);
        if (row) row.stage = d.stage;
      }
    });
    this.#streams.set(id, es);
  }

  #closeStream(id: string) {
    this.#streams.get(id)?.close();
    this.#streams.delete(id);
  }
}

export const sessions = new SessionsStore();
