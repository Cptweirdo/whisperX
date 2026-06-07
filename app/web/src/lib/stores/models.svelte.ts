// Global model-load state: full status from /api/models, kept live by the
// persistent /models/events stream. Drives the model <select> badges, the
// sidebar device chip, and the "loading models" → "ready" toasts (ported from
// base.html's models stream consumer).
import { api, urls } from "../api";
import { openSSE } from "../sse";
import { notify } from "./toast.svelte";
import { DEVICE_LABELS } from "../constants";

interface ModelMeta {
  name: string;
  loaded: boolean;
  loading: boolean;
  error: string | null;
}

class ModelsStore {
  status = $state<any>(null);
  modelsReady = $state(false);
  #es: EventSource | null = null;
  #loadingToast: HTMLElement | null = null;
  #sawLoading = false;

  get models(): ModelMeta[] {
    return this.status?.models ?? [];
  }
  get active(): string {
    return this.status?.active ?? "";
  }
  get device(): string {
    return this.status?.device ?? "cpu";
  }
  get deviceLabel(): string {
    return DEVICE_LABELS[this.device] ?? this.device;
  }

  /** A "model · cached/loading…/failed" label, matching the old select badges. */
  modelLabel(m: ModelMeta): string {
    const badge = m.loaded ? " · cached" : m.loading ? " · loading…" : m.error ? " · failed" : "";
    return m.name + badge;
  }

  async load() {
    this.status = await api.get("/models");
    this.modelsReady = this.#isReady();
  }

  /** Replace the full status (e.g. after a device switch returns it). */
  setStatus(status: any) {
    this.status = status;
    this.modelsReady = this.#isReady();
  }

  #isReady(): boolean {
    const a = this.models.find((m) => m.name === this.active);
    return !!(a && a.loaded);
  }

  async switchActive(model: string) {
    this.setStatus(await api.post("/models/active", { model }));
  }

  /** Switch device; throws ApiError (409 body carries `error:"busy"` + status). */
  async switchDevice(device: string) {
    this.setStatus(await api.post("/device", { device }));
  }

  start() {
    if (this.#es) return;
    this.#es = openSSE(urls.modelsEvents(), (d) => this.#onState(d));
  }

  #onState(d: any) {
    // Stream payload (_models_event) carries a subset; merge into full status.
    this.status = {
      ...(this.status ?? {}),
      models: d.models,
      active: d.active,
      diarize_available: d.diarize_available,
      diarize_error: d.diarize_error,
    };
    this.modelsReady = !!d.models_ready;

    if (d.bundle_error) return; // surfaced via its own dashboard toast
    if (!d.models_ready) {
      this.#sawLoading = true;
      if (!this.#loadingToast || !this.#loadingToast.isConnected) {
        this.#loadingToast = notify(
          "Loading models — your first upload will wait until they are ready.",
          "primary",
          Infinity,
        );
      }
      return;
    }
    if (this.#loadingToast?.isConnected) (this.#loadingToast as any).hide();
    this.#loadingToast = null;
    if (this.#sawLoading) {
      this.#sawLoading = false;
      notify("Models ready — you can start transcribing now.", "success", 5000);
    }
  }
}

export const models = new ModelsStore();
