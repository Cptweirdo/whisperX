// Global model-load state: full status from /api/models, kept live by the
// persistent /models/events stream. Drives the model <select> badges, the
// sidebar device chip, and the "loading models" → "ready" toasts (ported from
// base.html's models stream consumer).
import { api, urls } from "../api";
import { persistentSSE } from "../sse";
import { notify } from "./toast.svelte";
import { DEVICE_LABELS } from "../constants";
import type { ModelMeta, ModelStatus } from "../types";

class ModelsStore {
  status = $state<ModelStatus | null>(null);
  modelsReady = $state(false);
  #stop: (() => void) | null = null;
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
  get asrBackend(): string {
    return this.status?.asr_backend ?? "sherpa";
  }
  /** The single mutually-exclusive compute-target id shown in the picker:
   *  whisper.cpp owns Metal, otherwise the sherpa device (cpu/cuda/coreml). */
  get engine(): string {
    return this.asrBackend === "whispercpp" ? "whispercpp" : this.device;
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
    this.status = await api.models.status();
    this.modelsReady = this.#isReady();
  }

  /** Replace the full status (e.g. after a device switch returns it). */
  setStatus(status: ModelStatus) {
    this.status = status;
    this.modelsReady = this.#isReady();
  }

  #isReady(): boolean {
    const a = this.models.find((m) => m.name === this.active);
    return !!(a && a.loaded);
  }

  async switchActive(model: string) {
    this.setStatus(await api.models.setActive(model));
  }

  /** Switch device; throws ApiError (409 body carries `error:"busy"` + status). */
  async switchDevice(device: string) {
    this.setStatus(await api.models.setDevice(device));
  }

  /** Switch Stage-1 ASR backend (sherpa ↔ whispercpp); 409 like switchDevice. */
  async switchAsrBackend(backend: string) {
    this.setStatus(await api.models.setAsrBackend(backend));
  }

  /** Pick the unified engine target. whisper.cpp is its own backend (Metal);
   *  cpu/cuda/coreml mean the sherpa backend on that device. Dispatches to the
   *  right server axis so we never POST a backend id as a device. */
  async switchEngine(id: string) {
    if (id === "whispercpp") {
      await this.switchAsrBackend("whispercpp");
      return;
    }
    // Leaving whisper.cpp: revert to sherpa first, then set the device.
    if (this.asrBackend === "whispercpp") await this.switchAsrBackend("sherpa");
    await this.switchDevice(id);
  }

  start() {
    if (this.#stop) return;
    this.#stop = persistentSSE(urls.modelsEvents(), (d) => this.#onState(d));
  }

  #onState(d: any) {
    // Stream payload (_models_event) carries a subset; merge into full status.
    this.status = {
      ...(this.status ?? {}),
      models: d.models,
      active: d.active,
      diarize_available: d.diarize_available,
      diarize_error: d.diarize_error,
    } as ModelStatus;
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
