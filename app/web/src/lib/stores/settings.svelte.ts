// Settings + onboarding data. Wraps /api/settings and the token/key/diarize
// mutations; the model/device live state comes from the shared `models` store.
import { api } from "../api";
import { models } from "./models.svelte";

class SettingsStore {
  data = $state<any>(null);

  get onboarded(): boolean {
    return !!this.data?.onboarded;
  }

  async load() {
    this.data = await api.get("/settings");
    if (this.data?.models) models.setStatus(this.data.models);
    return this.data;
  }

  async saveLanguage(default_language: string) {
    await api.post("/settings", { default_language });
    if (this.data) this.data.default_language = default_language;
  }

  async setTranslationService(translation_service: string) {
    await api.post("/settings/translation-service", { translation_service });
    if (this.data) this.data.translation_service = translation_service;
  }

  async saveHfToken(hf_token: string) {
    const r = await api.post("/settings/hf-token", { hf_token });
    if (this.data) this.data.diarize = { ...this.data.diarize, token_set: r.token_set };
    return r;
  }
  async clearHfToken() {
    const r = await api.post("/settings/hf-token/clear");
    if (this.data) this.data.diarize = { ...this.data.diarize, token_set: r.token_set };
    return r;
  }

  async saveGoogleKey(google_key: string) {
    const r = await api.post("/settings/google-key", { google_key });
    if (this.data) this.data.google_key = { key_set: r.key_set };
    return r;
  }
  async clearGoogleKey() {
    const r = await api.post("/settings/google-key/clear");
    if (this.data) this.data.google_key = { key_set: r.key_set };
    return r;
  }

  async refreshDiarize() {
    const r = await api.post("/settings/diarize-model/refresh");
    if (this.data) {
      this.data.diarize = {
        version: r.version,
        model_name: r.model_name,
        token_set: r.token_set,
      };
    }
    return r;
  }
}

export const settings = new SettingsStore();
