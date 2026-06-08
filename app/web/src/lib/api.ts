// Typed client for the Flask JSON API. Endpoints are grouped into small classes
// (`api.sessions`, `api.models`, …) that expose one explicit, strongly-typed
// method per route — the raw HTTP verbs are private to this module so call sites
// can never hand-roll a path or response type. Response/body shapes live in
// ./types and mirror docs/api-reference.md (source of truth: app/server.py).
// Binary/SSE routes are at root and have their own URL helpers (`urls`) below.
import type {
  BackupMutationResult,
  BackupOverwriteResult,
  BackupStatus,
  CreateSessionResult,
  DeleteResult,
  DiarizeRefreshResult,
  FinishOnboardingResult,
  GoogleKeyResult,
  HfTokenResult,
  ModelStatus,
  OkResult,
  OnboardingData,
  OnboardingFinishBody,
  ReassignBody,
  RemoteInfoResult,
  RenameResult,
  SaveLanguageResult,
  SessionDetail,
  SessionsList,
  Settings,
  SpeakerEntry,
  TranscriptPayload,
  TranslateStartResult,
  TranslationView,
  VerifyTokenResult,
} from "./types";

export class ApiError extends Error {
  status: number;
  body: any;
  constructor(status: number, body: any) {
    super(typeof body?.error === "string" ? body.error : `HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}

async function parse(res: Response): Promise<any> {
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  const text = await res.text();
  return text ? { message: text } : {};
}

// --- Private transport: the only code that touches fetch/XHR ------------------
async function request<TResponse>(
  method: string,
  path: string,
  body?: unknown,
): Promise<TResponse> {
  const opts: RequestInit = { method, headers: {} };
  if (body !== undefined) {
    (opts.headers as Record<string, string>)["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(`/api${path}`, opts);
  const data = await parse(res);
  if (!res.ok) throw new ApiError(res.status, data);
  return data as TResponse;
}

/** Multipart upload via XHR so we get upload progress (fetch has none).
 *  Resolves with the parsed JSON body; rejects with ApiError on non-2xx. */
function upload<TResponse>(
  path: string,
  form: FormData,
  onProgress?: (fraction: number) => void,
): Promise<TResponse> {
  return new Promise<TResponse>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api${path}`);
    xhr.responseType = "text";
    if (onProgress && xhr.upload) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(e.loaded / e.total);
      };
    }
    xhr.onload = () => {
      let data: any = {};
      try {
        data = xhr.responseText ? JSON.parse(xhr.responseText) : {};
      } catch {
        data = { message: xhr.responseText };
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data as TResponse);
      else reject(new ApiError(xhr.status, data));
    };
    xhr.onerror = () => reject(new ApiError(0, { error: "network error" }));
    xhr.send(form);
  });
}

// --- Endpoint groups ---------------------------------------------------------
class SessionsApi {
  list() {
    return request<SessionsList>("GET", "/sessions");
  }
  get(id: string) {
    return request<SessionDetail>("GET", `/sessions/${id}`);
  }
  /** Upload a new recording (multipart); resolves with the new session id + status. */
  create(form: FormData, onProgress?: (fraction: number) => void) {
    return upload<CreateSessionResult>("/sessions", form, onProgress);
  }
  rename(id: string, name: string) {
    return request<RenameResult>("POST", `/sessions/${id}/rename`, { name });
  }
  remove(id: string) {
    return request<DeleteResult>("POST", `/sessions/${id}/delete`);
  }
  editTurn(id: string, index: number, text: string) {
    return request<TranscriptPayload>("POST", `/sessions/${id}/turns/${index}`, { text });
  }
  undo(id: string) {
    return request<TranscriptPayload>("POST", `/sessions/${id}/undo`);
  }
  speakers(id: string) {
    return request<SpeakerEntry[]>("GET", `/sessions/${id}/speakers`);
  }
  renameSpeaker(id: string, speaker: string, name: string) {
    return request<SpeakerEntry>("POST", `/sessions/${id}/speakers`, { speaker, name });
  }
  reassign(id: string, turn: number, body: ReassignBody) {
    return request<TranscriptPayload>("POST", `/sessions/${id}/turns/${turn}/speaker`, body);
  }
  translate(id: string, targetLanguage: string) {
    return request<TranslateStartResult>("POST", `/sessions/${id}/translate`, {
      target_language: targetLanguage,
    });
  }
  translation(id: string, lang: string) {
    return request<TranslationView>("GET", `/sessions/${id}/translation/${lang}`);
  }
}

class ModelsApi {
  status() {
    return request<ModelStatus>("GET", "/models");
  }
  setActive(model: string) {
    return request<ModelStatus>("POST", "/models/active", { model });
  }
  setDevice(device: string) {
    return request<ModelStatus>("POST", "/device", { device });
  }
}

class SettingsApi {
  get() {
    return request<Settings>("GET", "/settings");
  }
  saveLanguage(defaultLanguage: string) {
    return request<SaveLanguageResult>("POST", "/settings", { default_language: defaultLanguage });
  }
  setTranslationService(service: string) {
    return request<OkResult>("POST", "/settings/translation-service", {
      translation_service: service,
    });
  }
  saveHfToken(hfToken: string) {
    return request<HfTokenResult>("POST", "/settings/hf-token", { hf_token: hfToken });
  }
  clearHfToken() {
    return request<HfTokenResult>("POST", "/settings/hf-token/clear");
  }
  saveGoogleKey(googleKey: string) {
    return request<GoogleKeyResult>("POST", "/settings/google-key", { google_key: googleKey });
  }
  clearGoogleKey() {
    return request<GoogleKeyResult>("POST", "/settings/google-key/clear");
  }
  refreshDiarize() {
    return request<DiarizeRefreshResult>("POST", "/settings/diarize-model/refresh");
  }
}

class OnboardingApi {
  get() {
    return request<OnboardingData>("GET", "/onboarding");
  }
  verify(token: string) {
    return request<VerifyTokenResult>("POST", "/onboarding/verify", { token });
  }
  finish(body: OnboardingFinishBody) {
    return request<FinishOnboardingResult>("POST", "/onboarding", body);
  }
}

class BackupApi {
  status() {
    return request<BackupStatus>("GET", "/backup/status");
  }
  /** Start the background OAuth consent flow; watch /backup/events for the result. */
  connect(folder: string) {
    return request<unknown>("POST", "/backup/connect", { backup_folder: folder });
  }
  disconnect() {
    return request<BackupStatus>("POST", "/backup/disconnect");
  }
  now() {
    return request<BackupStatus>("POST", "/backup/now");
  }
  restore() {
    return request<BackupMutationResult>("POST", "/backup/restore");
  }
  adopt() {
    return request<BackupMutationResult>("POST", "/backup/bootstrap/adopt");
  }
  overwrite() {
    return request<BackupOverwriteResult>("POST", "/backup/bootstrap/overwrite");
  }
  remoteInfo() {
    return request<RemoteInfoResult>("GET", "/backup/remote-info");
  }
}

export const api = {
  sessions: new SessionsApi(),
  models: new ModelsApi(),
  settings: new SettingsApi(),
  onboarding: new OnboardingApi(),
  backup: new BackupApi(),
};

// --- Root (non-/api) URL helpers for the browser to hit directly ------------
export const urls = {
  audio: (id: string) => `/sessions/${id}/audio`,
  download: (id: string, fmt: string) => `/sessions/${id}/download/${fmt}`,
  exportMd: (id: string) => `/sessions/${id}/export.md`,
  translationDownload: (id: string, lang: string, fmt: string) =>
    `/sessions/${id}/translation/${lang}/download/${fmt}`,
  sessionEvents: (id: string) => `/sessions/${id}/events`,
  translateEvents: (id: string) => `/sessions/${id}/translate/events`,
  modelsEvents: () => `/models/events`,
  backupStatusEvents: () => `/backup/status/events`,
  backupEvents: () => `/backup/events`,
};
