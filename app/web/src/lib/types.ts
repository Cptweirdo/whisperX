// Typed contract for the Flask JSON API — the TypeScript mirror of the HTTP
// surface documented in docs/api-reference.md (source of truth: app/server.py).
// Request bodies and response shapes live here so api.ts stays pure transport and
// its grouped clients can be strongly typed: `api.settings.get()` → Settings.

// --- Enums (mirror pipeline.py constants) -----------------------------------
export type ModelName =
  | "tiny" | "tiny.en" | "base" | "base.en"
  | "small" | "small.en" | "medium" | "medium.en"
  | "large-v2" | "large-v3" | "large-v3-turbo" | "distil-large-v3";

export type Device = "cpu" | "cuda" | "coreml" | "mlx" | "whispercpp";
export type Format = "srt" | "vtt" | "txt" | "json";
export type Status = "queued" | "running" | "done" | "error";
export type Stage = "decoding" | "transcribing" | "loading_align" | "aligning" | "diarizing";

// --- Shared shapes ----------------------------------------------------------
export interface TranslationEntry {
  status: Status;
  service?: string;
  error?: string;
}
export type Translations = Record<string, TranslationEntry>;

/** Dashboard card / session header (server.py::_card). */
export interface SessionCard {
  id: string;
  name: string;
  chip_label: string;
  chip_class: string;
  viewable: boolean;
  dur: string;
  date: string;
  sub: string;
  model: ModelName | null;
  language: string | null;
  diarized: boolean;
  num_segments: number;
  status: Status;
  stage: Stage | null;
  error: string | null;
  translations: Translations;
}

/** Word inside a turn (server.py::_turn_words). */
export interface TurnWord {
  text: string;
  start?: number;
  end?: number;
  stale?: boolean;
}

/** Speaker-grouped turn the SPA renders (server.py::_build_turns). */
export interface Turn {
  index: number;
  speaker: string | null;
  label: string;
  start: number | null;
  end: number | null;
  words: TurnWord[];
  text: string;
}

/** Raw aligned segment (whisperx/schema.py::SingleAlignedSegment). */
export interface Segment {
  start: number;
  end: number;
  text: string;
  avg_logprob?: number;
  words?: { word: string; start?: number; end?: number; score?: number; speaker?: string }[];
  speaker?: string;
  stale?: boolean;
}

export interface ModelMeta {
  name: ModelName;
  loaded: boolean;
  loading: boolean;
  error: string | null;
}

/** Diarization revision info (diarize_model.derive_version). */
export interface DiarizeVersion {
  sha8: string;
  source: string;
  vendored_at?: string;
}

/** Whisper model-manager status (pipeline.ModelManager.status()). */
export interface ModelStatus {
  active: ModelName;
  device: Device;
  cuda_available: boolean;
  coreml_available: boolean;
  mlx_available: boolean;
  whispercpp_available: boolean;
  diarize: boolean;
  diarize_error: string | null;
  diarize_available: boolean;
  diarize_source: string;
  diarize_version: DiarizeVersion | null;
  diarize_token: boolean;
  models: ModelMeta[];
}

export interface RemoteState {
  exists: true;
  entries: number;
  total_size: number;
  size_human: string;
  created_at: string | null;
}

/** Backup card (server.py::_backup_json = BackupService.status() + view fields). */
export interface BackupStatus {
  state: string;
  linked: boolean;
  backend: string | null;
  dirty: boolean | null;
  last_root: string | null;
  last_backup_at: string | null;
  last_error: string | null;
  interval: number | null;
  provider_label: string;
  last_human: string | null;
  folder: string | null;
  remote: RemoteState | null;
  notice?: string | null;
  notice_ok?: boolean;
}

// --- Sessions ---------------------------------------------------------------
export interface Summary {
  count: number;
  transcribed: string;
  total_audio: string;
  pct: number;
}
export interface SessionsList {
  sessions: SessionCard[];
  summary: Summary;
}

/** Full session detail (GET /api/sessions/<id>): the card + transcript overlays. */
export interface SessionDetail extends SessionCard {
  result: ({ segments: Segment[] } & Record<string, unknown>) | null;
  turns?: Turn[];
  speaker_names: Record<string, string>;
  can_undo: boolean;
  created_at: string | null;
  updated_at: string | null;
  options: { language?: string | null; min_speakers?: number | null; max_speakers?: number | null };
  formats: Format[];
}

/** Shared shape returned by the edit / undo / reassign endpoints. */
export interface TranscriptPayload {
  turns: Turn[];
  segments: Segment[];
  can_undo: boolean;
}

export interface RenameResult {
  id: string;
  filename: string;
}
export interface DeleteResult {
  deleted: boolean;
}
export interface CreateSessionResult {
  id: string;
  status: Status;
}
export interface SpeakerEntry {
  key: string;
  label: string;
}

// --- Translation ------------------------------------------------------------
export interface TranslateStartResult {
  lang: string;
  status: Status;
  service: string;
}
export interface TranslationView {
  target_language: string;
  turns: Turn[];
  segments: Segment[];
}

// --- Onboarding -------------------------------------------------------------
export interface OnboardingSize {
  id: string;
  name: string;
  meta: string;
  note: string;
}
export interface OnboardingData {
  token: string;
  sizes: OnboardingSize[];
  selected_size: string;
  models: ModelStatus;
  diarize_model: string;
  backup: BackupStatus;
}
export interface VerifyTokenResult {
  ok: boolean;
  detail: string;
}
export interface FinishOnboardingResult {
  ok?: boolean;
  error?: string;
  store_error?: string;
}

// --- Settings ---------------------------------------------------------------
export interface TranscribeLanguage {
  code: string;
  label: string;
}
export interface TranslationLanguage {
  code: string;
  name: string;
  native: string;
}
export interface TranslationServiceOption {
  id: string;
  label: string;
}
export interface GoogleKeyInfo {
  key_set: boolean;
}
export interface DiarizeInfo {
  version: DiarizeVersion | null;
  model_name: string;
  token_set: boolean;
}
export interface Settings {
  default_language: string;
  languages: TranscribeLanguage[];
  models: ModelStatus;
  translation_service: string;
  translation_services: TranslationServiceOption[];
  translation_languages: TranslationLanguage[];
  google_key: GoogleKeyInfo;
  diarize: DiarizeInfo;
  backup: BackupStatus;
  onboarded: boolean;
}

export interface SaveLanguageResult {
  ok: boolean;
  default_language: string;
}
export interface OkResult {
  ok: boolean;
}
/** Shared payload for the HF-token endpoints. */
export interface HfTokenResult {
  token_set: boolean;
  notice: string;
  notice_ok: boolean;
}
/** Shared payload for the Google-key endpoints. */
export interface GoogleKeyResult {
  key_set: boolean;
  notice: string;
  notice_ok: boolean;
}
export interface DiarizeRefreshResult {
  version: DiarizeVersion | null;
  model_name: string;
  token_set: boolean;
  notice: string;
  notice_ok: boolean;
}

// --- Backup -----------------------------------------------------------------
export interface BackupMutationResult {
  restored: number;
  backup: BackupStatus;
}
export interface BackupOverwriteResult {
  uploaded: number;
  skipped: number;
  backup: BackupStatus;
}
export interface RemoteInfoResult {
  remote: RemoteState | null;
}

// --- Request bodies ---------------------------------------------------------
// Most endpoints take primitive args and build their body inside api.ts; only
// the multi-field bodies that travel as objects are typed here.
export interface ReassignBody {
  speaker?: string;
  name?: string;
}

/** Reassign a *selection* inside a turn to another speaker. `start`/`end` are
 *  character offsets into the turn's word-joined text (`Turn.words` joined by a
 *  single space); the turn splits in three — head + tail keep the original
 *  speaker, the `[start,end)` middle moves to `speaker`/`name`. Mirrors
 *  ReassignBody for the speaker target (existing key, or a `name` to mint one). */
export interface SplitReassignBody {
  start: number;
  end: number;
  speaker?: string;
  name?: string;
}
export interface OnboardingFinishBody {
  token: string;
  model: string;
  device: string;
}
