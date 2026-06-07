<script lang="ts">
  import { api, urls } from "../../lib/api";
  import { openSSE } from "../../lib/sse";
  import { router } from "../../lib/router.svelte";
  import { sessions } from "../../lib/stores/sessions.svelte";
  import { settings } from "../../lib/stores/settings.svelte";
  import { models } from "../../lib/stores/models.svelte";
  import { notify } from "../../lib/stores/toast.svelte";
  import { STAGE_LABELS, fmtEta } from "../../lib/constants";

  let dialog = $state<any>(null);
  let fileInput = $state<HTMLInputElement | null>(null);

  type Mode = "upload" | "processing" | "done" | "error";
  let mode = $state<Mode>("upload");
  let file = $state<File | null>(null);
  let language = $state("");
  let model = $state("");
  let name = $state("");
  let minSpeakers = $state("");
  let maxSpeakers = $state("");
  let progress = $state(0);
  let stageText = $state("Queued…");
  let errorMsg = $state("");
  let sessionId = $state("");
  let preview = $state<any>(null);
  let dragOver = $state(false);
  let es: EventSource | null = null;

  const languages = $derived(
    settings.data?.languages ?? [{ code: "", label: "Auto-detect" }],
  );

  export function open() {
    reset();
    language = settings.data?.default_language ?? "";
    model = models.active;
    dialog?.show();
  }

  function reset() {
    mode = "upload";
    file = null;
    name = "";
    minSpeakers = "";
    maxSpeakers = "";
    progress = 0;
    errorMsg = "";
    sessionId = "";
    preview = null;
    es?.close();
    es = null;
    if (fileInput) fileInput.value = "";
  }

  function pickFiles(files: FileList | null) {
    if (files && files.length) file = files[0];
  }

  async function start() {
    if (!file) {
      notify("Choose an audio file first.", "danger");
      return;
    }
    const form = new FormData();
    form.append("audio", file);
    if (name.trim()) form.append("name", name.trim());
    if (language) form.append("language", language);
    if (model) form.append("model", model);
    if (minSpeakers) form.append("min_speakers", minSpeakers);
    if (maxSpeakers) form.append("max_speakers", maxSpeakers);

    mode = "processing";
    progress = 0;
    stageText = "Uploading…";
    try {
      sessionId = await sessions.create(form, (f) => (progress = Math.round(f * 100)));
    } catch (e: any) {
      mode = "error";
      errorMsg =
        e?.status === 503
          ? "Models are still loading — try again in a moment."
          : e?.message || "Upload failed.";
      return;
    }
    stageText = "Queued…";
    watch(sessionId);
  }

  function watch(id: string) {
    es = openSSE(urls.sessionEvents(id), async (d, src) => {
      if (d.status === "done") {
        src.close();
        es = null;
        preview = await api.get(`/sessions/${id}`).catch(() => null);
        mode = "done";
        return;
      }
      if (d.status === "error") {
        src.close();
        es = null;
        mode = "error";
        errorMsg = "Transcription failed.";
        return;
      }
      if (d.stage && STAGE_LABELS[d.stage]) {
        let t = STAGE_LABELS[d.stage] + "…";
        if (d.eta) t += ` · ${fmtEta(d.eta)}`;
        stageText = t;
      }
    });
  }

  function viewFull() {
    dialog?.hide();
    router.navigate(`/sessions/${sessionId}`);
  }

  async function cancelClose() {
    if (mode === "processing" && sessionId) {
      await api.post(`/sessions/${sessionId}/delete`).catch(() => {});
      await sessions.load();
    }
    dialog?.hide();
  }
</script>

<sl-dialog
  bind:this={dialog}
  label={mode === "done" ? "Transcript" : "New Recording Session"}
  style={mode === "done" ? "--width:900px" : "--width:680px"}
  onsl-after-hide={reset}
>
  {#if mode === "upload"}
    <div class="modal__sub">Upload an audio file to generate a new manuscript draft.</div>

    <div class="field" style="display:flex;gap:18px">
      <div style="flex:1">
        <span class="field__label">Transcription Language</span>
        <sl-select value={language} placeholder="Auto-detect" hoist onsl-change={(e: any) => (language = e.target.value)}>
          {#each languages as l (l.code)}
            <sl-option value={l.code}>{l.label}</sl-option>
          {/each}
        </sl-select>
      </div>
      <div style="flex:1">
        <span class="field__label">Model</span>
        <sl-select value={model} hoist onsl-change={(e: any) => (model = e.target.value)}>
          {#each models.models as m (m.name)}
            <sl-option value={m.name}>{models.modelLabel(m)}</sl-option>
          {/each}
        </sl-select>
      </div>
    </div>

    <div class="field">
      <sl-input
        label="Name (optional)"
        placeholder="Defaults to the uploaded file name"
        value={name}
        onsl-input={(e: any) => (name = e.target.value)}
      ></sl-input>
    </div>

    <div class="field" style="display:flex;gap:18px">
      <sl-input type="number" min="1" label="Min speakers" placeholder="auto" style="flex:1"
        value={minSpeakers} onsl-input={(e: any) => (minSpeakers = e.target.value)}></sl-input>
      <sl-input type="number" min="1" label="Max speakers" placeholder="auto" style="flex:1"
        value={maxSpeakers} onsl-input={(e: any) => (maxSpeakers = e.target.value)}></sl-input>
    </div>

    <div class="field">
      <span class="field__label">Audio Source</span>
      <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
      <div
        class="dropzone"
        class:is-over={dragOver}
        onclick={() => fileInput?.click()}
        ondragenter={(e) => {
          e.preventDefault();
          dragOver = true;
        }}
        ondragover={(e) => e.preventDefault()}
        ondragleave={() => (dragOver = false)}
        ondrop={(e) => {
          e.preventDefault();
          dragOver = false;
          pickFiles(e.dataTransfer?.files ?? null);
        }}
      >
        <div class="dropzone__icon"><sl-icon name="file-earmark-arrow-up"></sl-icon></div>
        <div class="dropzone__title">Drag &amp; drop audio file here</div>
        <div class="dropzone__hint">or click to browse local files. Supports MP3, WAV, M4A &amp; video.</div>
        {#if file}<div class="dropzone__file">✓ {file.name}</div>{/if}
        <input
          bind:this={fileInput}
          type="file"
          accept="audio/*,video/*"
          hidden
          onchange={(e) => pickFiles((e.target as HTMLInputElement).files)}
        />
      </div>
    </div>
  {:else if mode === "processing"}
    <div style="padding:32px 0;text-align:center">
      <sl-spinner style="font-size:2rem"></sl-spinner>
      <p style="margin-top:16px">{stageText}</p>
      {#if progress > 0 && progress < 100}
        <div class="rec__progress" style="max-width:320px;margin:12px auto">
          <i style={`width:${progress}%`}></i>
        </div>
      {/if}
    </div>
  {:else if mode === "done"}
    <div class="tr-modal">
      <audio controls src={urls.audio(sessionId)} style="width:100%;margin-bottom:16px"></audio>
      <div class="tr__body" style="max-height:50vh;overflow:auto">
        {#each preview?.turns ?? [] as t (t.index)}
          <div class="turn">
            <div class="turn__who"><div class="turn__speaker">{t.label}</div></div>
            <div class="turn__text">{t.text}</div>
          </div>
        {/each}
      </div>
    </div>
  {:else}
    <div class="dialog-error" role="alert">{errorMsg}</div>
  {/if}

  <div slot="footer" style="display:flex;gap:14px;justify-content:flex-end;margin-top:22px">
    {#if mode === "done"}
      <sl-button variant="default" onclick={() => dialog?.hide()}>Back to recordings</sl-button>
      <sl-button variant="primary" onclick={viewFull}>
        Open full view <sl-icon slot="suffix" name="arrow-right"></sl-icon>
      </sl-button>
    {:else}
      <sl-button variant="default" onclick={cancelClose}>Cancel</sl-button>
      <sl-button variant="primary" loading={mode === "processing"} onclick={start} disabled={mode === "processing"}>
        <span>Start Processing</span> <sl-icon slot="suffix" name="arrow-right"></sl-icon>
      </sl-button>
    {/if}
  </div>
</sl-dialog>
