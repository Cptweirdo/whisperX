<script lang="ts">
  import { onMount } from "svelte";
  import { settings } from "../lib/stores/settings.svelte";
  import { models } from "../lib/stores/models.svelte";
  import { link } from "../lib/router.svelte";
  import { notify } from "../lib/stores/toast.svelte";
  import { DEVICE_LABELS } from "../lib/constants";
  import BackupCard from "../components/settings/BackupCard.svelte";

  const d = $derived(settings.data);

  let lang = $state("");
  let savingLang = $state(false);
  let hfValue = $state("");
  let hfBusy = $state("");
  let hfNotice = $state<{ msg: string; ok: boolean } | null>(null);
  let diarizeBusy = $state(false);
  let diarizeNotice = $state<{ msg: string; ok: boolean } | null>(null);
  let googleValue = $state("");
  let googleBusy = $state("");
  let googleNotice = $state<{ msg: string; ok: boolean } | null>(null);
  let service = $state("");
  let deviceNote = $state("");

  const DEVICES: { id: string; label: string }[] = [
    { id: "cpu", label: "CPU" },
    { id: "cuda", label: "GPU (CUDA)" },
    { id: "mlx", label: "Apple GPU (MLX)" },
    { id: "whispercpp", label: "whisper.cpp (Metal)" },
  ];
  function deviceAvailable(id: string): boolean {
    const m = models.status;
    if (id === "cpu") return true;
    if (id === "cuda") return !!m?.cuda_available;
    if (id === "mlx") return !!m?.mlx_available;
    if (id === "whispercpp") return !!m?.whispercpp_available;
    return false;
  }

  onMount(async () => {
    await settings.load();
    lang = settings.data?.default_language ?? "";
    service = settings.data?.translation_service ?? "";
  });

  async function saveLang() {
    savingLang = true;
    try {
      await settings.saveLanguage(lang);
      notify("Saved.", "success");
    } finally {
      savingLang = false;
    }
  }
  async function saveHf() {
    hfBusy = "save";
    try {
      const r = await settings.saveHfToken(hfValue);
      hfNotice = { msg: r.notice, ok: r.notice_ok };
      hfValue = "";
    } catch (e: any) {
      hfNotice = { msg: e?.body?.notice || e?.message || "Failed.", ok: false };
    } finally {
      hfBusy = "";
    }
  }
  async function clearHf() {
    hfBusy = "clear";
    try {
      const r = await settings.clearHfToken();
      hfNotice = { msg: r.notice, ok: r.notice_ok };
    } finally {
      hfBusy = "";
    }
  }
  async function refreshDiarize() {
    diarizeBusy = true;
    try {
      const r = await settings.refreshDiarize();
      diarizeNotice = { msg: r.notice, ok: r.notice_ok };
    } catch (e: any) {
      diarizeNotice = { msg: e?.body?.notice || e?.message || "Refresh failed.", ok: false };
    } finally {
      diarizeBusy = false;
    }
  }
  async function saveGoogle() {
    googleBusy = "save";
    try {
      const r = await settings.saveGoogleKey(googleValue);
      googleNotice = { msg: r.notice, ok: r.notice_ok };
      googleValue = "";
    } catch (e: any) {
      googleNotice = { msg: e?.body?.notice || e?.message || "Failed.", ok: false };
    } finally {
      googleBusy = "";
    }
  }
  async function clearGoogle() {
    googleBusy = "clear";
    try {
      const r = await settings.clearGoogleKey();
      googleNotice = { msg: r.notice, ok: r.notice_ok };
    } finally {
      googleBusy = "";
    }
  }
  async function saveService() {
    await settings.setTranslationService(service);
    notify("Saved.", "success");
  }
  async function switchDevice(id: string) {
    deviceNote = "";
    try {
      await models.switchDevice(id);
    } catch (e: any) {
      if (e?.status === 409) deviceNote = "Can't switch device while a transcription is queued or running.";
      else notify(e?.message || "Could not switch device.", "danger");
    }
  }
</script>

<div class="set">
  <h1 class="set__h">Settings</h1>
  <p class="set__sub">Manage your application preferences.</p>
  <div class="set__divider"></div>

  {#if !d}
    <div style="padding:40px"><sl-spinner style="font-size:1.6rem"></sl-spinner></div>
  {:else}
    <div class="set__panels">
      <section class="card">
        <h2 class="card__h">Transcription</h2>
        <div class="pref-row">
          <div>
            <div class="pref__title">Default Language</div>
            <div class="pref__desc">Pre-selected in the New Recording dialog. Auto-detect lets WhisperX infer the language per file (slightly slower).</div>
          </div>
          <div class="pref__control">
            <sl-select value={lang} placeholder="Auto-detect" hoist onsl-change={(e: any) => (lang = e.target.value)}>
              {#each d.languages as l (l.code)}<sl-option value={l.code}>{l.label}</sl-option>{/each}
            </sl-select>
          </div>
        </div>
        <div class="card__foot">
          <span></span>
          <sl-button variant="primary" loading={savingLang} onclick={saveLang}>Save Changes</sl-button>
        </div>
      </section>

      <section class="card">
        <h2 class="card__h">Hugging Face Access</h2>
        <div class="pref-row">
          <div>
            <div class="pref__title">Hugging Face Token</div>
            <div class="pref__desc">
              Optional. Speaker diarization works out of the box from the bundled model — a token
              only refreshes the gated pyannote models. Stored in your OS keyring. Status:
              <strong>{d.diarize.token_set ? "set" : "not set"}</strong>.
            </div>
          </div>
          <div class="pref__control">
            <sl-input type="password" password-toggle autocomplete="off" style="min-width:300px"
              placeholder={d.diarize.token_set ? "hf_•••• (stored)" : "hf_xxxxxxxx…"}
              value={hfValue} onsl-input={(e: any) => (hfValue = e.target.value)}></sl-input>
            <div style="display:flex;gap:10px;margin-top:10px;justify-content:flex-end">
              {#if d.diarize.token_set}
                <sl-button size="small" variant="default" loading={hfBusy === "clear"} onclick={clearHf}>Clear</sl-button>
              {/if}
              <sl-button size="small" variant="primary" loading={hfBusy === "save"} onclick={saveHf}>Verify &amp; Save</sl-button>
            </div>
          </div>
        </div>
        {#if hfNotice}
          <div class="frag {hfNotice.ok ? 'frag--ok' : 'frag--err'}" style="margin-top:8px;justify-content:flex-end">
            <sl-icon name={hfNotice.ok ? "check-circle" : "exclamation-triangle"}></sl-icon> {hfNotice.msg}
          </div>
        {/if}

        <div class="set__divider" style="margin:18px 0"></div>

        <div class="pref-row">
          <div>
            <div class="pref__title">Diarization Model</div>
            <div class="pref__desc">
              Speaker separation runs on <span class="ob__code">{d.diarize.model_name}</span>, bundled
              with Manuscript.
              {#if d.diarize.version}Current revision <strong>{d.diarize.version.sha8}</strong>
                ({d.diarize.version.source}{#if d.diarize.version.vendored_at}, {d.diarize.version.vendored_at}{/if}).{/if}
              Refreshing fetches the latest revision from Hugging Face.
            </div>
          </div>
          <div class="pref__control">
            <sl-button size="small" variant="default" disabled={!d.diarize.token_set} loading={diarizeBusy} onclick={refreshDiarize}>
              Refresh from Hugging Face
            </sl-button>
            {#if !d.diarize.token_set}
              <div class="pref__desc" style="margin-top:8px;text-align:right">Add a token above to enable updates.</div>
            {/if}
          </div>
        </div>
        {#if diarizeNotice}
          <div class="frag {diarizeNotice.ok ? 'frag--ok' : 'frag--err'}" style="margin-top:8px;justify-content:flex-end">
            <sl-icon name={diarizeNotice.ok ? "check-circle" : "exclamation-triangle"}></sl-icon> {diarizeNotice.msg}
          </div>
        {/if}
        <div class="pref__desc" style="margin-top:18px">
          Want to walk through setup again? <a class="ob__link" href="/onboarding" use:link>Re-run first-run setup</a>.
        </div>
      </section>

      <section class="card">
        <h2 class="card__h">Compute Device</h2>
        <div class="pref-row">
          <div>
            <div class="pref__title">Inference Device</div>
            <div class="pref__desc">Run transcription on CPU or GPU. Switching takes effect immediately, reloads all models, and is remembered. Blocked while a transcription is in progress.</div>
          </div>
          <div class="pref__control">
            <sl-select value={models.device} hoist onsl-change={(e: any) => switchDevice(e.target.value)}>
              {#each DEVICES as dev (dev.id)}
                <sl-option value={dev.id} disabled={!deviceAvailable(dev.id)}>
                  {dev.label}{!deviceAvailable(dev.id) ? " · unavailable" : ""}
                </sl-option>
              {/each}
            </sl-select>
            {#if deviceNote}<div class="pref__desc frag--err" style="margin-top:8px">{deviceNote}</div>{/if}
            <div class="pref__desc" style="margin-top:8px">
              Current device: <strong>{DEVICE_LABELS[models.device] ?? models.device}</strong>.
            </div>
          </div>
        </div>
      </section>

      <section class="card">
        <h2 class="card__h">Active Model</h2>
        <div class="pref-row">
          <div>
            <div class="pref__title">Default Whisper Model</div>
            <div class="pref__desc">The model new uploads use by default. Switching takes effect immediately and is remembered.</div>
          </div>
          <div class="pref__control">
            <sl-select value={models.active} hoist onsl-change={(e: any) => models.switchActive(e.target.value)}>
              {#each models.models as m (m.name)}<sl-option value={m.name}>{models.modelLabel(m)}</sl-option>{/each}
            </sl-select>
            <div class="pref__desc" style="margin-top:8px">
              Active model: <strong>{models.active}</strong>. Diarization {models.status?.diarize ? "enabled" : "disabled"}.
            </div>
          </div>
        </div>
      </section>

      <section class="card">
        <h2 class="card__h">Translation</h2>
        <div class="pref-row">
          <div>
            <div class="pref__title">Google Translation API Key</div>
            <div class="pref__desc">
              Required to translate transcripts with Google Translate. Stored in your OS keyring.
              Status: <strong>{d.google_key.key_set ? "set" : "not set"}</strong>.
            </div>
          </div>
          <div class="pref__control" style="min-width:300px">
            <sl-input type="password" password-toggle autocomplete="off" style="width:100%"
              placeholder={d.google_key.key_set ? "AIza•••• (stored)" : "AIzaSy…"}
              value={googleValue} onsl-input={(e: any) => (googleValue = e.target.value)}></sl-input>
            <div style="display:flex;gap:10px;margin-top:10px;justify-content:flex-end">
              {#if d.google_key.key_set}
                <sl-button size="small" variant="default" loading={googleBusy === "clear"} onclick={clearGoogle}>Clear</sl-button>
              {/if}
              <sl-button size="small" variant="primary" loading={googleBusy === "save"} onclick={saveGoogle}>Verify &amp; Save</sl-button>
            </div>
          </div>
        </div>
        {#if googleNotice}
          <div class="frag {googleNotice.ok ? 'frag--ok' : 'frag--err'}" style="margin-top:8px;justify-content:flex-end">
            <sl-icon name={googleNotice.ok ? "check-circle" : "exclamation-triangle"}></sl-icon> {googleNotice.msg}
          </div>
        {/if}
        <div class="set__divider" style="margin:18px 0"></div>
        <div class="pref-row">
          <div>
            <div class="pref__title">Translation Service</div>
            <div class="pref__desc">Which provider translates transcripts.</div>
          </div>
          <div class="pref__control">
            <sl-select value={service} style="min-width:220px" hoist onsl-change={(e: any) => (service = e.target.value)}>
              {#each d.translation_services as svc (svc.id)}<sl-option value={svc.id}>{svc.label}</sl-option>{/each}
            </sl-select>
          </div>
        </div>
        <div class="card__foot">
          <span></span>
          <sl-button variant="primary" onclick={saveService}>Save</sl-button>
        </div>
      </section>

      <section class="card">
        <h2 class="card__h">Backup &amp; Restore</h2>
        <BackupCard />
      </section>
    </div>
  {/if}
</div>
