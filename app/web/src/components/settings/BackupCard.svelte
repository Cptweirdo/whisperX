<script lang="ts">
  import { backup } from "../../lib/stores/backup.svelte";
  import { notify } from "../../lib/stores/toast.svelte";

  // `onboarding` trims the card to its essentials for the wizard step.
  let { onboarding = false }: { onboarding?: boolean } = $props();

  const s = $derived(backup.status);
  const state = $derived(s?.state ?? "disabled");
  let folder = $state("");
  let busy = $state("");
  let restoreModal: any = $state(null);
  let remote: any = $state(null);
  let remoteLoading = $state(false);

  $effect(() => {
    if (s?.folder != null && folder === "") folder = s.folder;
  });

  async function run(name: string, fn: () => Promise<any>, okMsg?: string) {
    busy = name;
    try {
      await fn();
      if (okMsg) notify(okMsg, "success");
    } catch (e: any) {
      notify(e?.message || "Backup action failed.", "danger");
    } finally {
      busy = "";
    }
  }

  async function openRestore() {
    remote = null;
    remoteLoading = true;
    restoreModal?.show();
    try {
      const r = await backup.remoteInfo();
      remote = r.remote;
    } catch {
      remote = null;
    } finally {
      remoteLoading = false;
    }
  }
</script>

<div id="backup-card">
  {#if state === "disabled"}
    <div class="pref__desc">
      Cloud backup isn't enabled on this server. Set
      <span class="ob__code">WHISPERX_BACKUP_BACKEND</span> (and the Google credentials for
      Drive) in your <span class="ob__code">.env</span>, then restart. See
      <span class="ob__code">app/.env.example</span> for the keys.
    </div>
  {:else if state === "conflict" || s?.remote?.exists}
    <div class="pref__title">A backup already exists on {s.provider_label}</div>
    <div class="pref__desc" style="margin-top:6px">
      {#if s.remote}
        {s.remote.entries} item{s.remote.entries === 1 ? "" : "s"}
        {#if s.remote.size_human}· {s.remote.size_human}{/if}
        {#if s.remote.created_at}· saved {s.remote.created_at}{/if}.
      {/if}
      Choose how to start:
    </div>
    <div class="backup__choice">
      <sl-button variant="primary" loading={busy === "adopt"}
        onclick={() => run("adopt", () => backup.adopt(), "Loaded the backup onto this device.")}>
        <sl-icon slot="prefix" name="cloud-download"></sl-icon> Load existing backup
      </sl-button>
      <sl-button variant="default" loading={busy === "overwrite"}
        onclick={() => run("overwrite", () => backup.overwrite(), "Started a fresh backup.")}>
        <sl-icon slot="prefix" name="cloud-upload"></sl-icon> Start fresh
      </sl-button>
    </div>
    <div class="pref__desc backup__warn">
      <strong>Load existing</strong> replaces the transcripts on this device with the backup.
      <strong>Start fresh</strong> overwrites the backup with this device's data.
    </div>
  {:else if s?.linked}
    <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions, a11y_no_noninteractive_tabindex -->
    <div
      class="backup__provider backup__provider--on"
      role={state === "backing_up" || state === "restoring" ? undefined : "button"}
      tabindex={state === "backing_up" || state === "restoring" ? undefined : 0}
      onclick={() => state !== "backing_up" && state !== "restoring" && openRestore()}
    >
      <div class="backup__ico">
        <sl-icon name={state === "error" ? "cloud-slash" : "cloud-check"}></sl-icon>
      </div>
      <div class="backup__meta">
        <div class="backup__name">{s.provider_label}</div>
        <div class="backup__sub">
          {#if state === "backing_up"}Syncing…
          {:else if state === "restoring"}Restoring…
          {:else if state === "error"}Last backup failed
          {:else if s.dirty}Changes not backed up yet
          {:else if s.last_human}Up to date · last backup {s.last_human}
          {:else}Up to date{/if}
          {#if s.folder} · folder <strong>{s.folder}</strong>{/if}
        </div>
      </div>
      {#if state === "backing_up" || state === "restoring"}
        <sl-spinner></sl-spinner>
      {:else if state === "error"}
        <span class="chip chip--err">● FAILED</span>
        <sl-icon class="backup__chev" name="chevron-right"></sl-icon>
      {:else}
        <span class="chip chip--ok">● CONNECTED</span>
        <sl-icon class="backup__chev" name="chevron-right"></sl-icon>
      {/if}
    </div>

    {#if state === "error" && s.last_error}
      <div class="frag frag--err" style="margin-top:10px">
        <sl-icon name="exclamation-triangle"></sl-icon> {s.last_error}
      </div>
    {/if}

    {#if state !== "backing_up" && state !== "restoring"}
      <div class="backup__actions">
        <sl-button size="small" variant="primary" loading={busy === "now"}
          onclick={() => run("now", () => backup.now())}>
          <sl-icon slot="prefix" name="cloud-arrow-up"></sl-icon> Back up now
        </sl-button>
        <sl-button size="small" variant="default" loading={busy === "disconnect"}
          onclick={() => run("disconnect", () => backup.disconnect())}>Disconnect</sl-button>
      </div>
    {/if}
  {:else}
    <div class="pref__desc" style="margin-bottom:14px">
      Keep your transcripts safe and pull them back onto any device by connecting a backup
      source. Everything stays local first — the backup is an extra copy you control.
    </div>
    <div class="backup__provider">
      <div class="backup__ico"><sl-icon name="cloud"></sl-icon></div>
      <div class="backup__meta">
        <div class="backup__name">{s?.provider_label ?? "Cloud backup"}</div>
        <div class="backup__sub">Not connected · transcripts stay on this device only</div>
      </div>
      <div>
        <sl-input
          size="small"
          label="Folder name"
          help-text="Where backups live in your Drive"
          style="margin-bottom:12px; max-width:320px"
          value={folder}
          onsl-input={(e: any) => (folder = e.target.value)}
        ></sl-input>
        <sl-button variant="primary" loading={busy === "connect" || backup.connecting}
          onclick={() => run("connect", () => backup.connect(folder))}>
          <sl-icon slot="prefix" name="cloud-upload"></sl-icon> Connect
        </sl-button>
      </div>
    </div>
  {/if}
</div>

{#if !onboarding}
  <sl-dialog bind:this={restoreModal} label={`Restore from ${s?.provider_label ?? "backup"}?`} style="--width:520px">
    <p class="modal__sub">This downloads the backup and makes this device an exact copy of it.</p>
    <div>
      {#if remoteLoading}
        <div class="frag"><sl-spinner></sl-spinner> Checking the backup…</div>
      {:else if remote?.exists}
        <div class="backup__detail">
          <div class="backup__detail-row"><span>Source</span><strong>{s.provider_label}</strong></div>
          {#if remote.created_at}<div class="backup__detail-row"><span>Last backup</span><strong>{remote.created_at}</strong></div>{/if}
          <div class="backup__detail-row"><span>Contents</span><strong>{remote.entries} item{remote.entries === 1 ? "" : "s"}</strong></div>
          {#if remote.size_human}<div class="backup__detail-row"><span>Download size</span><strong>{remote.size_human}</strong></div>{/if}
        </div>
      {:else}
        <div class="frag"><sl-icon name="info-circle"></sl-icon> This account has no backup yet — nothing to restore.</div>
      {/if}
    </div>
    <div class="frag--err backup__warn" style="margin-top:16px;font-size:14px;font-family:var(--mono)">
      <sl-icon name="exclamation-triangle"></sl-icon>
      Restore mirrors the backup: it <strong>replaces this device's transcripts</strong> and removes
      local recordings that aren't in the backup. This can't be undone.
    </div>
    <div slot="footer" style="display:flex;gap:14px;justify-content:flex-end;width:100%">
      <sl-button variant="default" onclick={() => restoreModal?.hide()}>Cancel</sl-button>
      <sl-button variant="danger" loading={busy === "restore"}
        onclick={() => run("restore", async () => { await backup.restore(); restoreModal?.hide(); }, "Restored from the backup.")}>
        <sl-icon slot="prefix" name="arrow-counterclockwise"></sl-icon> Restore now
      </sl-button>
    </div>
  </sl-dialog>
{/if}
