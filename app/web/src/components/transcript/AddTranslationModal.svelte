<script lang="ts">
  import { session } from "../../lib/stores/session.svelte";
  import { notify } from "../../lib/stores/toast.svelte";
  import { link } from "../../lib/router.svelte";

  let dialog = $state<any>(null);
  let pick = $state<string | null>(null);
  let busy = $state(false);

  export function open() {
    pick = null;
    dialog?.show();
  }

  async function confirm() {
    if (!pick) {
      notify("Pick a language first.", "warning");
      return;
    }
    busy = true;
    try {
      await session.startTranslation(pick);
      dialog?.hide();
    } catch (e: any) {
      notify(e?.message || "Could not start translation.", "danger");
    } finally {
      busy = false;
    }
  }
</script>

<sl-dialog bind:this={dialog} label="Add a translation" style="--width:560px">
  {#if !session.googleKeySet}
    <p class="enroll-hint">
      Translation needs a Google Translation API key.
      <a href="/settings" use:link>Add one in Settings</a> to enable it.
    </p>
  {/if}
  <p class="enroll-hint">
    This recording is in <b>{session.sourceLabel}</b>. Pick a language to translate into —
    translations are stored separately and never change the original.
  </p>
  <div class="enroll-list">
    {#each session.targetLanguages as lang (lang.code)}
      {@const added = !!session.translations[lang.code]}
      <button
        class="enroll-row"
        class:is-added={added}
        class:enroll-row--sel={pick === lang.code}
        type="button"
        onclick={() => !added && (pick = lang.code)}
      >
        <span class="add-radio"></span>
        <div>
          <div class="enroll-row__native">{lang.native}</div>
          <div class="enroll-row__name">{lang.name}</div>
        </div>
        {#if added}<span class="enroll-row__tag">Added</span>{/if}
      </button>
    {/each}
  </div>
  <div slot="footer" style="display:flex;gap:14px;justify-content:flex-end">
    <sl-button variant="default" onclick={() => dialog?.hide()}>Cancel</sl-button>
    <sl-button variant="primary" loading={busy} disabled={!session.googleKeySet} onclick={confirm}>
      <sl-icon slot="prefix" name="translate"></sl-icon> Translate
    </sl-button>
  </div>
</sl-dialog>
