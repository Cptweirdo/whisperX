<script lang="ts">
  let { onconfirm }: { onconfirm?: (id: string) => Promise<void> | void } = $props();

  let dialog = $state<any>(null);
  let id = $state("");
  let name = $state("");
  let busy = $state(false);

  export function open(rid: string, rname: string) {
    id = rid;
    name = rname || "this recording";
    dialog?.show();
  }

  async function confirm() {
    if (!id) return;
    busy = true;
    try {
      await onconfirm?.(id);
      dialog?.hide();
    } finally {
      busy = false;
    }
  }
</script>

<sl-dialog bind:this={dialog} label="Delete recording" style="--width:440px">
  <div>
    Delete <strong>{name}</strong>? This permanently removes the recording and its transcript.
  </div>
  <div slot="footer" style="display:flex;gap:14px;justify-content:flex-end">
    <sl-button variant="default" onclick={() => dialog?.hide()}>Cancel</sl-button>
    <sl-button variant="danger" loading={busy} onclick={confirm}>Delete</sl-button>
  </div>
</sl-dialog>
