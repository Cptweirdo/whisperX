<script lang="ts">
  import { session } from "../../lib/stores/session.svelte";

  let dialog = $state<any>(null);
  let input = $state<any>(null);
  let key = $state("");
  let name = $state("");
  let saving = $state(false);

  export function open(spkKey: string, label: string) {
    key = spkKey;
    name = label ?? "";
    dialog?.show();
    dialog?.updateComplete?.then(() => input?.focus());
  }

  async function save() {
    saving = true;
    try {
      await session.renameSpeaker(key, name.trim());
      dialog?.hide();
    } finally {
      saving = false;
    }
  }
</script>

<sl-dialog bind:this={dialog} label="Speaker name" style="--width:440px">
  <sl-input
    bind:this={input}
    label="Display name"
    placeholder="e.g. Alice"
    autocomplete="off"
    help-text="Leave empty to revert to the default label."
    value={name}
    onsl-input={(e: any) => (name = e.target.value)}
    onkeydown={(e: KeyboardEvent) => {
      if (e.key === "Enter") {
        e.preventDefault();
        save();
      }
    }}
  ></sl-input>
  <div slot="footer" style="display:flex;gap:14px;justify-content:flex-end">
    <sl-button variant="default" onclick={() => dialog?.hide()}>Cancel</sl-button>
    <sl-button variant="primary" loading={saving} onclick={save}>Save</sl-button>
  </div>
</sl-dialog>
