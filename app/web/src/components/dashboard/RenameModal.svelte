<script lang="ts">
  import { api } from "../../lib/api";

  let { onsaved }: { onsaved?: (id: string, name: string) => void } = $props();

  let dialog = $state<any>(null);
  let input = $state<any>(null);
  let id = $state("");
  let name = $state("");
  let saving = $state(false);

  export function open(rid: string, rname: string) {
    id = rid;
    name = rname ?? "";
    dialog?.show();
    dialog?.updateComplete?.then(() => input?.focus());
  }

  async function save() {
    const n = name.trim();
    if (!n || !id) {
      input?.focus();
      return;
    }
    saving = true;
    try {
      const r = await api.post(`/sessions/${id}/rename`, { name: n });
      onsaved?.(id, r.filename);
      dialog?.hide();
    } finally {
      saving = false;
    }
  }
</script>

<sl-dialog bind:this={dialog} label="Rename recording" style="--width:440px">
  <sl-input
    bind:this={input}
    label="Recording name"
    placeholder="e.g. Board meeting"
    autocomplete="off"
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
