<script lang="ts">
  import { session } from "../../lib/stores/session.svelte";

  let {
    turn,
    current,
    onclose,
  }: { turn: number; current: string; onclose: () => void } = $props();

  let speakers = $state<{ key: string; label: string }[]>([]);
  let newName = $state("");
  let busy = $state(false);

  $effect(() => {
    session.speakers().then((s) => (speakers = s)).catch(() => {});
  });

  function initials(label: string): string {
    return (
      (label || "")
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, 2)
        .map((w) => w[0])
        .join("")
        .toUpperCase()
        .replace(/[^A-Z0-9]/g, "") || "?"
    );
  }

  async function pick(params: { speaker?: string; name?: string }) {
    if (busy) return;
    busy = true;
    try {
      await session.reassign(turn, params);
      onclose();
    } finally {
      busy = false;
    }
  }
</script>

<div class="spk-menu" role="menu">
  <div class="spk-menu__head">Reassign to</div>
  {#each speakers as sp (sp.key)}
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <div
      class="spk-opt"
      role="menuitem"
      tabindex="0"
      onclick={() => sp.key !== current && pick({ speaker: sp.key })}
    >
      <span class="spk-opt__badge">{initials(sp.label)}</span>
      <span class="spk-opt__name">{sp.label}</span>
      {#if sp.key === current}
        <span class="spk-opt__check"><sl-icon name="check-lg"></sl-icon></span>
      {/if}
    </div>
  {/each}
  <div class="spk-add">
    <input
      type="text"
      placeholder="New speaker…"
      autocomplete="off"
      bind:value={newName}
      onkeydown={(e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          if (newName.trim()) pick({ name: newName.trim() });
        } else if (e.key === "Escape") {
          e.preventDefault();
          onclose();
        }
      }}
    />
    <button type="button" disabled={!newName.trim()} onclick={() => pick({ name: newName.trim() })}>
      Add
    </button>
  </div>
</div>
