<script lang="ts">
  // Edit-mode right-click menu: preview the selected passage, then pick a speaker
  // to move it to. Phase "main" → the action list; phase "pick" → the speaker
  // dropdown. `onpick` reports the chosen target ({speaker?, name?, label}); the
  // store performs the 3-way turn split.
  import { session } from "../../lib/stores/session.svelte";
  import type { SpeakerEntry } from "../../lib/types";

  let {
    x,
    y,
    text,
    onpick,
    onclose,
  }: {
    x: number;
    y: number;
    text: string;
    onpick: (target: { speaker?: string; name?: string }) => void;
    onclose: () => void;
  } = $props();

  let phase = $state<"main" | "pick">("main");
  let speakers = $state<SpeakerEntry[]>([]);
  let newName = $state("");

  $effect(() => {
    session.speakers().then((s) => (speakers = s)).catch(() => {});
  });

  // Keep the menu on-screen (it is ~286px wide; clamp like the design).
  const left = $derived(Math.min(x, (typeof window !== "undefined" ? window.innerWidth : 1600) - 300));

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

  function copy() {
    if (text && navigator.clipboard) navigator.clipboard.writeText(text);
    onclose();
  }
  function addNew() {
    const nm = newName.trim();
    if (nm) onpick({ name: nm });
  }
</script>

<!-- scrim closes the menu; right-click on it just closes too -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  class="ctx-scrim"
  onclick={onclose}
  oncontextmenu={(e) => {
    e.preventDefault();
    onclose();
  }}
></div>

<div class="ctx-menu" style="left:{left}px;top:{y}px" role="menu">
  {#if phase === "main"}
    <div class="ctx-menu__preview">
      <div class="ctx-menu__plabel">Selected passage</div>
      <div class="ctx-menu__ptext">“{text}”</div>
    </div>
    <button class="ctx-item ctx-item--primary" type="button" onclick={() => (phase = "pick")}>
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m16 3 4 4-4 4"/><path d="M20 7H4"/><path d="m8 21-4-4 4-4"/><path d="M4 17h16"/></svg>
      Reassign to speaker
      <span class="ctx-item__chev"><sl-icon name="chevron-right"></sl-icon></span>
    </button>
    <button class="ctx-item" type="button" onclick={copy}>
      <sl-icon name="clipboard"></sl-icon> Copy text
    </button>
  {:else}
    <button class="ctx-menu__back" type="button" onclick={() => (phase = "main")}>
      <sl-icon name="chevron-left"></sl-icon> Assign selection to
    </button>
    {#each speakers as sp (sp.key)}
      <button class="ctx-item" type="button" onclick={() => onpick({ speaker: sp.key })}>
        <span class="spk-opt__badge">{initials(sp.label)}</span>
        <span class="spk-opt__name" style="font-size:14px">{sp.label}</span>
      </button>
    {/each}
    <div class="spk-add">
      <!-- svelte-ignore a11y_autofocus -->
      <input
        type="text"
        placeholder="New speaker…"
        autocomplete="off"
        autofocus
        bind:value={newName}
        onkeydown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            addNew();
          } else if (e.key === "Escape") {
            e.preventDefault();
            onclose();
          }
        }}
      />
      <button type="button" disabled={!newName.trim()} onclick={addNew}>Add</button>
    </div>
  {/if}
</div>
