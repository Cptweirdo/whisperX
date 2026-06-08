<script lang="ts">
  import { session } from "../../lib/stores/session.svelte";
  import { fmtTs } from "../../lib/format";
  import SpeakerMenu from "./SpeakerMenu.svelte";
  import ReassignContextMenu from "./ReassignContextMenu.svelte";

  let {
    currentTime,
    onseek,
    onEditSpeaker,
  }: {
    currentTime: number;
    onseek: (t: number) => void;
    onEditSpeaker: (key: string, label: string) => void;
  } = $props();

  // Flatten turns into words with a global index + a start-sorted list for the
  // binary-search highlight (mirrors transcript.html's segs/highlight).
  const view = $derived.by(() => {
    let gi = 0;
    const timed: { gi: number; start: number }[] = [];
    const turns = session.viewTurns.map((t) => {
      const words = t.words.map((w) => {
        const item = { ...w, gi: gi++ };
        if (w.start != null) timed.push({ gi: item.gi, start: w.start });
        return item;
      });
      return { ...t, words };
    });
    timed.sort((a, b) => a.start - b.start);
    return { turns, timed };
  });

  const activeIdx = $derived.by(() => {
    const timed = view.timed;
    let lo = 0,
      hi = timed.length - 1,
      idx = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (timed[mid].start <= currentTime) {
        idx = mid;
        lo = mid + 1;
      } else hi = mid - 1;
    }
    return idx;
  });
  const activeGi = $derived(activeIdx >= 0 ? view.timed[activeIdx].gi : -1);
  const activeStart = $derived(activeIdx >= 0 ? view.timed[activeIdx].start : -1);

  function wordClass(w: { gi: number; start?: number }): string {
    if (w.start == null) return "seg";
    if (w.gi === activeGi) return "seg is-active";
    return w.start < activeStart ? "seg is-played" : "seg is-future";
  }

  // --- inline turn editing -------------------------------------------------
  let editing = $state<number | null>(null);
  let draft = $state("");
  let originalText = "";

  function enterEdit(t: { index: number; text: string }) {
    if (session.readonly || editing !== null) return;
    editing = t.index;
    draft = t.text;
    originalText = t.text;
  }
  async function saveEdit() {
    if (editing === null) return;
    const idx = editing;
    const text = draft.trim();
    editing = null;
    if (text !== originalText.trim()) await session.editTurn(idx, text);
  }
  function cancelEdit() {
    editing = null;
  }
  function focusEdit(node: HTMLElement) {
    node.focus();
    const r = document.createRange();
    r.selectNodeContents(node);
    r.collapse(false);
    const sel = window.getSelection();
    sel?.removeAllRanges();
    sel?.addRange(r);
  }

  // --- speaker reassign popover --------------------------------------------
  let openSwap = $state<number | null>(null);

  function onWindowClick(e: MouseEvent) {
    const el = e.target as HTMLElement;
    if (openSwap !== null && !el.closest?.(".swap-pick")) openSwap = null;
  }

  // --- edit mode: select a passage → right-click → reassign (3-way split) ----
  let ctx = $state<{ x: number; y: number; turnIndex: number; text: string; start: number; end: number } | null>(null);

  function onTurnContextMenu(e: MouseEvent, t: { index: number }) {
    if (!session.editing || editing !== null) return; // outside edit mode: native menu
    const container = e.currentTarget as HTMLElement;
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0 || sel.isCollapsed) return; // need a selection
    const range = sel.getRangeAt(0);
    if (!container.contains(range.startContainer) || !container.contains(range.endContainer)) return;
    const str = range.toString();
    if (!str.trim()) return;
    e.preventDefault();
    // char offset of the selection within the turn's word-joined text
    const pre = range.cloneRange();
    pre.selectNodeContents(container);
    pre.setEnd(range.startContainer, range.startOffset);
    const start = pre.toString().length;
    ctx = { x: e.clientX, y: e.clientY, turnIndex: t.index, text: str, start, end: start + str.length };
  }

  async function pickReassign(target: { speaker?: string; name?: string }) {
    if (!ctx) return;
    const { turnIndex, start, end } = ctx;
    ctx = null;
    window.getSelection()?.removeAllRanges();
    await session.splitReassign(turnIndex, { start, end }, target);
  }
</script>

<svelte:window onclick={onWindowClick} />

{#if view.turns.length === 0}
  <p class="tr__empty">No speech detected.</p>
{:else}
  {#each view.turns as t (t.index)}
    <div class="turn">
      <div class="turn__who">
        <div class="turn__speaker">{t.label}</div>
        {#if t.speaker}
          <button
            class="turn__edit"
            type="button"
            title="Edit speaker name"
            aria-label="Edit speaker name"
            onclick={() => onEditSpeaker(t.speaker!, t.label)}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z"/><path d="m15 5 4 4"/></svg>
          </button>
          <span class="swap-pick">
            <button
              class="turn__swap"
              type="button"
              title="Reassign speaker"
              aria-label="Reassign speaker"
              aria-haspopup="true"
              aria-expanded={openSwap === t.index}
              onclick={(e) => {
                e.stopPropagation();
                openSwap = openSwap === t.index ? null : t.index;
              }}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m16 3 4 4-4 4"/><path d="M20 7H4"/><path d="m8 21-4-4 4-4"/><path d="M4 17h16"/></svg>
            </button>
            {#if openSwap === t.index}
              <SpeakerMenu turn={t.index} current={t.speaker} onclose={() => (openSwap = null)} />
            {/if}
          </span>
        {/if}
        <div class="turn__time">{fmtTs(t.start)}</div>
      </div>

      {#if editing === t.index}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div
          class="turn__text is-editing"
          contenteditable="true"
          bind:textContent={draft}
          use:focusEdit
          onkeydown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              saveEdit();
            } else if (e.key === "Escape") {
              e.preventDefault();
              cancelEdit();
            }
          }}
          onblur={saveEdit}
        ></div>
      {:else}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div
          class="turn__text"
          class:is-selectable={session.editing}
          ondblclick={() => enterEdit(t)}
          oncontextmenu={(e) => onTurnContextMenu(e, t)}
        >
          {#each t.words as w (w.gi)}<!-- svelte-ignore a11y_no_noninteractive_tabindex --><span
              class={wordClass(w)}
              class:seg--untranslated={w.stale}
              onclick={() => !session.editing && w.start != null && onseek(w.start + 0.001)}
              role={!session.editing && w.start != null ? "button" : undefined}
              tabindex={!session.editing && w.start != null ? 0 : undefined}
              onkeydown={(e) => {
                if (!session.editing && w.start != null && (e.key === "Enter" || e.key === " ")) {
                  e.preventDefault();
                  onseek(w.start + 0.001);
                }
              }}>{w.text}</span
            >{" "}{/each}
        </div>
      {/if}
    </div>
  {/each}
{/if}

{#if ctx}
  <ReassignContextMenu
    x={ctx.x}
    y={ctx.y}
    text={ctx.text}
    onpick={pickReassign}
    onclose={() => (ctx = null)}
  />
{/if}
