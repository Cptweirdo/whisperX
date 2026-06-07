<script lang="ts">
  import { link } from "../../lib/router.svelte";
  import type { SessionCard } from "../../lib/stores/sessions.svelte";
  import RecEdit from "./RecEdit.svelte";

  let {
    card,
    onrename,
    ondelete,
  }: {
    card: SessionCard;
    onrename: (id: string, name: string) => void;
    ondelete: (id: string, name: string) => void;
  } = $props();
</script>

<article class="feature">
  <div class="feature__meta">
    <span class="chip {card.chip_class}">{card.chip_label}</span>
    <span class="feature__dur">{card.dur}</span>
    <button
      class="older__del"
      type="button"
      title="Delete"
      style="margin-left:auto"
      onclick={() => ondelete(card.id, card.name)}
    >
      <sl-icon name="trash3"></sl-icon>
    </button>
  </div>
  <h2 class="feature__title">
    <span class="rec-name">{card.name}</span>
    <RecEdit onclick={() => onrename(card.id, card.name)} />
  </h2>
  <p class="feature__desc">{card.sub}</p>
  <div class="feature__foot">
    <span class="feature__date">{card.date}</span>
    {#if card.viewable}
      <a class="btn-outline" href={`/sessions/${card.id}`} use:link>
        Open Transcript <sl-icon name="arrow-right"></sl-icon>
      </a>
    {:else}
      <span class="btn-outline" aria-disabled="true">
        {card.chip_label} <sl-icon name="hourglass-split"></sl-icon>
      </span>
    {/if}
  </div>
</article>
