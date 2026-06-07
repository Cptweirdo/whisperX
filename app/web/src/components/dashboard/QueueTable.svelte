<script lang="ts">
  import { link } from "../../lib/router.svelte";
  import { sessions, type SessionCard } from "../../lib/stores/sessions.svelte";
  import { ui } from "../../lib/stores/ui.svelte";
  import { STAGE_LABELS } from "../../lib/constants";
  import RecEdit from "./RecEdit.svelte";

  let {
    onrename,
    ondelete,
  }: {
    onrename: (id: string, name: string) => void;
    ondelete: (id: string, name: string) => void;
  } = $props();

  type Filter = "all" | "pending" | "failed" | "done";
  let filter = $state<Filter>("all");

  const rows = $derived(
    filter === "all"
      ? sessions.list
      : sessions.list.filter((c) => sessions.bucket(c) === filter),
  );

  function stageLabel(c: SessionCard): string {
    const base = STAGE_LABELS[c.stage ?? ""] ?? "Queued";
    return base + "…";
  }
</script>

<div class="rec">
  <div class="rec__head">
    <span class="rec__h">Transcription Queue</span>
    <div class="rec__filters">
      {#each [["all", "All"], ["pending", "In Progress"], ["failed", "Failed"], ["done", "Done"]] as [key, label] (key)}
        <button
          class="rec__filter"
          class:rec__filter--on={filter === key}
          onclick={() => (filter = key as Filter)}
        >
          {label} <span class="n">{sessions.count(key as Filter)}</span>
        </button>
      {/each}
    </div>
  </div>

  {#each rows as c (c.id)}
    {@const bucket = sessions.bucket(c)}
    <div class="rec__row">
      <div
        class="rec__icon"
        class:rec__icon--fail={bucket === "failed"}
        class:rec__icon--done={bucket === "done"}
        class:rec__icon--run={bucket === "pending"}
      >
        {#if bucket === "pending"}
          <sl-icon name="arrow-clockwise" style="animation:spin 1s linear infinite"></sl-icon>
        {:else if bucket === "failed"}
          <sl-icon name="exclamation-octagon"></sl-icon>
        {:else}
          <sl-icon name="check2"></sl-icon>
        {/if}
      </div>

      <div class="rec__main">
        <div class="rec__name">
          <span class="rec-name">{c.name}</span>
          <RecEdit onclick={() => onrename(c.id, c.name)} />
        </div>
        <div class="rec__meta" class:rec__meta--fail={bucket === "failed"}>
          <span>{c.date}</span>
          <span class="sep">·</span>
          <span>{c.dur}</span>
          {#if bucket === "pending"}
            <span class="sep">·</span>
            <span>{stageLabel(c)}</span>
          {:else if bucket === "failed"}
            <span class="sep">·</span>
            <span>{c.error || "Unknown error"}</span>
          {:else}
            {#if c.language}<span class="sep">·</span><span>{c.language}</span>{/if}
            {#if c.num_segments}<span class="sep">·</span><span>{c.num_segments} segments</span>{/if}
          {/if}
        </div>
        {#if bucket === "pending"}
          <div class="rec__progress"><i style="width:40%"></i></div>
        {/if}
      </div>

      <div class="rec__right">
        {#if bucket === "pending"}
          <button class="rec__cancel" type="button" onclick={() => ondelete(c.id, c.name)}>
            Cancel
          </button>
        {:else if bucket === "failed"}
          <button class="older__del" type="button" title="Delete" onclick={() => ondelete(c.id, c.name)}>
            <sl-icon name="trash3"></sl-icon>
          </button>
          <button class="btn-outline is-fail" onclick={() => ui.requestNewRecording()}>Retry</button>
        {:else}
          <button class="older__del" type="button" title="Delete" onclick={() => ondelete(c.id, c.name)}>
            <sl-icon name="trash3"></sl-icon>
          </button>
          <a class="btn-outline" href={`/sessions/${c.id}`} use:link>Open Transcript</a>
        {/if}
      </div>
    </div>
  {:else}
    <div class="rec__empty">No recordings yet. Upload an audio file to get started.</div>
  {/each}
</div>
